import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .runtime import GAMMA_FLOOR


def _is_torch_compiling() -> bool:
    # Prefer the public API when available; fall back to torch._dynamo for older versions.
    compiler_mod = getattr(torch, "compiler", None)
    if compiler_mod is not None:
        is_compiling_fn = getattr(compiler_mod, "is_compiling", None)
        if callable(is_compiling_fn):
            try:
                return bool(is_compiling_fn())
            except Exception:
                pass

    dynamo_mod = getattr(torch, "_dynamo", None)
    if dynamo_mod is not None:
        is_compiling_fn = getattr(dynamo_mod, "is_compiling", None)
        if callable(is_compiling_fn):
            try:
                return bool(is_compiling_fn())
            except Exception:
                pass

    return False


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.scale


class MLP(nn.Module):
    def __init__(self, d: int, mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Linear(d, d * mult)
        self.down = nn.Linear(d * mult, d)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(F.gelu(self.up(x))))


class K2Layer(nn.Module):
    """Core causal K2 mixing block with configurable gamma-decay backend."""

    def __init__(
        self,
        window: int,
        d: int,
        rank: int,
        mlp_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        alpha_cap: float = 1.0,
        decay_impl: str = "mask",
    ):
        super().__init__()
        self.window = window
        self.alpha_cap = alpha_cap
        if decay_impl not in ("mask", "block"):
            raise ValueError(f"Unknown decay_impl: {decay_impl}")
        self.decay_impl = decay_impl

        gamma_min = 0.95
        gamma_max = 0.995
        gammas = torch.linspace(gamma_min, gamma_max, rank)
        self.decay_logit = nn.Parameter(torch.log(gammas / (1.0 - gammas)))

        pos = torch.arange(window, dtype=torch.float)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(min=0)
        causal = torch.tril(torch.ones(window, window))
        k_init = torch.exp(-0.15 * dist) * causal
        k_init = k_init / k_init.sum(dim=1, keepdim=True).clamp(min=1e-8)

        self.k_base = nn.Parameter(k_init.contiguous())
        self.k_base_gate_logit = nn.Parameter(torch.tensor(8.0))
        self.register_buffer("causal_mask", causal)
        self.register_buffer("decay_diff", (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0).contiguous())
        self._causal_mask_smoke_checked = False

        self.u = nn.Parameter(torch.randn(d, rank) * 0.05)
        self.v = nn.Parameter(torch.randn(d, rank) * 0.05)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

        self.proj = nn.Linear(d, d)
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, dropout=mlp_dropout)
        self.drop = nn.Dropout(residual_dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, window, _ = h.shape
        if window > self.window:
            raise ValueError(f"Expected window <= {self.window}, got {window}")
        if (not self._causal_mask_smoke_checked) and (not _is_torch_compiling()):
            mask = self.causal_mask
            if mask.ndim != 2 or mask.size(0) != self.window or mask.size(1) != self.window:
                raise RuntimeError(
                    f"causal_mask shape mismatch: expected ({self.window}, {self.window}), got {tuple(mask.shape)}"
                )
            if not torch.equal(mask, torch.tril(mask)):
                raise RuntimeError("causal_mask must be lower triangular (triangular smoke test failed).")
            self._causal_mask_smoke_checked = True

        residual = h
        h_norm = self.norm1(h)

        gate_strength = torch.sigmoid(self.k_base_gate_logit)
        k_base = self.k_base[:window, :window] * self.causal_mask[:window, :window] * gate_strength
        out = torch.matmul(k_base, h_norm)

        q = F.normalize(h_norm @ self.u, dim=-1, eps=1e-8)
        k = F.normalize(h_norm @ self.v, dim=-1, eps=1e-8)

        raw = torch.sigmoid(self.decay_logit)
        gamma_vec = (GAMMA_FLOOR + (1.0 - GAMMA_FLOOR) * raw).to(dtype=h.dtype, device=h.device)
        alpha = (self.alpha_cap * torch.sigmoid(self.alpha_logit)).to(dtype=h.dtype, device=h.device)

        if self.decay_impl == "mask":
            kh = k.unsqueeze(-1) * h_norm.unsqueeze(-2)
            dist = self.decay_diff[:window, :window].to(dtype=h.dtype, device=h.device).unsqueeze(-1)
            log_gamma = torch.log(gamma_vec.clamp(min=1e-8)).view(1, 1, -1)
            decay_mask = torch.exp(dist * log_gamma) * self.causal_mask[:window, :window].unsqueeze(-1)
            causal_inner = torch.einsum("ijr,bjrd->bird", decay_mask, kh)
            out = out + alpha * torch.einsum("bwr,bwrd->bwd", q, causal_inner)
        else:
            gamma = gamma_vec.view(1, 1, -1, 1)
            state = torch.zeros(batch, self.u.size(1), h_norm.size(2), device=h.device, dtype=h.dtype)

            block = min(128, window)
            calc_dtype = torch.float32 if h.dtype in (torch.float16, torch.bfloat16) else h.dtype
            idx = torch.arange(block, device=h.device, dtype=calc_dtype).unsqueeze(1)
            log_gamma = torch.log(gamma_vec.to(dtype=calc_dtype).clamp(min=1e-8)).unsqueeze(0)
            pow_base = torch.exp(idx * log_gamma).to(dtype=h.dtype)
            inv_pow_base = torch.exp(-idx * log_gamma).to(dtype=h.dtype)

            for start in range(0, window, block):
                end = min(start + block, window)
                length = end - start

                h_blk = h_norm[:, start:end]
                q_blk = q[:, start:end]
                k_blk = k[:, start:end]

                pow_l = pow_base[:length].view(1, length, -1, 1)
                inv_pow_l = inv_pow_base[:length].view(1, length, -1, 1)

                kh_blk = k_blk.unsqueeze(-1) * h_blk.unsqueeze(-2)
                prefix = torch.cumsum(kh_blk * inv_pow_l, dim=1)
                state_blk = prefix * pow_l + state.unsqueeze(1) * (pow_l * gamma)

                out[:, start:end].add_(alpha * (q_blk.unsqueeze(-1) * state_blk).sum(dim=2))
                state = state_blk[:, -1]

        out = self.drop(self.proj(out))
        h = residual + out
        h = h + self.mlp(self.norm2(h))
        return h


class K1Layer(nn.Module):
    def __init__(self, d: int, mlp_dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.feature_mix = nn.Linear(d, d)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, dropout=mlp_dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        residual = h
        out = self.feature_mix(self.norm1(h))
        h = residual + out
        h = h + self.mlp(self.norm2(h))
        return h


class K0Layer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = RMSNorm(d)
        self.gain = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + (self.norm(h) * self.gain + self.bias)


class KStack(nn.Module):
    """Stacked backbone composed of K1/K2/K0 residual blocks."""

    def __init__(
        self,
        window: int,
        d: int,
        rank: int,
        n_k2: int,
        mlp_dropout: float,
        residual_dropout: float,
        alpha_cap: float = 1.0,
        decay_impl: str = "mask",
    ):
        super().__init__()
        layers: List[nn.Module] = [K1Layer(d, mlp_dropout=mlp_dropout)]
        for _ in range(n_k2):
            layers.append(
                K2Layer(
                    window,
                    d,
                    rank,
                    mlp_dropout=mlp_dropout,
                    residual_dropout=residual_dropout,
                    alpha_cap=alpha_cap,
                    decay_impl=decay_impl,
                )
            )
        layers.append(K1Layer(d, mlp_dropout=mlp_dropout))
        layers.append(K0Layer(d))
        self.layers = nn.ModuleList(layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h)
        return h


def resolve_adaptive_cutoffs(vocab_size: int, cutoffs: List[int] | None = None) -> List[int]:
    if vocab_size <= 1:
        raise ValueError(f"Adaptive softmax requires vocab_size > 1, got {vocab_size}.")

    if cutoffs is None:
        if vocab_size < 1024:
            raise ValueError(
                "Adaptive softmax is not useful for very small vocabularies. "
                "Use a larger tokenizer vocab or provide explicit --adaptive-cutoffs."
            )
        if vocab_size <= 4096:
            proposed = [max(256, vocab_size // 4), max(768, (3 * vocab_size) // 4)]
        elif vocab_size <= 16384:
            proposed = [2000, max(4000, vocab_size // 2)]
        else:
            proposed = [2000, 10000, max(20000, (3 * vocab_size) // 4)]
        cutoffs = proposed

    normalized = sorted({int(cutoff) for cutoff in cutoffs if 0 < int(cutoff) < vocab_size})
    if not normalized:
        raise ValueError(f"Adaptive softmax cutoffs must be within (0, vocab_size). Got {cutoffs} for vocab={vocab_size}.")
    return normalized


class KStackModel(nn.Module):
    """Token-level language model with iterative K-Stack refinement."""

    def __init__(
        self,
        vocab_size: int,
        window: int,
        d: int,
        emb_dim: int | None,
        rank: int,
        n_k2: int,
        emb_dropout: float,
        mlp_dropout: float,
        residual_dropout: float,
        head_mode: str = "linear",
        head_mult: int = 6,
        head_dropout: float = 0.0,
        adaptive_cutoffs: List[int] | None = None,
        adaptive_div_value: float = 4.0,
        refine_steps: int = 8,
        train_refine_steps: int | None = None,
        alpha_cap: float = 1.0,
        decay_impl: str = "mask",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d
        self.emb_dim = d if emb_dim is None else max(int(emb_dim), 1)
        self.head_mode = head_mode
        self.head_mult = head_mult

        self.emb = nn.Embedding(vocab_size, self.emb_dim)
        self.emb_to_model = nn.Identity() if self.emb_dim == d else nn.Linear(self.emb_dim, d, bias=False)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.k_stack = KStack(
            window,
            d,
            rank,
            n_k2,
            mlp_dropout,
            residual_dropout,
            alpha_cap=alpha_cap,
            decay_impl=decay_impl,
        )
        self.norm = RMSNorm(d)
        self.head_to_emb = nn.Identity()
        self.tie_weights = False
        self.adaptive_cutoffs = resolve_adaptive_cutoffs(vocab_size, adaptive_cutoffs) if head_mode == "adaptive" else []
        self.adaptive_div_value = float(adaptive_div_value)
        if head_mode == "linear":
            if self.emb_dim != d:
                self.head_to_emb = nn.Linear(d, self.emb_dim, bias=False)
            self.head = nn.Linear(self.emb_dim, vocab_size, bias=False)
            self.head.weight = self.emb.weight
            self.tie_weights = True
            self.head_drop = nn.Identity()
        elif head_mode == "gelu":
            hidden = max(d, head_mult * d)
            self.head = nn.Sequential(
                RMSNorm(d),
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Linear(hidden, vocab_size, bias=False),
            )
            self.head_drop = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        elif head_mode == "adaptive":
            self.head = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=d,
                n_classes=vocab_size,
                cutoffs=self.adaptive_cutoffs,
                div_value=self.adaptive_div_value,
                head_bias=False,
            )
            self.head_drop = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        else:
            raise ValueError(f"Unknown head_mode: {head_mode}")

        eta_init = 0.3
        eta_logit_init = math.log(eta_init / (1.0 - eta_init))
        self.eta_logit = nn.Parameter(torch.tensor(eta_logit_init))
        self.refine_steps = max(int(refine_steps), 0)
        if train_refine_steps is None:
            self.train_refine_steps = self.refine_steps
        else:
            self.train_refine_steps = max(int(train_refine_steps), 0)
        self._eval_refine_delta_sums: List[float] = [0.0 for _ in range(self.refine_steps)]
        self._eval_refine_batches = 0

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def eta(self) -> torch.Tensor:
        return torch.sigmoid(self.eta_logit)

    def reset_eval_refine_diagnostics(self) -> None:
        self._eval_refine_delta_sums = [0.0 for _ in range(self.refine_steps)]
        self._eval_refine_batches = 0

    def _record_eval_refine_diagnostics(self, deltas: List[float]) -> None:
        if len(deltas) != self.refine_steps:
            return
        for i, delta in enumerate(deltas):
            self._eval_refine_delta_sums[i] += float(delta)
        self._eval_refine_batches += 1

    def get_eval_refine_diagnostics(self) -> Dict[str, object]:
        eta_value = float(self.eta().detach().cpu().item())
        if self._eval_refine_batches == 0:
            return {"eta": eta_value, "delta_mean": [], "batches": 0}
        means = [v / self._eval_refine_batches for v in self._eval_refine_delta_sums]
        return {"eta": eta_value, "delta_mean": means, "batches": self._eval_refine_batches}

    def _forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h = self.emb_to_model(h)
        h = self.emb_drop(h)
        steps = self.train_refine_steps if self.training else self.refine_steps
        if steps == 0:
            # Disable iterative refinement: run one plain feedforward pass.
            h = self.k_stack(h)
        else:
            eta = self.eta().to(dtype=h.dtype, device=h.device)
            # Eval-only diagnostics are disabled while compiling to avoid graph breaks from scalar extraction.
            collect_eval_refine_diag = (not self.training) and (not _is_torch_compiling())
            eval_loop_deltas = [] if collect_eval_refine_diag else None
            for _ in range(steps):
                h_new = self.k_stack(h)
                if eval_loop_deltas is not None:
                    delta = (h_new - h).norm() / (h.norm() + 1e-6)
                    eval_loop_deltas.append(float(delta.detach().item()))
                h = h + eta * (h_new - h)
            if eval_loop_deltas is not None:
                self._record_eval_refine_diagnostics(eval_loop_deltas)
        h = self.norm(h)
        return h

    def _dense_scores_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        if self.head_mode == "linear":
            return self.head(self.head_to_emb(h))

        if self.head_mode == "gelu":
            h = self.head[0](h)
            h = self.head[1](h)
            h = self.head[2](h)
            h = self.head_drop(h)
            return self.head[3](h)

        if self.head_mode == "adaptive":
            head_in = self.head_drop(h).reshape(-1, h.size(-1))
            return self.head.log_prob(head_in).view(h.size(0), h.size(1), self.vocab_size)

        raise RuntimeError(f"Unsupported head_mode: {self.head_mode}")

    def _loss_from_hidden(self, h: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        flat_targets = targets.reshape(-1)
        if self.head_mode == "adaptive":
            head_in = self.head_drop(h).reshape(-1, h.size(-1))
            out = self.head(head_in, flat_targets)
            if reduction == "mean":
                return out.loss
            return out.loss * flat_targets.numel()

        scores = self._dense_scores_from_hidden(h)
        return F.cross_entropy(scores.reshape(-1, self.vocab_size), flat_targets, reduction=reduction)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        h = self._forward_hidden(x)
        if targets is not None:
            return self._loss_from_hidden(h, targets, reduction=reduction)
        return self._dense_scores_from_hidden(h)

    def count_params(self) -> Dict[str, int]:
        def _count_unique(params, skip_ptrs: set[int] | None = None) -> tuple[int, set[int]]:
            seen = set() if skip_ptrs is None else set(skip_ptrs)
            total_count = 0
            for p in params:
                if not p.requires_grad:
                    continue
                ptr = p.data_ptr()
                if ptr in seen:
                    continue
                seen.add(ptr)
                total_count += p.numel()
            return total_count, seen

        emb_params = list(self.emb.parameters()) + list(self.emb_to_model.parameters())
        head_params = list(self.head.parameters()) + list(self.head_to_emb.parameters())

        emb, emb_ptrs = _count_unique(emb_params)
        stack, emb_stack_ptrs = _count_unique(self.k_stack.parameters(), skip_ptrs=emb_ptrs)
        head, head_ptrs = _count_unique(head_params, skip_ptrs=emb_stack_ptrs)
        total, _ = _count_unique(self.parameters())
        other, _ = _count_unique(self.parameters(), skip_ptrs=head_ptrs)
        return {"total": total, "embedding": emb, "k_stack": stack, "head": head, "other": other}
