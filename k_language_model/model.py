import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .runtime import GAMMA_FLOOR


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


class KStackModel(nn.Module):
    """Character-level language model with iterative K-Stack refinement."""

    def __init__(
        self,
        vocab_size: int,
        window: int,
        d: int,
        rank: int,
        n_k2: int,
        emb_dropout: float,
        mlp_dropout: float,
        residual_dropout: float,
        head_mode: str = "linear",
        head_mult: int = 6,
        head_dropout: float = 0.0,
        refine_steps: int = 8,
        train_refine_steps: int | None = None,
        alpha_cap: float = 1.0,
        decay_impl: str = "mask",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.head_mode = head_mode
        self.head_mult = head_mult

        self.emb = nn.Embedding(vocab_size, d)
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
        if head_mode == "linear":
            self.head = nn.Linear(d, vocab_size, bias=False)
            self.head.weight = self.emb.weight
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb_drop(self.emb(x))
        steps = self.train_refine_steps if self.training else self.refine_steps
        if steps == 0:
            # Disable iterative refinement: run one plain feedforward pass.
            h = self.k_stack(h)
        else:
            eta = self.eta().to(dtype=h.dtype, device=h.device)
            loop_deltas = [] if not self.training else None
            for _ in range(steps):
                h_new = self.k_stack(h)
                if loop_deltas is not None:
                    delta = (h_new - h).norm() / (h.norm() + 1e-6)
                    loop_deltas.append(float(delta.detach().item()))
                h = h + eta * (h_new - h)
            if loop_deltas is not None:
                self._record_eval_refine_diagnostics(loop_deltas)
        h = self.norm(h)

        if self.head_mode == "gelu":
            h = self.head[0](h)
            h = self.head[1](h)
            h = self.head[2](h)
            h = self.head_drop(h)
            return self.head[3](h)

        return self.head(h)

    def count_params(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb = self.emb.weight.numel()
        stack = sum(p.numel() for p in self.k_stack.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return {"total": total, "embedding": emb, "k_stack": stack, "head": head}
