import math
import re
import warnings
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decay_kernel import decay_kernel, is_decay_kernel_available


LAYER_K_BASE_KEY_RE = re.compile(r"^k_stack\.layers\.(\d+)\.k_base$")

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
        k_base_rank: int = 0,
        k_base_impl: str = "auto",
        use_shared_k_base: bool = False,
        mlp_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        alpha_cap: float = 1.0,
        gamma_min: float = 0.15,
        gamma_max: float = 1.0,
        decay_impl: str = "mask",
    ):
        super().__init__()
        self.window = window
        self.alpha_cap = alpha_cap
        self.k_base_rank = max(int(k_base_rank), 0)
        self.use_shared_k_base = bool(use_shared_k_base)
        if k_base_impl not in ("auto", "fused", "scan"):
            raise ValueError(f"Unknown k_base_impl: {k_base_impl}")
        self.k_base_impl = k_base_impl
        if self.use_shared_k_base and self.k_base_rank > 0:
            raise ValueError("use_shared_k_base=True is supported only for dense k_base (k_base_rank <= 0).")
        if decay_impl not in ("mask", "block", "kernel"):
            raise ValueError(f"Unknown decay_impl: {decay_impl}")
        self.decay_impl = decay_impl

        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        if not (0.0 < self.gamma_min < self.gamma_max <= 1.0):
            raise ValueError(
                f"Expected 0 < gamma_min < gamma_max <= 1, got gamma_min={self.gamma_min}, gamma_max={self.gamma_max}."
            )

        init_low = max(self.gamma_min + 1e-4, 0.95)
        init_high = min(self.gamma_max - 1e-4, 0.995)
        if init_low >= init_high:
            mid = min(max((self.gamma_min + self.gamma_max) * 0.5, self.gamma_min + 1e-4), self.gamma_max - 1e-4)
            radius = 0.05 * (self.gamma_max - self.gamma_min)
            init_low = max(self.gamma_min + 1e-4, mid - radius)
            init_high = min(self.gamma_max - 1e-4, mid + radius)
            if init_low >= init_high:
                init_low = mid
                init_high = mid
        gammas = torch.linspace(init_low, init_high, rank)
        gamma_raw = (gammas - self.gamma_min) / (self.gamma_max - self.gamma_min)
        gamma_raw = gamma_raw.clamp(min=1e-6, max=1.0 - 1e-6)
        self.decay_logit = nn.Parameter(torch.log(gamma_raw / (1.0 - gamma_raw)))

        pos = torch.arange(window, dtype=torch.float)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(min=0)
        causal = torch.tril(torch.ones(window, window))
        k_init = torch.exp(-0.15 * dist) * causal
        k_init = k_init / k_init.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.k_base_gate_logit = nn.Parameter(torch.tensor(8.0))
        self.register_buffer("causal_mask", causal)
        self.register_buffer("decay_diff", (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0).contiguous())
        self._causal_mask_smoke_checked = False
        self._kernel_fallback_notified = False
        if self.k_base_rank > 0:
            self.w_k1 = nn.Linear(window, self.k_base_rank, bias=False)
            self.w_k2 = nn.Linear(self.k_base_rank, window, bias=False)
            self.register_buffer("k_base", k_init.contiguous())
            self._init_low_rank_k_base(k_init)
        else:
            if self.use_shared_k_base:
                self.register_parameter("k_base", None)
            else:
                self.k_base = nn.Parameter(k_init.contiguous())

        self.u = nn.Parameter(torch.randn(d, rank) * 0.05)
        self.v = nn.Parameter(torch.randn(d, rank) * 0.05)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

        self.proj = nn.Linear(d, d)
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, dropout=mlp_dropout)
        self.drop = nn.Dropout(residual_dropout)

    def _init_low_rank_k_base(self, k_base_dense: torch.Tensor) -> None:
        rank_eff = min(self.k_base_rank, k_base_dense.size(0), k_base_dense.size(1))
        with torch.no_grad():
            self.w_k1.weight.zero_()
            self.w_k2.weight.zero_()
            if rank_eff <= 0:
                return
            u, s, vh = torch.linalg.svd(k_base_dense, full_matrices=False)
            root_s = torch.sqrt(s[:rank_eff])
            left = u[:, :rank_eff] * root_s.unsqueeze(0)      # [W, R]
            right = vh[:rank_eff, :] * root_s.unsqueeze(1)    # [R, W]
            self.w_k2.weight[:, :rank_eff].copy_(left)
            self.w_k1.weight[:rank_eff, :].copy_(right)

    def _use_fused_k_base_path(
        self,
        batch: int,
        window: int,
        d_model: int,
        rank_eff: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> bool:
        if self.k_base_impl == "fused":
            return True
        if self.k_base_impl == "scan":
            return False
        # Auto mode: fused helps mostly on CUDA; scan is typically better on CPU/MPS.
        if device.type != "cuda":
            return False
        element_size = torch.empty((), dtype=dtype).element_size()
        temp_elems = int(batch) * int(window) * int(d_model) * int(rank_eff)
        est_temp_bytes = temp_elems * element_size * 2
        return est_temp_bytes <= 256 * 1024 * 1024

    def forward(self, h: torch.Tensor, shared_k_base: torch.Tensor | None = None) -> torch.Tensor:
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

        gate_strength = torch.sigmoid(self.k_base_gate_logit).to(dtype=h.dtype, device=h.device)
        if self.k_base_rank > 0:
            rank_eff = min(self.k_base_rank, self.w_k1.weight.size(0))
            if rank_eff <= 0:
                out = torch.zeros_like(h_norm)
            else:
                k_base_u = self.w_k2.weight[:window, :rank_eff].to(dtype=h.dtype, device=h.device).contiguous()  # [W, R]
                k_base_v = (
                    self.w_k1.weight[:rank_eff, :window]
                    .transpose(0, 1)
                    .to(dtype=h.dtype, device=h.device)
                    .contiguous()
                )  # [W, R]
                use_fused = self._use_fused_k_base_path(
                    batch=batch,
                    window=window,
                    d_model=h_norm.size(2),
                    rank_eff=rank_eff,
                    dtype=h.dtype,
                    device=h.device,
                )
                if use_fused:
                    base_terms = h_norm.unsqueeze(-2) * k_base_v.view(1, window, rank_eff, 1)
                    base_prefix = torch.cumsum(base_terms, dim=1)
                    out = (base_prefix * k_base_u.view(1, window, rank_eff, 1)).sum(dim=2)
                else:
                    # Scan path: avoids materializing [B, W, R, d] temporary tensor.
                    out = torch.zeros_like(h_norm)
                    for v_r, u_r in zip(k_base_v.unbind(dim=1), k_base_u.unbind(dim=1)):
                        v_r = v_r.view(1, window, 1)
                        u_r = u_r.view(1, window, 1)
                        prefix_r = torch.cumsum(h_norm * v_r, dim=1)
                        out.addcmul_(prefix_r, u_r)
                out.mul_(gate_strength)
        else:
            if self.use_shared_k_base:
                if shared_k_base is None:
                    raise RuntimeError("shared_k_base tensor is required when use_shared_k_base=True.")
                k_base_source = shared_k_base
            else:
                k_base_source = self.k_base
            k_base = k_base_source[:window, :window].to(dtype=h.dtype, device=h.device)
            k_base = k_base * self.causal_mask[:window, :window].to(dtype=h.dtype, device=h.device)
            out = gate_strength * torch.matmul(k_base, h_norm)

        # Compute q/k in one projection matmul to reduce GEMM overhead.
        qk = h_norm @ torch.cat((self.u, self.v), dim=1)
        q_raw, k_raw = qk.split(self.u.size(1), dim=-1)
        q = F.normalize(q_raw, dim=-1, eps=1e-8)
        k = F.normalize(k_raw, dim=-1, eps=1e-8)

        raw = torch.sigmoid(self.decay_logit)
        gamma_vec = (self.gamma_min + (self.gamma_max - self.gamma_min) * raw).to(dtype=h.dtype, device=h.device)
        alpha = (self.alpha_cap * torch.sigmoid(self.alpha_logit)).to(dtype=h.dtype, device=h.device)

        if self.decay_impl == "mask":
            kh = k.unsqueeze(-1) * h_norm.unsqueeze(-2)
            dist = self.decay_diff[:window, :window].to(dtype=h.dtype, device=h.device).unsqueeze(-1)
            log_gamma = torch.log(gamma_vec.clamp(min=1e-8)).view(1, 1, -1)
            decay_mask = torch.exp(dist * log_gamma) * self.causal_mask[:window, :window].unsqueeze(-1)
            causal_inner = torch.einsum("ijr,bjrd->bird", decay_mask, kh)
            out = out + alpha * torch.einsum("bwr,bwrd->bwd", q, causal_inner)
        elif self.decay_impl == "block":
            out = out + alpha * self._decay_block_term(q=q, k=k, h_norm=h_norm, gamma_vec=gamma_vec)
        else:
            can_use_kernel = is_decay_kernel_available(h.device, q.size(-1))
            if can_use_kernel:
                try:
                    out = out + alpha * decay_kernel(q=q, k=k, h=h_norm, gamma=gamma_vec.to(dtype=torch.float32))
                except Exception as exc:
                    if not self._kernel_fallback_notified:
                        warnings.warn(f"decay_impl='kernel' launch failed; fallback to block backend ({exc}).", stacklevel=2)
                        self._kernel_fallback_notified = True
                    out = out + alpha * self._decay_block_term(q=q, k=k, h_norm=h_norm, gamma_vec=gamma_vec)
            else:
                if not self._kernel_fallback_notified:
                    if h.device.type != "cuda":
                        reason = f"device={h.device.type}"
                    else:
                        reason = f"rank={q.size(-1)} not in supported range for kernel backend"
                    warnings.warn(f"decay_impl='kernel' fallback to block backend ({reason}).", stacklevel=2)
                    self._kernel_fallback_notified = True
                out = out + alpha * self._decay_block_term(q=q, k=k, h_norm=h_norm, gamma_vec=gamma_vec)

        out = self.drop(self.proj(out))
        h = residual + out
        h = h + self.mlp(self.norm2(h))
        return h

    def _decay_block_term(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        h_norm: torch.Tensor,
        gamma_vec: torch.Tensor,
    ) -> torch.Tensor:
        batch, window, _ = h_norm.shape
        decay_out = torch.zeros_like(h_norm)
        gamma = gamma_vec.view(1, 1, -1, 1)
        state = torch.zeros(batch, self.u.size(1), h_norm.size(2), device=h_norm.device, dtype=h_norm.dtype)

        block = min(128, window)
        calc_dtype = torch.float32 if h_norm.dtype in (torch.float16, torch.bfloat16) else h_norm.dtype
        idx = torch.arange(block, device=h_norm.device, dtype=calc_dtype).unsqueeze(1)
        log_gamma = torch.log(gamma_vec.to(dtype=calc_dtype).clamp(min=1e-8)).unsqueeze(0)
        pow_base = torch.exp(idx * log_gamma).to(dtype=h_norm.dtype)
        inv_pow_base = torch.exp(-idx * log_gamma).to(dtype=h_norm.dtype)

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

            decay_out[:, start:end].add_((q_blk.unsqueeze(-1) * state_blk).sum(dim=2))
            state = state_blk[:, -1]

        return decay_out


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
        k_base_rank: int = 2,
        k_base_impl: str = "auto",
        share_k_base: bool = False,
        alpha_cap: float = 1.0,
        gamma_min: float = 0.85,
        gamma_max: float = 1.0,
        decay_impl: str = "mask",
    ):
        super().__init__()
        self.share_k_base = bool(share_k_base)
        if self.share_k_base and int(k_base_rank) > 0:
            raise ValueError("share_k_base requires dense k_base (set k_base_rank <= 0).")

        if self.share_k_base:
            pos = torch.arange(window, dtype=torch.float)
            dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(min=0)
            causal = torch.tril(torch.ones(window, window))
            k_init = torch.exp(-0.15 * dist) * causal
            k_init = k_init / k_init.sum(dim=1, keepdim=True).clamp(min=1e-8)
            self.shared_k_base = nn.Parameter(k_init.contiguous())
        else:
            self.register_parameter("shared_k_base", None)

        layers: List[nn.Module] = [K1Layer(d, mlp_dropout=mlp_dropout)]
        for _ in range(n_k2):
            layers.append(
                K2Layer(
                    window,
                    d,
                    rank,
                    k_base_rank=k_base_rank,
                    k_base_impl=k_base_impl,
                    use_shared_k_base=self.share_k_base,
                    mlp_dropout=mlp_dropout,
                    residual_dropout=residual_dropout,
                    alpha_cap=alpha_cap,
                    gamma_min=gamma_min,
                    gamma_max=gamma_max,
                    decay_impl=decay_impl,
                )
            )
        layers.append(K1Layer(d, mlp_dropout=mlp_dropout))
        layers.append(K0Layer(d))
        self.layers = nn.ModuleList(layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        shared_k_base = self.shared_k_base
        for layer in self.layers:
            if isinstance(layer, K2Layer):
                h = layer(h, shared_k_base=shared_k_base)
            else:
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
        k_base_rank: int = 2,
        k_base_impl: str = "auto",
        share_k_base: bool = False,
        head_mode: str = "linear",
        head_mult: int = 6,
        head_dropout: float = 0.0,
        adaptive_cutoffs: List[int] | None = None,
        adaptive_div_value: float = 4.0,
        refine_steps: int = 8,
        train_refine_steps: int | None = None,
        alpha_cap: float = 1.0,
        gamma_min: float = 0.85,
        gamma_max: float = 1.0,
        decay_impl: str = "mask",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d
        self.emb_dim = d if emb_dim is None else max(int(emb_dim), 1)
        self.head_mode = head_mode
        self.head_mult = head_mult
        self.k_base_rank = max(int(k_base_rank), 0)
        self.k_base_impl = k_base_impl
        self.share_k_base = bool(share_k_base)

        self.emb = nn.Embedding(vocab_size, self.emb_dim)
        self.emb_to_model = nn.Identity() if self.emb_dim == d else nn.Linear(self.emb_dim, d, bias=False)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.k_stack = KStack(
            window=window,
            d=d,
            rank=rank,
            n_k2=n_k2,
            mlp_dropout=mlp_dropout,
            residual_dropout=residual_dropout,
            k_base_rank=self.k_base_rank,
            k_base_impl=self.k_base_impl,
            share_k_base=self.share_k_base,
            alpha_cap=alpha_cap,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
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
        self.bootstrap_low_rank_k_base()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def bootstrap_low_rank_k_base(self) -> None:
        for layer in self.k_stack.layers:
            if isinstance(layer, K2Layer) and layer.k_base_rank > 0 and hasattr(layer, "k_base"):
                layer._init_low_rank_k_base(layer.k_base.detach())

    def _k2_layer_indices(self) -> List[int]:
        return [idx for idx, layer in enumerate(self.k_stack.layers) if isinstance(layer, K2Layer)]

    def prepare_state_dict_for_load(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not isinstance(state, dict):
            return state
        if self.k_base_rank > 0:
            return state

        adapted = dict(state)
        shared_key = "k_stack.shared_k_base"
        layer_indices = self._k2_layer_indices()

        if self.share_k_base:
            legacy_items: List[tuple[int, torch.Tensor, float]] = []
            for key, tensor in adapted.items():
                match = LAYER_K_BASE_KEY_RE.match(key)
                if match is None or not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                    continue
                layer_idx = int(match.group(1))
                gate_key = f"k_stack.layers.{layer_idx}.k_base_gate_logit"
                gate_tensor = adapted.get(gate_key)
                if isinstance(gate_tensor, torch.Tensor) and gate_tensor.numel() == 1:
                    gate_weight = float(torch.sigmoid(gate_tensor.detach().float().cpu()).item())
                else:
                    gate_weight = 1.0
                legacy_items.append((layer_idx, tensor, gate_weight))

            if shared_key not in adapted and legacy_items:
                weight_sum = sum(max(weight, 0.0) for _, _, weight in legacy_items)
                if weight_sum <= 1e-8:
                    merged = torch.stack([tensor.detach().float() for _, tensor, _ in legacy_items], dim=0).mean(dim=0)
                else:
                    merged = torch.zeros_like(legacy_items[0][1], dtype=torch.float32)
                    for _, tensor, weight in legacy_items:
                        merged.add_(tensor.detach().to(dtype=torch.float32), alpha=max(weight, 0.0) / weight_sum)
                base_tensor = legacy_items[0][1]
                adapted[shared_key] = merged.to(dtype=base_tensor.dtype, device=base_tensor.device).contiguous()

            for layer_idx in layer_indices:
                adapted.pop(f"k_stack.layers.{layer_idx}.k_base", None)
            return adapted

        shared_tensor = adapted.pop(shared_key, None)
        if isinstance(shared_tensor, torch.Tensor):
            for layer_idx in layer_indices:
                layer_key = f"k_stack.layers.{layer_idx}.k_base"
                if layer_key not in adapted:
                    adapted[layer_key] = shared_tensor.clone()
        return adapted

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
