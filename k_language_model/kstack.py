from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decay import DecayBuffers, build_decay_impl
from .heads import build_head
from .kbase import DEFAULT_K_BASE_KERNEL_SIZE, build_kbase_impl, resize_kernel, resolve_kbase_impl_name
from .layers import K0Layer, K1Layer, MLP, RMSNorm
from .model_utils import (
    LAYER_ALPHA_LOGIT_KEY_RE,
    LAYER_K_BASE_KERNEL_KEY_RE,
    describe_k2_layer_mask,
    is_torch_compiling,
    resolve_adaptive_cutoffs,
    resolve_k2_layer_mask,
)
from .rosa_backends import build_rosa_backend


class K2Layer(nn.Module):
    """Core causal K2 mixing block with pluggable decay and k_base implementations."""

    def __init__(
        self,
        window: int,
        d: int,
        rank: int,
        k_base_rank: int = 0,
        k_base_impl: str = "conv",
        use_shared_k_base: bool = False,
        mlp_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        alpha_cap: float = 1.0,
        gamma_min: float = 0.15,
        gamma_max: float = 1.0,
        decay_impl: str = "mask",
        k_base_kernel_size: int = DEFAULT_K_BASE_KERNEL_SIZE,
        use_shared_decay: bool = False,
    ):
        super().__init__()
        self.window = int(window)
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError(f"Expected rank > 0, got {self.rank}.")
        self.alpha_cap = float(alpha_cap)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        if not (0.0 < self.gamma_min < self.gamma_max <= 1.0):
            raise ValueError(
                f"Expected 0 < gamma_min < gamma_max <= 1, got gamma_min={self.gamma_min}, gamma_max={self.gamma_max}."
            )
        self.k_base_rank = max(int(k_base_rank), 0)
        self.k_base_impl = resolve_kbase_impl_name(k_base_impl)
        self.k_base_kernel_size = max(int(k_base_kernel_size), 1)
        self.use_shared_k_base = bool(use_shared_k_base)
        self.use_shared_decay = bool(use_shared_decay)
        self.decay_impl = str(decay_impl).strip().lower()
        self.k_base_backend = build_kbase_impl(self.k_base_impl)
        if self.use_shared_decay:
            self.decay_backend = None
            self._causal_mask_smoke_checked = True
        else:
            self.decay_backend = build_decay_impl(self.decay_impl)
            self.decay_logit = nn.Parameter(torch.zeros(self.rank))

            pos = torch.arange(self.window, dtype=torch.float32)
            self.register_buffer("causal_mask", torch.tril(torch.ones(self.window, self.window)), persistent=False)
            self.register_buffer(
                "decay_diff",
                (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0).contiguous(),
                persistent=False,
            )
            self._causal_mask_smoke_checked = False

        self.k_base_gate_logit = nn.Parameter(torch.tensor(0.0))
        if self.use_shared_k_base:
            self.register_parameter("k_base_kernel", None)
        else:
            self.k_base_kernel = self.k_base_backend.default_parameter(self.k_base_kernel_size)

        if not self.use_shared_decay:
            self.u = nn.Parameter(torch.randn(d, self.rank) * 0.05)
            self.v = nn.Parameter(torch.randn(d, self.rank) * 0.05)
        self.alpha_logit = nn.Parameter(torch.zeros(self.rank))
        self.rho_logit = nn.Parameter(torch.tensor(-2.1972246))

        self.proj = nn.Linear(d, d)
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, dropout=mlp_dropout)
        self.drop = nn.Dropout(residual_dropout)

    def _resolve_k_base_kernel(self, shared_k_base: torch.Tensor | None, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.use_shared_k_base:
            if shared_k_base is None:
                raise RuntimeError("shared_k_base is required when use_shared_k_base=True.")
            kernel = shared_k_base
        else:
            if self.k_base_kernel is None:
                raise RuntimeError("Per-layer k_base_kernel parameter is missing.")
            kernel = self.k_base_kernel
        return kernel.to(dtype=dtype, device=device)

    def decay_gamma(self, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
        if not hasattr(self, "decay_logit"):
            raise RuntimeError("decay_gamma is unavailable when use_shared_decay=True.")
        gamma = torch.sigmoid(self.decay_logit).clamp(min=self.gamma_min, max=self.gamma_max)
        if dtype is not None or device is not None:
            gamma = gamma.to(dtype=dtype if dtype is not None else gamma.dtype, device=device if device is not None else gamma.device)
        return gamma

    def forward(
        self,
        h: torch.Tensor,
        shared_k_base: torch.Tensor | None = None,
        shared_decay_basis: torch.Tensor | None = None,
        rosa_h: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, window, _ = h.shape
        if window > self.window:
            raise ValueError(f"Expected window <= {self.window}, got {window}")
        if (not self.use_shared_decay) and (not self._causal_mask_smoke_checked) and (not is_torch_compiling()):
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
        k_base_kernel = self._resolve_k_base_kernel(shared_k_base, dtype=h.dtype, device=h.device)
        out = gate_strength * self.k_base_backend.compute(h_norm, k_base_kernel)

        alpha = (self.alpha_cap * torch.sigmoid(self.alpha_logit)).to(dtype=h.dtype, device=h.device)
        if self.use_shared_decay:
            if shared_decay_basis is None:
                raise RuntimeError("shared_decay_basis is required when use_shared_decay=True.")
            if (
                shared_decay_basis.ndim != 4
                or shared_decay_basis.size(0) != h.size(0)
                or shared_decay_basis.size(1) != h.size(1)
                or shared_decay_basis.size(2) != alpha.numel()
                or shared_decay_basis.size(3) != h.size(2)
            ):
                raise RuntimeError(
                    "shared_decay_basis shape mismatch: "
                    f"expected {(h.size(0), h.size(1), int(alpha.numel()), h.size(2))}, "
                    f"got {tuple(shared_decay_basis.shape)}"
                )
            decay_basis = shared_decay_basis.to(dtype=h.dtype, device=h.device)
            decay_out = (decay_basis * alpha.view(1, 1, -1, 1)).sum(dim=2)
        else:
            qk = h_norm @ torch.cat((self.u, self.v), dim=1)
            q_raw, k_raw = qk.split(self.u.size(1), dim=-1)
            q = F.normalize(q_raw, dim=-1, eps=1e-8)
            k = F.normalize(k_raw, dim=-1, eps=1e-8)
            gamma_vec = self.decay_gamma(dtype=h.dtype, device=h.device)
            q_alpha = q * alpha.view(1, 1, -1)
            decay_out = self.decay_backend.compute(
                q_alpha=q_alpha,
                k=k,
                h_norm=h_norm,
                gamma_vec=gamma_vec,
                buffers=DecayBuffers(causal_mask=self.causal_mask, decay_diff=self.decay_diff),
            )
        out = out + decay_out

        if rosa_h is not None:
            if rosa_h.shape != h.shape:
                raise ValueError(f"Expected rosa_h shape {tuple(h.shape)}, got {tuple(rosa_h.shape)}.")
            rho = torch.sigmoid(self.rho_logit).to(dtype=h.dtype, device=h.device)
            out = out + rho * rosa_h.to(dtype=h.dtype, device=h.device)

        out = self.drop(self.proj(out))
        h = residual + out
        h = h + self.mlp(self.norm2(h))
        return h


class KStack(nn.Module):
    def __init__(
        self,
        window: int,
        d: int,
        rank: int,
        n_k2: int,
        mlp_dropout: float,
        residual_dropout: float,
        k_base_rank: int = 0,
        k_base_impl: str = "conv",
        share_k_base: bool = False,
        k_base_kernel_size: int = DEFAULT_K_BASE_KERNEL_SIZE,
        alpha_cap: float = 1.0,
        gamma_min: float = 0.05,
        gamma_max: float = 1.0,
        decay_impl: str = "mask",
    ):
        super().__init__()
        self.window = int(window)
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError(f"Expected rank > 0, got {self.rank}.")
        self.share_k_base = bool(share_k_base)
        self.k_base_impl = resolve_kbase_impl_name(k_base_impl)
        self.k_base_kernel_size = max(int(k_base_kernel_size), 1)
        self.k_base_backend = build_kbase_impl(self.k_base_impl)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        if not (0.0 < self.gamma_min < self.gamma_max <= 1.0):
            raise ValueError(
                f"Expected 0 < gamma_min < gamma_max <= 1, got gamma_min={self.gamma_min}, gamma_max={self.gamma_max}."
            )
        self.decay_impl = str(decay_impl).strip().lower()
        self.decay_backend = build_decay_impl(self.decay_impl)
        self.decay_norm = RMSNorm(d)
        self.decay_u = nn.Parameter(torch.randn(d, self.rank) * 0.05)
        self.decay_v = nn.Parameter(torch.randn(d, self.rank) * 0.05)
        if self.rank == 1:
            decay_gamma_init = torch.tensor([0.5], dtype=torch.float32)
        else:
            decay_gamma_init = torch.linspace(0.2, 0.8, self.rank, dtype=torch.float32)
        decay_gamma_init = decay_gamma_init.clamp(min=1e-6, max=1.0 - 1e-6)
        self.decay_logit = nn.Parameter(torch.log(decay_gamma_init / (1.0 - decay_gamma_init)))
        pos = torch.arange(self.window, dtype=torch.float32)
        self.register_buffer("causal_mask", torch.tril(torch.ones(self.window, self.window)), persistent=False)
        self.register_buffer(
            "decay_diff",
            (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0).contiguous(),
            persistent=False,
        )
        self._causal_mask_smoke_checked = False
        if self.share_k_base:
            self.shared_k_base_kernel = self.k_base_backend.default_parameter(self.k_base_kernel_size)
        else:
            self.register_parameter("shared_k_base_kernel", None)
        self.n_k2_layers = int(n_k2)
        self.kappa_proj = nn.Linear(d, rank)
        nn.init.constant_(self.kappa_proj.bias, -1.0)

        layers: List[nn.Module] = [K1Layer(d, mlp_dropout=mlp_dropout)]
        for _ in range(n_k2):
            layers.append(
                K2Layer(
                    window=window,
                    d=d,
                    rank=rank,
                    k_base_rank=k_base_rank,
                    k_base_impl=self.k_base_impl,
                    use_shared_k_base=self.share_k_base,
                    mlp_dropout=mlp_dropout,
                    residual_dropout=residual_dropout,
                    alpha_cap=alpha_cap,
                    gamma_min=gamma_min,
                    gamma_max=gamma_max,
                    decay_impl=decay_impl,
                    k_base_kernel_size=self.k_base_kernel_size,
                    use_shared_decay=True,
                )
            )
        layers.append(K1Layer(d, mlp_dropout=mlp_dropout))
        layers.append(K0Layer(d))
        self.layers = nn.ModuleList(layers)

    def decay_gamma(self, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
        gamma = torch.sigmoid(self.decay_logit).clamp(min=self.gamma_min, max=self.gamma_max)
        if dtype is not None or device is not None:
            gamma = gamma.to(dtype=dtype if dtype is not None else gamma.dtype, device=device if device is not None else gamma.device)
        return gamma

    def _compute_shared_decay_rank_basis(self, h: torch.Tensor) -> torch.Tensor:
        _, window, _ = h.shape
        if window > self.window:
            raise ValueError(f"Expected window <= {self.window}, got {window}")
        if (not self._causal_mask_smoke_checked) and (not is_torch_compiling()):
            mask = self.causal_mask
            if mask.ndim != 2 or mask.size(0) != self.window or mask.size(1) != self.window:
                raise RuntimeError(
                    f"causal_mask shape mismatch: expected ({self.window}, {self.window}), got {tuple(mask.shape)}"
                )
            if not torch.equal(mask, torch.tril(mask)):
                raise RuntimeError("causal_mask must be lower triangular (triangular smoke test failed).")
            self._causal_mask_smoke_checked = True

        h_norm = self.decay_norm(h)
        kappa = torch.sigmoid(self.kappa_proj(h_norm))
        qk = h_norm @ torch.cat((self.decay_u, self.decay_v), dim=1)
        q_raw, k_raw = qk.split(self.rank, dim=-1)

        q_raw = q_raw * kappa
        k_raw = k_raw * kappa

        q = F.normalize(q_raw, dim=-1, eps=1e-8)
        k = F.normalize(k_raw, dim=-1, eps=1e-8)
        gamma_vec = self.decay_gamma(dtype=h.dtype, device=h.device)

        eye = torch.eye(self.rank, dtype=q.dtype, device=q.device)
        rank_values: List[torch.Tensor] = []
        for rank_idx in range(self.rank):
            q_rank = q * eye[rank_idx].view(1, 1, -1)
            out_rank = self.decay_backend.compute(
                q_alpha=q_rank,
                k=k,
                h_norm=h_norm,
                gamma_vec=gamma_vec,
                buffers=DecayBuffers(causal_mask=self.causal_mask, decay_diff=self.decay_diff),
            )
            rank_values.append(out_rank.unsqueeze(2))
        return torch.cat(rank_values, dim=2)

    def forward(
        self,
        h: torch.Tensor,
        rosa_h: torch.Tensor | None = None,
        rosa_layer_mask: List[bool] | None = None,
    ) -> torch.Tensor:
        if rosa_layer_mask is not None and len(rosa_layer_mask) != self.n_k2_layers:
            raise ValueError(f"Expected rosa_layer_mask length {self.n_k2_layers}, got {len(rosa_layer_mask)}.")

        k2_idx = 0
        shared_decay_basis: torch.Tensor | None = None
        for layer in self.layers:
            if isinstance(layer, K2Layer):
                use_rosa = rosa_h is not None and (rosa_layer_mask is None or bool(rosa_layer_mask[k2_idx]))
                if shared_decay_basis is None:
                    shared_decay_basis = self._compute_shared_decay_rank_basis(h)
                h = layer(
                    h,
                    shared_k_base=self.shared_k_base_kernel,
                    shared_decay_basis=shared_decay_basis,
                    rosa_h=rosa_h if use_rosa else None,
                )
                k2_idx += 1
            else:
                h = layer(h)
        return h


class KStackModel(nn.Module):
    """Token-level language model built around a single-pass K-Stack."""

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
        k_base_rank: int = 0,
        k_base_impl: str = "conv",
        share_k_base: bool = False,
        k_base_kernel_size: int = DEFAULT_K_BASE_KERNEL_SIZE,
        head_mode: str = "linear",
        head_mult: int = 6,
        head_dropout: float = 0.0,
        future_summary_horizons: List[int] | None = None,
        adaptive_cutoffs: List[int] | None = None,
        adaptive_div_value: float = 4.0,
        alpha_cap: float = 1.0,
        gamma_min: float = 0.05,
        gamma_max: float = 1.0,
        decay_impl: str = "mask",
        trajectory_aux: bool = False,
        rosa_impl: str = "exact",
        rosa_layers: str | None = "all",
    ):
        super().__init__()
        self.model_version = "v2"
        self.vocab_size = int(vocab_size)
        self.window = int(window)
        self.d_model = int(d)
        self.emb_dim = self.d_model if emb_dim is None else max(int(emb_dim), 1)
        self.head_mode = str(head_mode)
        self.head_mult = int(head_mult)
        self.future_summary_horizons = tuple(sorted({int(h) for h in (future_summary_horizons or []) if int(h) > 0}))
        self.k_base_rank = max(int(k_base_rank), 0)
        self.k_base_impl = resolve_kbase_impl_name(k_base_impl)
        self.k_base_kernel_size = max(int(k_base_kernel_size), 1)
        self.share_k_base = bool(share_k_base)
        self.decay_impl = str(decay_impl).strip().lower()
        self.trajectory_aux = bool(trajectory_aux) and self.head_mode != "trajectory"
        self.rosa_impl = str(rosa_impl).strip().lower()
        self.rosa_k2_layer_mask = resolve_k2_layer_mask(rosa_layers, n_k2, label="rosa_layers")

        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.emb_to_model = nn.Identity() if self.emb_dim == self.d_model else nn.Linear(self.emb_dim, self.d_model, bias=False)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.rosa_backend = build_rosa_backend(self.rosa_impl)
        self.rosa_emb = nn.Embedding(self.vocab_size + 1, self.emb_dim)
        self.rosa_to_model = nn.Identity() if self.emb_dim == self.d_model else nn.Linear(self.emb_dim, self.d_model, bias=False)
        self._set_rosa_projection_trainable(self.rosa_impl != "off" and any(self.rosa_k2_layer_mask))
        self.k_stack = KStack(
            window=self.window,
            d=self.d_model,
            rank=rank,
            n_k2=n_k2,
            mlp_dropout=mlp_dropout,
            residual_dropout=residual_dropout,
            k_base_rank=self.k_base_rank,
            k_base_impl=self.k_base_impl,
            share_k_base=self.share_k_base,
            k_base_kernel_size=self.k_base_kernel_size,
            alpha_cap=alpha_cap,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            decay_impl=self.decay_impl,
        )
        self.norm = RMSNorm(self.d_model)
        self.head = build_head(
            head_mode=self.head_mode,
            d_model=self.d_model,
            emb_dim=self.emb_dim,
            vocab_size=self.vocab_size,
            head_mult=self.head_mult,
            head_dropout=head_dropout,
            adaptive_cutoffs=resolve_adaptive_cutoffs(self.vocab_size, adaptive_cutoffs) if head_mode == "adaptive" else [],
            adaptive_div_value=adaptive_div_value,
            embedding=self.emb,
        )
        self.tie_weights = self.head.tie_weights
        self.adaptive_cutoffs = list(self.head.adaptive_cutoffs)
        self.adaptive_div_value = float(self.head.adaptive_div_value)
        if self.trajectory_aux:
            self.trajectory_aux_head = build_head(
                head_mode="trajectory",
                d_model=self.d_model,
                emb_dim=self.emb_dim,
                vocab_size=self.vocab_size,
                head_mult=self.head_mult,
                head_dropout=head_dropout,
                adaptive_cutoffs=[],
                adaptive_div_value=adaptive_div_value,
                embedding=self.emb,
            )
        else:
            self.trajectory_aux_head = None
        if self.future_summary_horizons:
            predictor_hidden = self.d_model * 2
            self.future_summary_norm = RMSNorm(self.d_model)
            self.future_summary_up = nn.Linear(self.d_model, predictor_hidden)
            self.future_summary_activation = nn.GELU()
            self.future_summary_heads = nn.ModuleDict(
                {str(h): nn.Linear(predictor_hidden, self.d_model) for h in self.future_summary_horizons}
            )
        else:
            self.future_summary_norm = None
            self.future_summary_up = None
            self.future_summary_activation = None
            self.future_summary_heads = nn.ModuleDict()

        self.apply(self._init_weights)
        if hasattr(self.k_stack, "kappa_proj"):
            nn.init.constant_(self.k_stack.kappa_proj.bias, -1.0)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _set_rosa_projection_trainable(self, enabled: bool) -> None:
        for p in self.rosa_emb.parameters():
            p.requires_grad_(enabled)
        for p in self.rosa_to_model.parameters():
            p.requires_grad_(enabled)

    def _k2_layer_indices(self) -> List[int]:
        return [idx for idx, layer in enumerate(self.k_stack.layers) if isinstance(layer, K2Layer)]

    def prepare_state_dict_for_load(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not isinstance(state, dict):
            return state

        adapted = dict(state)
        adapted.pop("eta_logit", None)
        for key in list(adapted.keys()):
            if key.endswith(".causal_mask") or key.endswith(".decay_diff"):
                adapted.pop(key, None)
        layer_indices = self._k2_layer_indices()
        shared_key = "k_stack.shared_k_base_kernel"

        if self.share_k_base:
            shared_tensor = adapted.get(shared_key)
            if isinstance(shared_tensor, torch.Tensor):
                adapted[shared_key] = resize_kernel(shared_tensor, self.k_base_kernel_size)
            for layer_idx in layer_indices:
                adapted.pop(f"k_stack.layers.{layer_idx}.k_base_kernel", None)
        else:
            shared_kernel = adapted.pop(shared_key, None)
            if isinstance(shared_kernel, torch.Tensor):
                shared_kernel = resize_kernel(shared_kernel, self.k_base_kernel_size)
            for layer_idx in layer_indices:
                layer_kernel_key = f"k_stack.layers.{layer_idx}.k_base_kernel"
                layer_kernel = adapted.get(layer_kernel_key)
                if isinstance(layer_kernel, torch.Tensor):
                    adapted[layer_kernel_key] = resize_kernel(layer_kernel, self.k_base_kernel_size)
                elif shared_kernel is not None:
                    adapted[layer_kernel_key] = shared_kernel.clone()

        def _resize_rank_vector(vec: torch.Tensor, target_rank: int) -> torch.Tensor:
            if vec.ndim == 0:
                return vec.expand(target_rank).clone()
            flat = vec.reshape(-1)
            if flat.numel() == target_rank:
                return flat.clone()
            if flat.numel() == 1:
                return flat.expand(target_rank).clone()
            out = flat.new_zeros(target_rank)
            copy = min(int(flat.numel()), int(target_rank))
            out[:copy] = flat[:copy]
            return out

        def _resize_rank_matrix(mat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
            if mat.ndim == 1:
                mat = mat.unsqueeze(0)
            out = mat.new_zeros((rows, cols))
            copy_rows = min(int(mat.size(0)), int(rows))
            copy_cols = min(int(mat.size(1)), int(cols))
            out[:copy_rows, :copy_cols] = mat[:copy_rows, :copy_cols]
            return out

        shared_decay_specs = (
            ("k_stack.decay_u", "u", self.k_stack.decay_u),
            ("k_stack.decay_v", "v", self.k_stack.decay_v),
            ("k_stack.decay_logit", "decay_logit", self.k_stack.decay_logit),
        )
        for shared_key_name, legacy_suffix, target_param in shared_decay_specs:
            shared_tensor = adapted.get(shared_key_name)
            if isinstance(shared_tensor, torch.Tensor):
                if target_param.ndim == 1:
                    adapted[shared_key_name] = _resize_rank_vector(shared_tensor, int(target_param.numel()))
                else:
                    target_rows, target_cols = int(target_param.size(0)), int(target_param.size(1))
                    adapted[shared_key_name] = _resize_rank_matrix(shared_tensor, target_rows, target_cols)
            else:
                legacy_tensors: List[torch.Tensor] = []
                for layer_idx in layer_indices:
                    legacy_key = f"k_stack.layers.{layer_idx}.{legacy_suffix}"
                    tensor = adapted.pop(legacy_key, None)
                    if isinstance(tensor, torch.Tensor):
                        legacy_tensors.append(tensor)
                if legacy_tensors:
                    if target_param.ndim == 1:
                        resized = [_resize_rank_vector(tensor, int(target_param.numel())) for tensor in legacy_tensors]
                    else:
                        target_rows, target_cols = int(target_param.size(0)), int(target_param.size(1))
                        resized = [_resize_rank_matrix(tensor, target_rows, target_cols) for tensor in legacy_tensors]
                    adapted[shared_key_name] = torch.stack(resized, dim=0).mean(dim=0)
            for layer_idx in layer_indices:
                adapted.pop(f"k_stack.layers.{layer_idx}.{legacy_suffix}", None)

        if "k_stack.decay_norm.scale" not in adapted:
            legacy_norm_scales: List[torch.Tensor] = []
            for layer_idx in layer_indices:
                legacy_norm_key = f"k_stack.layers.{layer_idx}.norm1.scale"
                tensor = adapted.get(legacy_norm_key)
                if isinstance(tensor, torch.Tensor):
                    legacy_norm_scales.append(tensor)
            if legacy_norm_scales:
                adapted["k_stack.decay_norm.scale"] = torch.stack(legacy_norm_scales, dim=0).mean(dim=0)

        for key, tensor in list(adapted.items()):
            match = LAYER_ALPHA_LOGIT_KEY_RE.match(key)
            if match is None or not isinstance(tensor, torch.Tensor):
                continue
            layer_idx = int(match.group(1))
            layer = self.k_stack.layers[layer_idx] if 0 <= layer_idx < len(self.k_stack.layers) else None
            if not isinstance(layer, K2Layer):
                continue
            target_rank = int(layer.alpha_logit.numel())
            if tensor.ndim == 0:
                adapted[key] = tensor.expand(target_rank).clone()
            elif tensor.ndim == 1 and tensor.numel() == 1:
                adapted[key] = tensor.expand(target_rank).clone()

        for key, tensor in list(adapted.items()):
            match = LAYER_K_BASE_KERNEL_KEY_RE.match(key)
            if match is None or not isinstance(tensor, torch.Tensor):
                continue
            adapted[key] = resize_kernel(tensor, self.k_base_kernel_size)

        if not self.future_summary_horizons:
            for key in list(adapted.keys()):
                if key.startswith("future_summary_"):
                    adapted.pop(key, None)
        else:
            valid_head_prefixes = {f"future_summary_heads.{h}." for h in self.future_summary_horizons}
            for key in list(adapted.keys()):
                if key.startswith("future_summary_heads.") and not any(
                    key.startswith(prefix) for prefix in valid_head_prefixes
                ):
                    adapted.pop(key, None)
        if not self.trajectory_aux:
            for key in list(adapted.keys()):
                if key.startswith("trajectory_aux_head."):
                    adapted.pop(key, None)

        return self.head.adapt_state_dict(adapted)

    def describe_rosa_layers(self) -> str:
        return describe_k2_layer_mask(self.rosa_k2_layer_mask)

    def _compute_rosa_hidden(self, x: torch.Tensor, device: torch.device) -> torch.Tensor | None:
        if not any(self.rosa_k2_layer_mask):
            return None
        next_ids = self.rosa_backend.next_token_ids(x, vocab_size=self.vocab_size)
        if next_ids is None:
            return None
        rosa_shifted = (next_ids + 1).clamp(min=0)
        rosa_emb = self.rosa_emb(rosa_shifted.to(device))
        return self.rosa_to_model(rosa_emb)

    def _forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h = self.emb_to_model(h)
        rosa_h = self._compute_rosa_hidden(x, device=h.device)
        h = self.emb_drop(h)
        h = self.k_stack(h, rosa_h=rosa_h, rosa_layer_mask=self.rosa_k2_layer_mask)
        return self.norm(h)

    def predict_future_summaries(self, h: torch.Tensor) -> Dict[int, torch.Tensor]:
        if not self.future_summary_horizons or self.future_summary_norm is None or self.future_summary_up is None:
            return {}
        shared = self.future_summary_norm(h)
        shared = self.future_summary_up(shared)
        shared = self.future_summary_activation(shared)
        return {int(key): head(shared) for key, head in self.future_summary_heads.items()}

    def _loss_from_hidden(self, h: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        return self.head.loss(h, targets, reduction=reduction)

    def hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_hidden(x)

    def scores_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        return self.head.scores(h)

    def loss_from_hidden(self, h: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        return self._loss_from_hidden(h, targets, reduction=reduction)

    def trajectory_aux_scores_from_hidden(self, h: torch.Tensor) -> torch.Tensor | None:
        if self.trajectory_aux_head is None:
            return None
        return self.trajectory_aux_head.scores(h)

    def trajectory_aux_loss_from_hidden(
        self,
        h: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if self.trajectory_aux_head is None:
            return torch.zeros((), device=h.device, dtype=torch.float32)
        return self.trajectory_aux_head.loss(h, targets, reduction=reduction)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        h = self._forward_hidden(x)
        if targets is not None:
            return self._loss_from_hidden(h, targets, reduction=reduction)
        return self.head.scores(h)

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

        emb_params = (
            list(self.emb.parameters())
            + list(self.emb_to_model.parameters())
            + list(self.rosa_emb.parameters())
            + list(self.rosa_to_model.parameters())
        )
        head_params = list(self.head.parameters())
        if self.trajectory_aux_head is not None:
            head_params += list(self.trajectory_aux_head.parameters())

        emb, emb_ptrs = _count_unique(emb_params)
        stack, emb_stack_ptrs = _count_unique(self.k_stack.parameters(), skip_ptrs=emb_ptrs)
        head, head_ptrs = _count_unique(head_params, skip_ptrs=emb_stack_ptrs)
        total, _ = _count_unique(self.parameters())
        other, _ = _count_unique(self.parameters(), skip_ptrs=head_ptrs)
        return {"total": total, "embedding": emb, "k_stack": stack, "head": head, "other": other}


__all__ = ["K2Layer", "KStack", "KStackModel"]
