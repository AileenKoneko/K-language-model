from __future__ import annotations

import math
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
    ):
        super().__init__()
        self.window = int(window)
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
        self.decay_impl = str(decay_impl).strip().lower()
        self.k_base_backend = build_kbase_impl(self.k_base_impl)
        self.decay_backend = build_decay_impl(self.decay_impl)

        tau_low = 1.0 / max(1.0 - self.gamma_min, 1e-6)
        tau_high = 1.0 / max(1.0 - min(self.gamma_max, 1.0 - 1e-6), 1e-6)
        taus = torch.logspace(math.log10(tau_low), math.log10(tau_high), rank)
        gammas = 1.0 - 1.0 / taus
        p = (gammas - self.gamma_min) / (self.gamma_max - self.gamma_min)
        p = p.clamp(min=1e-6, max=1.0 - 1e-6)
        self.decay_logit = nn.Parameter(torch.log(p / (1.0 - p)))

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

        self.u = nn.Parameter(torch.randn(d, rank) * 0.05)
        self.v = nn.Parameter(torch.randn(d, rank) * 0.05)
        self.alpha_logit = nn.Parameter(torch.zeros(rank))
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

    def forward(
        self,
        h: torch.Tensor,
        shared_k_base: torch.Tensor | None = None,
        rosa_h: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        residual = h
        h_norm = self.norm1(h)

        gate_strength = torch.sigmoid(self.k_base_gate_logit).to(dtype=h.dtype, device=h.device)
        k_base_kernel = self._resolve_k_base_kernel(shared_k_base, dtype=h.dtype, device=h.device)
        out = gate_strength * self.k_base_backend.compute(h_norm, k_base_kernel)

        qk = h_norm @ torch.cat((self.u, self.v), dim=1)
        q_raw, k_raw = qk.split(self.u.size(1), dim=-1)
        q = F.normalize(q_raw, dim=-1, eps=1e-8)
        k = F.normalize(k_raw, dim=-1, eps=1e-8)

        raw = torch.sigmoid(self.decay_logit)
        gamma_vec = (self.gamma_min + (self.gamma_max - self.gamma_min) * raw).to(dtype=h.dtype, device=h.device)
        alpha = (self.alpha_cap * torch.sigmoid(self.alpha_logit)).to(dtype=h.dtype, device=h.device)
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
        gamma_min: float = 0.85,
        gamma_max: float = 1.0,
        decay_impl: str = "mask",
    ):
        super().__init__()
        self.share_k_base = bool(share_k_base)
        self.k_base_impl = resolve_kbase_impl_name(k_base_impl)
        self.k_base_kernel_size = max(int(k_base_kernel_size), 1)
        self.k_base_backend = build_kbase_impl(self.k_base_impl)
        if self.share_k_base:
            self.shared_k_base_kernel = self.k_base_backend.default_parameter(self.k_base_kernel_size)
        else:
            self.register_parameter("shared_k_base_kernel", None)
        self.n_k2_layers = int(n_k2)

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
                )
            )
        layers.append(K1Layer(d, mlp_dropout=mlp_dropout))
        layers.append(K0Layer(d))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        h: torch.Tensor,
        rosa_h: torch.Tensor | None = None,
        rosa_layer_mask: List[bool] | None = None,
    ) -> torch.Tensor:
        if rosa_layer_mask is not None and len(rosa_layer_mask) != self.n_k2_layers:
            raise ValueError(f"Expected rosa_layer_mask length {self.n_k2_layers}, got {len(rosa_layer_mask)}.")

        k2_idx = 0
        for layer in self.layers:
            if isinstance(layer, K2Layer):
                use_rosa = rosa_h is not None and (rosa_layer_mask is None or bool(rosa_layer_mask[k2_idx]))
                h = layer(h, shared_k_base=self.shared_k_base_kernel, rosa_h=rosa_h if use_rosa else None)
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
        adaptive_cutoffs: List[int] | None = None,
        adaptive_div_value: float = 4.0,
        alpha_cap: float = 1.0,
        gamma_min: float = 0.85,
        gamma_max: float = 1.0,
        decay_impl: str = "mask",
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
        self.k_base_rank = max(int(k_base_rank), 0)
        self.k_base_impl = resolve_kbase_impl_name(k_base_impl)
        self.k_base_kernel_size = max(int(k_base_kernel_size), 1)
        self.share_k_base = bool(share_k_base)
        self.decay_impl = str(decay_impl).strip().lower()
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

        self.apply(self._init_weights)

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

        for key, tensor in list(adapted.items()):
            match = LAYER_ALPHA_LOGIT_KEY_RE.match(key)
            if match is None or not isinstance(tensor, torch.Tensor):
                continue
            layer_idx = int(match.group(1))
            layer = self.k_stack.layers[layer_idx] if 0 <= layer_idx < len(self.k_stack.layers) else None
            if not isinstance(layer, K2Layer):
                continue
            target_rank = int(layer.decay_logit.numel())
            if tensor.ndim == 0:
                adapted[key] = tensor.expand(target_rank).clone()
            elif tensor.ndim == 1 and tensor.numel() == 1:
                adapted[key] = tensor.expand(target_rank).clone()

        for key, tensor in list(adapted.items()):
            match = LAYER_K_BASE_KERNEL_KEY_RE.match(key)
            if match is None or not isinstance(tensor, torch.Tensor):
                continue
            adapted[key] = resize_kernel(tensor, self.k_base_kernel_size)

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

    def _loss_from_hidden(self, h: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        return self.head.loss(h, targets, reduction=reduction)

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

        emb, emb_ptrs = _count_unique(emb_params)
        stack, emb_stack_ptrs = _count_unique(self.k_stack.parameters(), skip_ptrs=emb_ptrs)
        head, head_ptrs = _count_unique(head_params, skip_ptrs=emb_stack_ptrs)
        total, _ = _count_unique(self.parameters())
        other, _ = _count_unique(self.parameters(), skip_ptrs=head_ptrs)
        return {"total": total, "embedding": emb, "k_stack": stack, "head": head, "other": other}


__all__ = ["K2Layer", "KStack", "KStackModel"]
