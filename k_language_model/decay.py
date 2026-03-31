from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Type

import torch
import torch.nn as nn

from .decay_kernel import decay_kernel, is_decay_kernel_available


_DECAY_REGISTRY: Dict[str, Type["DecayImplementation"]] = {}


def register_decay_impl(cls: Type["DecayImplementation"]) -> Type["DecayImplementation"]:
    if not cls.name:
        raise ValueError("DecayImplementation subclasses must define a non-empty name.")
    _DECAY_REGISTRY[cls.name] = cls
    return cls


def build_decay_impl(name: str) -> "DecayImplementation":
    try:
        return _DECAY_REGISTRY[str(name).strip().lower()]()
    except KeyError as exc:
        raise ValueError(f"Unknown decay_impl: {name}") from exc


@dataclass
class DecayBuffers:
    causal_mask: torch.Tensor
    decay_diff: torch.Tensor


class DecayImplementation(nn.Module, ABC):
    name = ""

    @abstractmethod
    def compute(
        self,
        *,
        q_alpha: torch.Tensor,
        k: torch.Tensor,
        h_norm: torch.Tensor,
        gamma_vec: torch.Tensor,
        buffers: DecayBuffers,
    ) -> torch.Tensor:
        raise NotImplementedError


@register_decay_impl
class MaskDecay(DecayImplementation):
    name = "mask"

    def compute(self, *, q_alpha, k, h_norm, gamma_vec, buffers: DecayBuffers) -> torch.Tensor:
        window = h_norm.size(1)
        kh = k.unsqueeze(-1) * h_norm.unsqueeze(-2)
        dist = buffers.decay_diff[:window, :window].to(dtype=h_norm.dtype, device=h_norm.device).unsqueeze(-1)
        log_gamma = torch.log(gamma_vec.clamp(min=1e-8)).view(1, 1, -1)
        causal = buffers.causal_mask[:window, :window].to(dtype=h_norm.dtype, device=h_norm.device).unsqueeze(-1)
        decay_mask = torch.exp(dist * log_gamma) * causal
        causal_inner = torch.einsum("ijr,bjrd->bird", decay_mask, kh)
        return torch.einsum("bwr,bwrd->bwd", q_alpha, causal_inner)


@register_decay_impl
class BlockDecay(DecayImplementation):
    name = "block"

    def compute(self, *, q_alpha, k, h_norm, gamma_vec, buffers: DecayBuffers) -> torch.Tensor:
        batch, window, _ = h_norm.shape
        block = min(128, window)
        calc_dtype = torch.float32 if h_norm.dtype in (torch.float16, torch.bfloat16) else h_norm.dtype
        decay_out = torch.zeros(batch, window, h_norm.size(2), device=h_norm.device, dtype=calc_dtype)
        state = torch.zeros(batch, q_alpha.size(-1), h_norm.size(2), device=h_norm.device, dtype=calc_dtype)
        log_gamma = torch.log(gamma_vec.to(dtype=calc_dtype, device=h_norm.device).clamp(min=1e-8))
        local_dist = buffers.decay_diff[:block, :block].to(device=h_norm.device, dtype=calc_dtype).unsqueeze(-1)
        local_causal = buffers.causal_mask[:block, :block].to(device=h_norm.device, dtype=calc_dtype).unsqueeze(-1)
        prev_decay_base = torch.exp(
            torch.arange(1, block + 1, device=h_norm.device, dtype=calc_dtype).unsqueeze(1) * log_gamma.unsqueeze(0)
        )

        for start in range(0, window, block):
            end = min(start + block, window)
            length = end - start

            h_blk = h_norm[:, start:end].to(dtype=calc_dtype)
            q_blk = q_alpha[:, start:end].to(dtype=calc_dtype)
            k_blk = k[:, start:end].to(dtype=calc_dtype)
            decay_mask = torch.exp(local_dist[:length, :length] * log_gamma.view(1, 1, -1)) * local_causal[:length, :length]
            prev_decay = prev_decay_base[:length].view(1, length, -1, 1)
            kh_blk = k_blk.unsqueeze(-1) * h_blk.unsqueeze(-2)
            local_state = torch.einsum("ijr,bjrd->bird", decay_mask, kh_blk)
            state_blk = local_state + state.unsqueeze(1) * prev_decay

            decay_out[:, start:end].add_((q_blk.unsqueeze(-1) * state_blk).sum(dim=2))
            state = state_blk[:, -1]

        return decay_out.to(dtype=h_norm.dtype)


@register_decay_impl
class KernelDecay(DecayImplementation):
    name = "kernel"

    def __init__(self):
        super().__init__()
        self._fallback = BlockDecay()
        self._warned = False

    def compute(self, *, q_alpha, k, h_norm, gamma_vec, buffers: DecayBuffers) -> torch.Tensor:
        if is_decay_kernel_available(h_norm.device, q_alpha.size(-1)):
            try:
                return decay_kernel(q=q_alpha, k=k, h=h_norm, gamma=gamma_vec.to(dtype=torch.float32))
            except Exception as exc:
                if not self._warned:
                    warnings.warn(f"decay_impl='kernel' launch failed; fallback to block backend ({exc}).", stacklevel=2)
                    self._warned = True
        else:
            if not self._warned:
                if h_norm.device.type != "cuda":
                    reason = f"device={h_norm.device.type}"
                else:
                    reason = f"rank={q_alpha.size(-1)} not in supported range for kernel backend"
                warnings.warn(f"decay_impl='kernel' fallback to block backend ({reason}).", stacklevel=2)
                self._warned = True
        return self._fallback.compute(q_alpha=q_alpha, k=k, h_norm=h_norm, gamma_vec=gamma_vec, buffers=buffers)
