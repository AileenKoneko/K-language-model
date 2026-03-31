from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_K_BASE_KERNEL_SIZE = 8
_KBASE_REGISTRY: Dict[str, Type["KBaseImplementation"]] = {}


def register_kbase_impl(cls: Type["KBaseImplementation"]) -> Type["KBaseImplementation"]:
    if not cls.name:
        raise ValueError("KBaseImplementation subclasses must define a non-empty name.")
    _KBASE_REGISTRY[cls.name] = cls
    return cls


def resolve_kbase_impl_name(raw: str) -> str:
    aliases = {
        "auto": "conv",
        "fused": "conv",
        "scan": "conv",
    }
    normalized = aliases.get(str(raw).strip().lower(), str(raw).strip().lower())
    if normalized not in _KBASE_REGISTRY:
        raise ValueError(f"Unknown k_base_impl: {raw}")
    return normalized


def build_kbase_impl(raw: str) -> "KBaseImplementation":
    name = resolve_kbase_impl_name(raw)
    return _KBASE_REGISTRY[name]()


def resize_kernel(kernel: torch.Tensor, target_size: int) -> torch.Tensor:
    target = torch.zeros(target_size, dtype=kernel.dtype, device=kernel.device)
    copy_count = min(int(kernel.numel()), int(target_size))
    if copy_count > 0:
        target[:copy_count] = kernel.reshape(-1)[:copy_count]
    return target


class KBaseImplementation(nn.Module, ABC):
    name = ""

    @abstractmethod
    def default_parameter(self, kernel_size: int) -> nn.Parameter:
        raise NotImplementedError

    @abstractmethod
    def compute(self, h_norm: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def prepare_parameter_for_load(self, kernel: torch.Tensor, kernel_size: int) -> torch.Tensor:
        return resize_kernel(kernel, kernel_size)


@register_kbase_impl
class ConvKBase(KBaseImplementation):
    name = "conv"

    def default_parameter(self, kernel_size: int) -> nn.Parameter:
        lags = torch.arange(kernel_size, dtype=torch.float32)
        kernel = torch.exp(-0.15 * lags)
        kernel = kernel / kernel.sum().clamp(min=1e-8)
        return nn.Parameter(kernel)

    def compute(self, h_norm: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        batch, window, d_model = h_norm.shape
        kernel = parameter[: min(window, parameter.numel())]
        x = h_norm.transpose(1, 2)
        if kernel.numel() > 1:
            x = F.pad(x, (kernel.numel() - 1, 0))
        weight = kernel.flip(0).view(1, 1, -1).expand(d_model, 1, -1).contiguous()
        out = F.conv1d(x, weight, groups=d_model)
        return out.transpose(1, 2).contiguous()
