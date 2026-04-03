# This file wires ROSA backends around the ROSA implementation in `rosa.py`.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt for attribution and terms.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Type

import torch

from .model_utils import is_torch_compiling
from .rosa import rosa_next_token_ids_batch


_ROSA_REGISTRY: Dict[str, Type["RosaBackend"]] = {}


def _dynamo_disable(fn):
    compiler_disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if callable(compiler_disable):
        return compiler_disable(fn)
    dynamo_disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    if callable(dynamo_disable):
        return dynamo_disable(fn)
    return fn


@_dynamo_disable
def _rosa_next_token_ids_exact_eager(token_ids: torch.Tensor) -> torch.Tensor:
    return rosa_next_token_ids_batch(token_ids, impl="exact")


def register_rosa_backend(cls: Type["RosaBackend"]) -> Type["RosaBackend"]:
    if not cls.name:
        raise ValueError("RosaBackend subclasses must define a non-empty name.")
    _ROSA_REGISTRY[cls.name] = cls
    return cls


def build_rosa_backend(name: str) -> "RosaBackend":
    try:
        return _ROSA_REGISTRY[str(name).strip().lower()]()
    except KeyError as exc:
        raise ValueError(f"Unknown rosa_impl: {name}") from exc


class RosaBackend(ABC):
    name = ""

    @abstractmethod
    def next_token_ids(self, token_ids: torch.Tensor, *, vocab_size: int) -> torch.Tensor | None:
        raise NotImplementedError


@register_rosa_backend
class OffRosaBackend(RosaBackend):
    name = "off"

    def next_token_ids(self, token_ids: torch.Tensor, *, vocab_size: int) -> torch.Tensor | None:
        return None


@register_rosa_backend
class ExactRosaBackend(RosaBackend):
    name = "exact"

    def next_token_ids(self, token_ids: torch.Tensor, *, vocab_size: int) -> torch.Tensor | None:
        return _rosa_next_token_ids_exact_eager(token_ids)


@register_rosa_backend
class GpuApproxRosaBackend(RosaBackend):
    name = "gpu_approx"

    def next_token_ids(self, token_ids: torch.Tensor, *, vocab_size: int) -> torch.Tensor | None:
        return rosa_next_token_ids_batch(token_ids, impl="gpu_approx")


@register_rosa_backend
class NgramCacheRosaBackend(RosaBackend):
    name = "ngram_cache"

    def next_token_ids(self, token_ids: torch.Tensor, *, vocab_size: int) -> torch.Tensor | None:
        return rosa_next_token_ids_batch(token_ids, impl="ngram_cache")


@register_rosa_backend
class CopyPriorRosaBackend(RosaBackend):
    name = "copy_prior"

    def next_token_ids(self, token_ids: torch.Tensor, *, vocab_size: int) -> torch.Tensor | None:
        return rosa_next_token_ids_batch(token_ids, impl="copy_prior")


@register_rosa_backend
class AutoRosaBackend(RosaBackend):
    name = "auto"

    def __init__(self):
        self._resolved_impl: str | None = None
        self._warned = False

    def next_token_ids(self, token_ids: torch.Tensor, *, vocab_size: int) -> torch.Tensor | None:
        impl = self._resolve_impl(token_ids, vocab_size=vocab_size)
        if impl == "exact":
            return _rosa_next_token_ids_exact_eager(token_ids)
        return rosa_next_token_ids_batch(token_ids, impl=impl)

    def _resolve_impl(self, token_ids: torch.Tensor, *, vocab_size: int) -> str:
        if self._resolved_impl is not None:
            return self._resolved_impl
        if token_ids.device.type == "cpu" or is_torch_compiling() or token_ids.device.type != "cuda" or int(vocab_size) <= 256:
            self._resolved_impl = "exact"
            return self._resolved_impl

        sample = token_ids[: max(1, min(int(token_ids.size(0)), 4))]
        approx = rosa_next_token_ids_batch(sample, impl="gpu_approx")
        exact = rosa_next_token_ids_batch(sample, impl="exact")
        if torch.equal(approx, exact):
            self._resolved_impl = "gpu_approx"
        else:
            self._resolved_impl = "exact"
            if not self._warned:
                diff_rate = float((approx != exact).float().mean().item())
                warnings.warn(
                    (
                        f"rosa_impl='auto' parity check failed on sample batch "
                        f"(diff_rate={diff_rate:.6f}); falling back to exact CPU ROSA."
                    ),
                    stacklevel=2,
                )
                self._warned = True
        return self._resolved_impl
