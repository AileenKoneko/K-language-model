import re
from typing import List

import torch


LAYER_ALPHA_LOGIT_KEY_RE = re.compile(r"^k_stack\.layers\.(\d+)\.alpha_logit$")
LAYER_K_BASE_KERNEL_KEY_RE = re.compile(r"^k_stack\.layers\.(\d+)\.k_base_kernel$")


def resolve_k2_layer_mask(raw: str | None, n_k2: int, label: str) -> List[bool]:
    if n_k2 < 0:
        raise ValueError(f"Expected non-negative n_k2, got {n_k2}.")
    if n_k2 == 0:
        return []

    spec = "all" if raw is None else str(raw).strip().lower()
    if spec in {"", "all"}:
        return [True] * n_k2
    if spec in {"none", "off"}:
        return [False] * n_k2
    if spec in {"final", "last"}:
        return [idx == (n_k2 - 1) for idx in range(n_k2)]

    parts = [part.strip() for part in str(raw).split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(
            f"{label} must be one of: all, none, final, or a comma-separated list of 1-based K2 layer ids."
        )

    enabled = [False] * n_k2
    for part in parts:
        if not part.isdigit():
            raise ValueError(
                f"{label} must be one of: all, none, final, or a comma-separated list of 1-based K2 layer ids. Got {raw!r}."
            )
        idx = int(part)
        if not 1 <= idx <= n_k2:
            raise ValueError(f"{label} layer ids must be within [1, {n_k2}], got {idx}.")
        enabled[idx - 1] = True
    return enabled


def describe_k2_layer_mask(mask: List[bool]) -> str:
    if not mask:
        return "none"
    enabled = [idx + 1 for idx, use_layer in enumerate(mask) if use_layer]
    if len(enabled) == len(mask):
        return "all"
    if not enabled:
        return "none"
    if enabled == [len(mask)]:
        return "final"
    return ",".join(str(idx) for idx in enabled)


def is_torch_compiling() -> bool:
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
            cutoffs = [max(256, vocab_size // 4), max(768, (3 * vocab_size) // 4)]
        elif vocab_size <= 16384:
            cutoffs = [2000, max(4000, vocab_size // 2)]
        else:
            cutoffs = [2000, 10000, max(20000, (3 * vocab_size) // 4)]

    normalized = sorted({int(cutoff) for cutoff in cutoffs if 0 < int(cutoff) < vocab_size})
    if not normalized:
        raise ValueError(f"Adaptive softmax cutoffs must be within (0, vocab_size). Got {cutoffs} for vocab={vocab_size}.")
    return normalized
