# Portions of this file are derived from RWKV-v8 ROSA pseudocode/materials.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt for attribution and terms.

from __future__ import annotations

import importlib.util
import os
import shutil
import site
import sys
import sysconfig
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.cpp_extension import _get_build_directory
from torch.utils.cpp_extension import load as load_cpp_extension


_VALID_INT_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}

_ROSA_CPP_DISABLE_ENV = "KLM_DISABLE_ROSA_CPP"
_ROSA_CPP_MODULE_NAME = "k_language_model_rosa_cpp"
_ROSA_NGRAM_CACHE_MAX_BIGRAM_VOCAB = 256


def _validate_integer_tensor(token_ids: torch.Tensor, *, expected_ndim: int) -> None:
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError(f"ROSA expects a torch.Tensor input, got {type(token_ids)!r}.")
    if token_ids.ndim != expected_ndim:
        raise ValueError(f"ROSA expects a {expected_ndim}D tensor of token ids, got shape {tuple(token_ids.shape)}.")
    if token_ids.dtype not in _VALID_INT_DTYPES:
        raise TypeError(f"ROSA expects an integer tensor, got dtype {token_ids.dtype}.")


def _validate_token_ids(token_ids: torch.Tensor) -> None:
    _validate_integer_tensor(token_ids, expected_ndim=1)
    if token_ids.device.type != "cpu":
        raise ValueError(f"ROSA currently supports CPU tensors only, got device {token_ids.device}.")


def _validate_token_id_batch(token_ids: torch.Tensor) -> None:
    _validate_integer_tensor(token_ids, expected_ndim=2)


def _find_windows_cpp_compiler() -> str | None:
    candidates = []
    env_cxx = os.environ.get("CXX", "").strip()
    if env_cxx:
        candidates.append(env_cxx)
    candidates.extend(["cl", "clang-cl"])
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _candidate_python_script_dirs() -> list[str]:
    candidates: list[str] = []

    def add(path: str | Path | None) -> None:
        if not path:
            return
        path_obj = Path(path)
        if not path_obj.exists():
            return
        path_str = str(path_obj)
        if path_str not in candidates:
            candidates.append(path_str)

    add(sysconfig.get_path("scripts"))
    if os.name == "nt":
        try:
            add(sysconfig.get_path("scripts", scheme="nt_user"))
        except Exception:
            pass
        add(Path(site.USER_BASE) / f"Python{sysconfig.get_python_version().replace('.', '')}" / "Scripts")
        user_base = Path(site.USER_BASE)
        if user_base.exists():
            for path in sorted(user_base.glob("Python*/Scripts")):
                add(path)
        roaming_python = Path.home() / "AppData" / "Roaming" / "Python"
        if roaming_python.exists():
            for path in sorted(roaming_python.glob("Python*/Scripts")):
                add(path)
    return candidates


def _rosa_cpp_cached_binary_exists() -> bool:
    return _rosa_cpp_cached_binary_path() is not None


def _rosa_cpp_cached_binary_path() -> Path | None:
    try:
        build_dir = Path(_get_build_directory(_ROSA_CPP_MODULE_NAME, verbose=False))
    except Exception:
        return None
    suffixes = [".pyd"] if os.name == "nt" else [".so", ".dylib"]
    for suffix in suffixes:
        for path in build_dir.glob(f"{_ROSA_CPP_MODULE_NAME}*{suffix}"):
            return path
    return None


def _load_rosa_cpp_cached_binary():
    cached_path = _rosa_cpp_cached_binary_path()
    if cached_path is None:
        return None
    module = sys.modules.get(_ROSA_CPP_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(_ROSA_CPP_MODULE_NAME, cached_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ROSA_CPP_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_rosa_cpp_extension():
    if os.environ.get(_ROSA_CPP_DISABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        return None

    source_path = Path(__file__).with_name("rosa_ext.cpp")
    extra_cflags = ["/O2"] if os.name == "nt" else ["-O3"]
    env = os.environ.copy()
    script_candidates = _candidate_python_script_dirs()
    if shutil.which("ninja", path=env.get("PATH")) is None:
        prepend = [str(Path(path)) for path in script_candidates if path and Path(path).exists()]
        if prepend:
            env["PATH"] = os.pathsep.join(prepend + [env.get("PATH", "")])
    cached_binary_exists = _rosa_cpp_cached_binary_exists()
    if cached_binary_exists:
        try:
            return _load_rosa_cpp_cached_binary()
        except Exception:
            sys.modules.pop(_ROSA_CPP_MODULE_NAME, None)
    if not cached_binary_exists and shutil.which("ninja", path=env.get("PATH")) is None:
        warnings.warn(
            "ROSA C++ extension unavailable; falling back to Python exact backend (missing ninja on PATH).",
            stacklevel=2,
        )
        return None
    if not cached_binary_exists and os.name == "nt" and _find_windows_cpp_compiler() is None:
        warnings.warn(
            "ROSA C++ extension unavailable; falling back to Python exact backend "
            "(missing MSVC/clang-cl toolchain on PATH).",
            stacklevel=2,
        )
        return None
    try:
        os.environ["PATH"] = env.get("PATH", "")
        return load_cpp_extension(
            name=_ROSA_CPP_MODULE_NAME,
            sources=[str(source_path)],
            extra_cflags=extra_cflags,
            with_cuda=False,
            verbose=False,
        )
    except Exception as exc:
        warnings.warn(
            f"ROSA C++ extension unavailable; falling back to Python exact backend ({exc}).",
            stacklevel=2,
        )
        return None


def rosa_cpp_extension_available() -> bool:
    return _load_rosa_cpp_extension() is not None


def _rosa_next_token_ids_python(token_ids: torch.Tensor) -> torch.Tensor:
    """Return the next-token continuation for the longest previous matching suffix.

    The output is an int64 tensor of the same length as ``token_ids``. Each position
    contains the token that followed the longest previous occurrence of the current
    suffix, or ``-1`` when no such continuation exists.
    """

    _validate_token_ids(token_ids)

    x = [int(token) for token in token_ids.tolist()]
    n = len(x)
    predictions = [-1] * n

    num_states = 2 * n + 1
    transitions: List[Dict[int, int] | None] = [None] * num_states
    suffix_link = [-1] * num_states
    depth = [0] * num_states
    end_pos = [-1] * num_states

    transitions[0] = {}
    last_state = 0
    next_state = 1

    for i, token in enumerate(x):
        current = next_state
        next_state += 1
        transitions[current] = {}
        depth[current] = depth[last_state] + 1
        state = last_state

        while state != -1 and token not in transitions[state]:
            transitions[state][token] = current
            state = suffix_link[state]

        if state == -1:
            suffix_link[current] = 0
        else:
            target = transitions[state][token]
            if depth[state] + 1 == depth[target]:
                suffix_link[current] = target
            else:
                clone = next_state
                next_state += 1
                transitions[clone] = dict(transitions[target])
                depth[clone] = depth[target] + 1
                suffix_link[clone] = suffix_link[target]
                end_pos[clone] = end_pos[target]

                while state != -1 and transitions[state].get(token) == target:
                    transitions[state][token] = clone
                    state = suffix_link[state]

                suffix_link[target] = clone
                suffix_link[current] = clone

        last_state = current
        state = last_state
        predicted = -1

        while state != -1:
            if depth[state] > 0 and end_pos[state] >= 0:
                predicted = x[end_pos[state] + 1]
                break
            state = suffix_link[state]

        predictions[i] = predicted
        state = last_state

        while state != -1 and end_pos[state] < i:
            end_pos[state] = i
            state = suffix_link[state]

    return torch.tensor(predictions, dtype=torch.int64, device=token_ids.device)


def _rosa_next_token_ids_batch_exact(token_ids: torch.Tensor) -> torch.Tensor:
    _validate_token_id_batch(token_ids)

    x_cpu = token_ids.detach().to(device="cpu", dtype=torch.int64).contiguous()
    ext = _load_rosa_cpp_extension()
    if ext is not None:
        out_cpu = ext.rosa_next_token_ids_batch_exact_cpu(x_cpu)
    else:
        out_cpu = torch.stack([_rosa_next_token_ids_python(row) for row in x_cpu], dim=0)
    return out_cpu.to(token_ids.device)


def rosa_next_token_ids(token_ids: torch.Tensor) -> torch.Tensor:
    _validate_token_ids(token_ids)
    out = _rosa_next_token_ids_batch_exact(token_ids.view(1, -1))
    return out.squeeze(0)


def _rosa_next_token_ids_batch_gpu_approx(token_ids: torch.Tensor) -> torch.Tensor:
    """Approximate batched ROSA using a GPU-friendly LCS suffix DP.

    This path is fully tensorized and accelerator-friendly, but it is not guaranteed
    to be bit-identical to the exact suffix-automaton reference for all low-vocabulary,
    highly repetitive sequences.
    """

    _validate_token_id_batch(token_ids)

    batch, window = token_ids.shape
    if window == 0:
        return torch.empty((batch, 0), dtype=torch.int64, device=token_ids.device)

    # `window` is typically <= 512, so int16 is sufficient and saves bandwidth.
    work_dtype = torch.int16 if window <= 32767 else torch.int32
    prev = torch.zeros((batch, window), dtype=work_dtype, device=token_ids.device)
    prev_shift = torch.zeros_like(prev)
    predictions = torch.full((batch, window), -1, dtype=torch.int64, device=token_ids.device)

    positions = torch.arange(window, device=token_ids.device, dtype=torch.int32).view(1, window)
    next_valid = positions < (window - 1)
    neg_one_matrix = torch.full((batch, window), -1, dtype=torch.int32, device=token_ids.device)
    neg_one_vec = torch.full((batch,), -1, dtype=torch.int64, device=token_ids.device)

    for i in range(window):
        eq = token_ids[:, i : i + 1] == token_ids
        prev_shift.zero_()
        prev_shift[:, 1:] = prev[:, :-1]
        curr = torch.where(eq, prev_shift + 1, torch.zeros_like(prev))

        valid = (positions < i) & next_valid
        curr_valid = curr.masked_fill(~valid, 0)
        best_len = curr_valid.max(dim=1).values

        best_mask = (curr_valid == best_len.unsqueeze(1)) & (best_len.unsqueeze(1) > 0)
        best_pos_scores = torch.where(best_mask, positions.expand(batch, -1), neg_one_matrix)
        best_pos = best_pos_scores.max(dim=1).values.to(dtype=torch.int64)

        has_match = best_pos >= 0
        next_pos = (best_pos + 1).clamp(min=0)
        next_token = token_ids.gather(1, next_pos.unsqueeze(1)).squeeze(1).to(torch.int64)
        predictions[:, i] = torch.where(has_match, next_token, neg_one_vec)
        prev = curr

    return predictions


def _rosa_next_token_ids_batch_ngram_cache(token_ids: torch.Tensor) -> torch.Tensor:
    """Behavior-oriented ROSA proxy using online unigram/bigram continuation caches.

    This path keeps ROSA's key behavior (emit likely continuation token for repeated
    contexts) while avoiding suffix-automaton construction.
    """

    _validate_token_id_batch(token_ids)

    batch, window = token_ids.shape
    if window == 0:
        return torch.empty((batch, 0), dtype=torch.int64, device=token_ids.device)

    token_ids_i64 = token_ids.to(dtype=torch.int64)
    vocab = int(token_ids_i64.max().item()) + 1
    predictions = torch.full((batch, window), -1, dtype=torch.int64, device=token_ids.device)
    if vocab <= 0:
        return predictions

    unigram_follow = torch.full((batch, vocab), -1, dtype=torch.int64, device=token_ids.device)
    use_bigram = vocab <= _ROSA_NGRAM_CACHE_MAX_BIGRAM_VOCAB
    if use_bigram:
        bigram_follow = torch.full((batch, vocab * vocab), -1, dtype=torch.int64, device=token_ids.device)

    for i in range(window):
        token = token_ids_i64[:, i]
        pred_unigram = unigram_follow.gather(1, token.view(batch, 1)).squeeze(1)

        if use_bigram and i > 0:
            prev = token_ids_i64[:, i - 1]
            pair_key = prev * vocab + token
            pred_bigram = bigram_follow.gather(1, pair_key.view(batch, 1)).squeeze(1)
            predictions[:, i] = torch.where(pred_bigram >= 0, pred_bigram, pred_unigram)
        else:
            predictions[:, i] = pred_unigram

        if i + 1 < window:
            nxt = token_ids_i64[:, i + 1]
            unigram_follow.scatter_(1, token.view(batch, 1), nxt.view(batch, 1))
            if use_bigram and i > 0:
                bigram_follow.scatter_(1, pair_key.view(batch, 1), nxt.view(batch, 1))

    return predictions


def _rosa_next_token_ids_batch_copy_prior(token_ids: torch.Tensor) -> torch.Tensor:
    """Cheap history-copy prior: predict continuation from last seen token match.

    Compared to suffix automata, this intentionally keeps only a lightweight signal:
    for token `x[i]`, emit the token that followed the most recent prior occurrence
    of `x[i]` in the same sequence, else `-1`.
    """

    _validate_token_id_batch(token_ids)

    token_ids_i64 = token_ids.to(dtype=torch.int64)
    batch, window = token_ids_i64.shape
    if window == 0:
        return torch.empty((batch, 0), dtype=torch.int64, device=token_ids.device)

    vocab = int(token_ids_i64.max().item()) + 1
    predictions = torch.full((batch, window), -1, dtype=torch.int64, device=token_ids.device)
    if vocab <= 0:
        return predictions

    seen_next = torch.full((batch, vocab), -1, dtype=torch.int64, device=token_ids.device)
    for i in range(window):
        token = token_ids_i64[:, i]
        predictions[:, i] = seen_next.gather(1, token.view(batch, 1)).squeeze(1)
        if i + 1 < window:
            nxt = token_ids_i64[:, i + 1]
            seen_next.scatter_(1, token.view(batch, 1), nxt.view(batch, 1))
    return predictions


def rosa_next_token_ids_batch(token_ids: torch.Tensor, impl: str = "exact") -> torch.Tensor:
    """Batched ROSA for [B, W] integer token tensors.

    `impl`:
    - `exact`: bit-identical to reference CPU implementation (may move tensors to CPU).
    - `gpu_approx`: GPU-native tensorized approximation.
    - `ngram_cache`: online unigram/bigram continuation cache proxy.
    - `copy_prior`: cheap history-copy continuation proxy.
    - `auto`: use `gpu_approx` on accelerators, else `exact`.
    """

    if impl not in {"exact", "gpu_approx", "ngram_cache", "copy_prior", "auto"}:
        raise ValueError(f"Unknown ROSA batch impl: {impl}")
    _validate_token_id_batch(token_ids)

    if impl == "auto":
        impl = "gpu_approx" if token_ids.device.type in {"cuda", "mps"} else "exact"

    if impl == "gpu_approx":
        return _rosa_next_token_ids_batch_gpu_approx(token_ids)
    if impl == "ngram_cache":
        return _rosa_next_token_ids_batch_ngram_cache(token_ids)
    if impl == "copy_prior":
        return _rosa_next_token_ids_batch_copy_prior(token_ids)

    return _rosa_next_token_ids_batch_exact(token_ids)


class ROSA(nn.Module):
    """Deterministic ROSA next-token predictor for 1D token id sequences."""

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return rosa_next_token_ids(token_ids)
