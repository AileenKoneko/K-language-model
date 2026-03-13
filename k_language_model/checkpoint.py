import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .runtime import DEVICE, LOG, _unwrap_model


def _capture_rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "torch_cpu": torch.random.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except RuntimeError:
            state["torch_cuda"] = None
    return state


def _restore_rng_state(state: object) -> bool:
    if not isinstance(state, dict):
        return False

    restored = False
    torch_cpu = state.get("torch_cpu")
    if torch_cpu is not None:
        torch.random.set_rng_state(torch_cpu)
        restored = True

    torch_cuda = state.get("torch_cuda")
    if torch_cuda is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(torch_cuda)
            restored = True
        except RuntimeError as exc:
            LOG.warning("CUDA RNG state not restored (%s). Continuing.", exc)

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
        restored = True

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
        restored = True

    return restored


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, best_ppl: float) -> None:
    torch.save(
        {
            "step": step,
            "best_ppl": best_ppl,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_state": _capture_rng_state(),
        },
        path,
    )
    LOG.info("Checkpoint saved | step=%d | best_ppl=%.3f | path=%s", step, best_ppl, path)


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = state_dict
    while True:
        changed = False
        for prefix in ("_orig_mod.", "module."):
            if normalized and all(k.startswith(prefix) for k in normalized.keys()):
                normalized = {k[len(prefix):]: v for k, v in normalized.items()}
                changed = True
        if not changed:
            break
    return normalized


def _extract_checkpoint_model_state(ck: object) -> Tuple[Dict[str, torch.Tensor], int | None, float | None]:
    step = None
    best_ppl = None
    if isinstance(ck, dict):
        if "step" in ck:
            try:
                step = int(ck["step"])
            except (TypeError, ValueError):
                step = None
        if "best_ppl" in ck:
            try:
                best_ppl = float(ck["best_ppl"])
            except (TypeError, ValueError):
                best_ppl = None
        raw_state = ck.get("model", ck)
    else:
        raw_state = ck

    if not isinstance(raw_state, dict):
        raise ValueError(f"Checkpoint at {ck!r} does not contain a model state_dict.")
    return _normalize_state_dict_keys(raw_state), step, best_ppl


def _log_state_load_mismatch(missing_keys: List[str], unexpected_keys: List[str]) -> None:
    if missing_keys:
        preview = ", ".join(missing_keys[:8])
        suffix = " ..." if len(missing_keys) > 8 else ""
        LOG.warning("Checkpoint missing keys (%d): %s%s", len(missing_keys), preview, suffix)
    if unexpected_keys:
        preview = ", ".join(unexpected_keys[:8])
        suffix = " ..." if len(unexpected_keys) > 8 else ""
        LOG.warning("Checkpoint unexpected keys (%d): %s%s", len(unexpected_keys), preview, suffix)


def load_model_checkpoint(path: Path, model: nn.Module) -> Tuple[int | None, float | None]:
    # This repository stores full training snapshots (not just tensors), so weights_only=False is intentional.
    # Do not load untrusted checkpoints.
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    state, step, best_ppl = _extract_checkpoint_model_state(ck)
    core_model = _unwrap_model(model)
    incompatible = core_model.load_state_dict(state, strict=False)
    _log_state_load_mismatch(list(incompatible.missing_keys), list(incompatible.unexpected_keys))
    step_str = "N/A" if step is None else str(step)
    best_str = "N/A" if best_ppl is None else f"{best_ppl:.3f}"
    LOG.info("Model checkpoint loaded | step=%s | best_ppl=%s | path=%s", step_str, best_str, path)
    return step, best_ppl


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[int, float]:
    # This repository stores full training snapshots (not just tensors), so weights_only=False is intentional.
    # Do not load untrusted checkpoints.
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    state, step, best_ppl = _extract_checkpoint_model_state(ck)
    core_model = _unwrap_model(model)
    incompatible = core_model.load_state_dict(state, strict=False)
    _log_state_load_mismatch(list(incompatible.missing_keys), list(incompatible.unexpected_keys))

    optimizer_loaded = False
    if isinstance(ck, dict) and "optimizer" in ck:
        try:
            optimizer.load_state_dict(ck["optimizer"])
            optimizer_loaded = True
        except ValueError as exc:
            LOG.warning("Optimizer state not loaded (%s). Continuing with fresh optimizer state.", exc)

    rng_restored = False
    if isinstance(ck, dict) and "rng_state" in ck:
        rng_restored = _restore_rng_state(ck["rng_state"])

    step_out = 0 if step is None else int(step)
    best_ppl_out = float("inf") if best_ppl is None else float(best_ppl)
    LOG.info(
        "Checkpoint loaded | step=%d | best_ppl=%s | optimizer_loaded=%s | rng_restored=%s | path=%s",
        step_out,
        "inf" if math.isinf(best_ppl_out) else f"{best_ppl_out:.3f}",
        str(optimizer_loaded).lower(),
        str(rng_restored).lower(),
        path,
    )
    return step_out, best_ppl_out
