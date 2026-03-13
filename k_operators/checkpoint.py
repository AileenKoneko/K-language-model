import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .runtime import DEVICE, LOG, _unwrap_model


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, best_ppl: float) -> None:
    torch.save(
        {
            "step": step,
            "best_ppl": best_ppl,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
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

    step_out = 0 if step is None else int(step)
    best_ppl_out = float("inf") if best_ppl is None else float(best_ppl)
    LOG.info(
        "Checkpoint loaded | step=%d | best_ppl=%s | optimizer_loaded=%s | path=%s",
        step_out,
        "inf" if math.isinf(best_ppl_out) else f"{best_ppl_out:.3f}",
        str(optimizer_loaded).lower(),
        path,
    )
    return step_out, best_ppl_out
