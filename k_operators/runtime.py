import argparse
import hashlib
import json
import logging
import os
import platform
import random
import shlex
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LOG = logging.getLogger("kstack_lm")
GAMMA_FLOOR = 0.85

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
USE_AMP = DEVICE == "cuda"
AMP_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float16


def _unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _autocast_context():
    if USE_AMP:
        return torch.amp.autocast(DEVICE, dtype=AMP_DTYPE)
    return nullcontext()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_reproducibility(seed: int, deterministic: bool, deterministic_warn_only: bool, allow_tf32: bool) -> None:
    set_seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
        if torch.cuda.is_available():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        return

    # Preserve historical training behavior unless deterministic mode is requested.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def log_runtime_metadata() -> None:
    amp_dtype_name = str(AMP_DTYPE).replace("torch.", "")
    cuda_version = torch.version.cuda if torch.version.cuda is not None else "N/A"
    cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
    tf32_matmul = torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False
    tf32_cudnn = torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else False
    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        gpu_name = "N/A"
    LOG.info(
        "Environment | torch=%s | cuda_runtime=%s | cudnn=%s | amp_dtype=%s | gpu=%s | det_algos=%s | cudnn_deterministic=%s | cudnn_benchmark=%s | tf32_matmul=%s | tf32_cudnn=%s",
        torch.__version__,
        cuda_version,
        cudnn_version,
        amp_dtype_name,
        gpu_name,
        str(torch.are_deterministic_algorithms_enabled()).lower(),
        str(torch.backends.cudnn.deterministic).lower(),
        str(torch.backends.cudnn.benchmark).lower(),
        str(tf32_matmul).lower(),
        str(tf32_cudnn).lower(),
    )


def _command_string() -> str:
    return " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])


def _run_config_hash(args: argparse.Namespace) -> str:
    args_for_hash = {k: v for k, v in vars(args).items() if k not in {"run_manifest"}}
    payload = json.dumps(args_for_hash, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def maybe_write_run_manifest(path: Path | None, args: argparse.Namespace) -> None:
    if path is None:
        return
    args_for_hash = {k: v for k, v in vars(args).items() if k not in {"run_manifest"}}

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": _command_string(),
        "config_hash": _run_config_hash(args),
        "config_args_for_hash": args_for_hash,
        "args": vars(args),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()) if DEVICE == "cuda" else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOG.info("Run manifest written | path=%s", path)
