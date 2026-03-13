#!/usr/bin/env python3
"""Train and evaluate a K-Stack character-level language model on Tiny Shakespeare."""

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import random
import shlex
import sys
import time
import urllib.request
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "input.txt"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
USE_AMP = DEVICE == "cuda"
AMP_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float16

LOG = logging.getLogger("kstack_lm")
GAMMA_FLOOR = 0.85


def _unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _autocast_context():
    if USE_AMP:
        return torch.amp.autocast(DEVICE, dtype=AMP_DTYPE)
    return nullcontext()


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


def download_shakespeare() -> None:
    if DATA_PATH.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading Tiny Shakespeare to %s", DATA_PATH)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def tokenize_char(text: str) -> Tuple[List[int], Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    ids = [stoi[ch] for ch in text]
    return ids, stoi, itos


def load_shakespeare(val_frac: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, int], Dict[int, str]]:
    download_shakespeare()
    text = DATA_PATH.read_text(encoding="utf-8")

    ids, stoi, itos = tokenize_char(text)
    split = int(len(ids) * (1.0 - val_frac))
    train_ids = torch.tensor(ids[:split], dtype=torch.long)
    val_ids = torch.tensor(ids[split:], dtype=torch.long)

    LOG.info(
        "Dataset ready | vocab=%d | train_tokens=%d | val_tokens=%d",
        len(stoi),
        len(train_ids),
        len(val_ids),
    )
    return train_ids, val_ids, len(stoi), stoi, itos


def get_batch(data: torch.Tensor, window: int, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= window + 1:
        raise ValueError(f"Data length ({len(data)}) must be > window+1 ({window+1}).")

    src_device = data.device
    ix = torch.randint(len(data) - window - 1, (batch_size,), device=src_device)
    offsets = torch.arange(window, device=src_device)
    pos = ix.unsqueeze(1) + offsets.unsqueeze(0)
    flat = pos.reshape(-1)

    x = data.index_select(0, flat).view(batch_size, window)
    y = data.index_select(0, flat + 1).view(batch_size, window)

    dst_device = torch.device(device)
    if x.device != dst_device:
        x = x.to(dst_device, non_blocking=True)
        y = y.to(dst_device, non_blocking=True)
    return x, y


@torch.no_grad()
def eval_deterministic(model: nn.Module, data: torch.Tensor, window: int, batch_size: int) -> Tuple[float, float]:
    model.eval()
    core_model = _unwrap_model(model)
    if hasattr(core_model, "reset_eval_refine_diagnostics"):
        core_model.reset_eval_refine_diagnostics()

    device_obj = torch.device(DEVICE)
    if data.device != device_obj:
        data = data.to(device_obj, non_blocking=True)
    if not data.is_contiguous():
        data = data.contiguous()

    n_tokens = len(data) - 1
    usable_tokens = (n_tokens // window) * window

    total_loss = 0.0
    total_count = 0

    if usable_tokens <= 0:
        return 0.0, 1.0

    x_all = data[:usable_tokens].view(-1, window)
    y_all = data[1 : usable_tokens + 1].view(-1, window)

    for start in range(0, x_all.size(0), batch_size):
        x = x_all[start : start + batch_size]
        y = y_all[start : start + batch_size]

        with _autocast_context():
            logits = core_model(x)
            loss = F.cross_entropy(logits.reshape(-1, core_model.vocab_size), y.reshape(-1), reduction="sum")

        total_loss += loss.item()
        total_count += y.numel()

    ce = total_loss / max(total_count, 1)
    ppl = math.exp(min(ce, 20.0))
    return ce, ppl


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.scale


class MLP(nn.Module):
    def __init__(self, d: int, mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Linear(d, d * mult)
        self.down = nn.Linear(d * mult, d)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(F.gelu(self.up(x))))


class K2Layer(nn.Module):
    """Core causal K2 mixing block with configurable gamma-decay backend."""
    def __init__(
        self,
        window: int,
        d: int,
        rank: int,
        mlp_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        alpha_cap: float = 1.0,
        decay_impl: str = "mask",
    ):
        super().__init__()
        self.window = window
        self.alpha_cap = alpha_cap
        if decay_impl not in ("mask", "block"):
            raise ValueError(f"Unknown decay_impl: {decay_impl}")
        self.decay_impl = decay_impl

        gamma_min = 0.95
        gamma_max = 0.995
        gammas = torch.linspace(gamma_min, gamma_max, rank)
        self.decay_logit = nn.Parameter(torch.log(gammas / (1.0 - gammas)))

        pos = torch.arange(window, dtype=torch.float)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(min=0)
        causal = torch.tril(torch.ones(window, window))
        k_init = torch.exp(-0.15 * dist) * causal
        k_init = k_init / k_init.sum(dim=1, keepdim=True).clamp(min=1e-8)

        self.k_base = nn.Parameter(k_init.contiguous())
        self.k_base_gate_logit = nn.Parameter(torch.tensor(8.0))
        self.register_buffer("causal_mask", causal)
        self.register_buffer("decay_diff", (pos.unsqueeze(1) - pos.unsqueeze(0)).clamp(min=0).contiguous())

        self.u = nn.Parameter(torch.randn(d, rank) * 0.05)
        self.v = nn.Parameter(torch.randn(d, rank) * 0.05)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

        self.proj = nn.Linear(d, d)
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, dropout=mlp_dropout)
        self.drop = nn.Dropout(residual_dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, window, _ = h.shape
        if window > self.window:
            raise ValueError(f"Expected window <= {self.window}, got {window}")

        residual = h
        h_norm = self.norm1(h)

        gate_strength = torch.sigmoid(self.k_base_gate_logit)
        k_base = self.k_base[:window, :window] * self.causal_mask[:window, :window] * gate_strength
        out = torch.matmul(k_base, h_norm)

        q = F.normalize(h_norm @ self.u, dim=-1, eps=1e-8)
        k = F.normalize(h_norm @ self.v, dim=-1, eps=1e-8)

        raw = torch.sigmoid(self.decay_logit)
        gamma_vec = (GAMMA_FLOOR + (1.0 - GAMMA_FLOOR) * raw).to(dtype=h.dtype, device=h.device)
        alpha = (self.alpha_cap * torch.sigmoid(self.alpha_logit)).to(dtype=h.dtype, device=h.device)

        if self.decay_impl == "mask":
            kh = k.unsqueeze(-1) * h_norm.unsqueeze(-2)
            dist = self.decay_diff[:window, :window].to(dtype=h.dtype, device=h.device).unsqueeze(-1)
            log_gamma = torch.log(gamma_vec.clamp(min=1e-8)).view(1, 1, -1)
            decay_mask = torch.exp(dist * log_gamma) * self.causal_mask[:window, :window].unsqueeze(-1)
            causal_inner = torch.einsum("ijr,bjrd->bird", decay_mask, kh)
            out = out + alpha * torch.einsum("bwr,bwrd->bwd", q, causal_inner)
        else:
            gamma = gamma_vec.view(1, 1, -1, 1)
            state = torch.zeros(batch, self.u.size(1), h_norm.size(2), device=h.device, dtype=h.dtype)

            block = min(128, window)
            calc_dtype = torch.float32 if h.dtype in (torch.float16, torch.bfloat16) else h.dtype
            idx = torch.arange(block, device=h.device, dtype=calc_dtype).unsqueeze(1)
            log_gamma = torch.log(gamma_vec.to(dtype=calc_dtype).clamp(min=1e-8)).unsqueeze(0)
            pow_base = torch.exp(idx * log_gamma).to(dtype=h.dtype)
            inv_pow_base = torch.exp(-idx * log_gamma).to(dtype=h.dtype)

            for start in range(0, window, block):
                end = min(start + block, window)
                length = end - start

                h_blk = h_norm[:, start:end]
                q_blk = q[:, start:end]
                k_blk = k[:, start:end]

                pow_l = pow_base[:length].view(1, length, -1, 1)
                inv_pow_l = inv_pow_base[:length].view(1, length, -1, 1)

                kh_blk = k_blk.unsqueeze(-1) * h_blk.unsqueeze(-2)
                prefix = torch.cumsum(kh_blk * inv_pow_l, dim=1)
                state_blk = prefix * pow_l + state.unsqueeze(1) * (pow_l * gamma)

                out[:, start:end].add_(alpha * (q_blk.unsqueeze(-1) * state_blk).sum(dim=2))
                state = state_blk[:, -1]

        out = self.drop(self.proj(out))
        h = residual + out
        h = h + self.mlp(self.norm2(h))
        return h


class K1Layer(nn.Module):
    def __init__(self, d: int, mlp_dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.feature_mix = nn.Linear(d, d)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d, dropout=mlp_dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        residual = h
        out = self.feature_mix(self.norm1(h))
        h = residual + out
        h = h + self.mlp(self.norm2(h))
        return h


class K0Layer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = RMSNorm(d)
        self.gain = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + (self.norm(h) * self.gain + self.bias)


class KStack(nn.Module):
    """Stacked backbone composed of K1/K2/K0 residual blocks."""
    def __init__(
        self,
        window: int,
        d: int,
        rank: int,
        n_k2: int,
        mlp_dropout: float,
        residual_dropout: float,
        alpha_cap: float = 1.0,
        decay_impl: str = "mask",
    ):
        super().__init__()
        layers: List[nn.Module] = [K1Layer(d, mlp_dropout=mlp_dropout)]
        for _ in range(n_k2):
            layers.append(
                K2Layer(
                    window,
                    d,
                    rank,
                    mlp_dropout=mlp_dropout,
                    residual_dropout=residual_dropout,
                    alpha_cap=alpha_cap,
                    decay_impl=decay_impl,
                )
            )
        layers.append(K1Layer(d, mlp_dropout=mlp_dropout))
        layers.append(K0Layer(d))
        self.layers = nn.ModuleList(layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h)
        return h


class KStackModel(nn.Module):
    """Character-level language model with iterative K-Stack refinement."""
    def __init__(
        self,
        vocab_size: int,
        window: int,
        d: int,
        rank: int,
        n_k2: int,
        emb_dropout: float,
        mlp_dropout: float,
        residual_dropout: float,
        head_mode: str = "linear",
        head_mult: int = 6,
        head_dropout: float = 0.0,
        refine_steps: int = 8,
        train_refine_steps: int | None = None,
        alpha_cap: float = 1.0,
        decay_impl: str = "mask",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.head_mode = head_mode
        self.head_mult = head_mult

        self.emb = nn.Embedding(vocab_size, d)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.k_stack = KStack(
            window,
            d,
            rank,
            n_k2,
            mlp_dropout,
            residual_dropout,
            alpha_cap=alpha_cap,
            decay_impl=decay_impl,
        )
        self.norm = RMSNorm(d)
        if head_mode == "linear":
            self.head = nn.Linear(d, vocab_size, bias=False)
            self.head.weight = self.emb.weight
            self.head_drop = nn.Identity()
        elif head_mode == "gelu":
            hidden = max(d, head_mult * d)
            self.head = nn.Sequential(
                RMSNorm(d),
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Linear(hidden, vocab_size, bias=False),
            )
            self.head_drop = nn.Dropout(head_dropout) if head_dropout > 0 else nn.Identity()
        else:
            raise ValueError(f"Unknown head_mode: {head_mode}")

        eta_init = 0.3
        eta_logit_init = math.log(eta_init / (1.0 - eta_init))
        self.eta_logit = nn.Parameter(torch.tensor(eta_logit_init))
        self.refine_steps = max(int(refine_steps), 0)
        if train_refine_steps is None:
            self.train_refine_steps = self.refine_steps
        else:
            self.train_refine_steps = max(int(train_refine_steps), 0)
        self._eval_refine_delta_sums: List[float] = [0.0 for _ in range(self.refine_steps)]
        self._eval_refine_batches = 0

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def eta(self) -> torch.Tensor:
        return torch.sigmoid(self.eta_logit)

    def reset_eval_refine_diagnostics(self) -> None:
        self._eval_refine_delta_sums = [0.0 for _ in range(self.refine_steps)]
        self._eval_refine_batches = 0

    def _record_eval_refine_diagnostics(self, deltas: List[float]) -> None:
        if len(deltas) != self.refine_steps:
            return
        for i, delta in enumerate(deltas):
            self._eval_refine_delta_sums[i] += float(delta)
        self._eval_refine_batches += 1

    def get_eval_refine_diagnostics(self) -> Dict[str, object]:
        eta_value = float(self.eta().detach().cpu().item())
        if self._eval_refine_batches == 0:
            return {"eta": eta_value, "delta_mean": [], "batches": 0}
        means = [v / self._eval_refine_batches for v in self._eval_refine_delta_sums]
        return {"eta": eta_value, "delta_mean": means, "batches": self._eval_refine_batches}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb_drop(self.emb(x))
        steps = self.train_refine_steps if self.training else self.refine_steps
        if steps == 0:
            # Disable iterative refinement: run one plain feedforward pass.
            h = self.k_stack(h)
        else:
            eta = self.eta().to(dtype=h.dtype, device=h.device)
            loop_deltas = [] if not self.training else None
            for _ in range(steps):
                h_new = self.k_stack(h)
                if loop_deltas is not None:
                    delta = (h_new - h).norm() / (h.norm() + 1e-6)
                    loop_deltas.append(float(delta.detach().item()))
                h = h + eta * (h_new - h)
            if loop_deltas is not None:
                self._record_eval_refine_diagnostics(loop_deltas)
        h = self.norm(h)

        if self.head_mode == "gelu":
            h = self.head[0](h)
            h = self.head[1](h)
            h = self.head[2](h)
            h = self.head_drop(h)
            return self.head[3](h)

        return self.head(h)

    def count_params(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb = self.emb.weight.numel()
        stack = sum(p.numel() for p in self.k_stack.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return {"total": total, "embedding": emb, "k_stack": stack, "head": head}


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
                normalized = {k[len(prefix) :]: v for k, v in normalized.items()}
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


@dataclass
class TrainConfig:
    window: int = 128
    d_model: int = 16
    rank: int = 16
    n_k2: int = 3
    batch_size: int = 128
    steps: int = 25000
    lr: float = 4e-3
    beta1: float = 0.8
    beta2: float = 0.999
    lr_floor: float = 1e-4
    warmup_steps: int = 3000
    eval_interval: int = 250
    weight_decay: float = 0.02
    bias_lr_mult: float = 0.5
    norm_lr_mult: float = 0.5
    emb_lr_mult: float = 0.75
    k_logit_lr_mult: float = 0.5
    optimizer_mode: str = "grouped"
    use_fused_adamw: bool = True
    alpha_cap: float = 0.8
    emb_dropout: float = 0.08
    mlp_dropout: float = 0.10
    residual_dropout: float = 0.05
    clip_grad_norm: float = 1.0
    min_improve_ce: float = 1e-4
    plateau_patience_evals: int = 8
    grad_topk: int = 3


def _collect_grad_stats(model: nn.Module, topk: int = 3) -> Dict[str, object]:
    total_g2 = 0.0
    max_abs = 0.0
    nz_count = 0
    elem_count = 0
    per_param = []

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        g_norm = g.norm().item()
        g_abs_max = g.abs().max().item()
        total_g2 += g.pow(2).sum().item()
        max_abs = max(max_abs, g_abs_max)
        nz_count += (g != 0).sum().item()
        elem_count += g.numel()
        per_param.append((name, g_norm))

    total_norm = math.sqrt(total_g2) if total_g2 > 0 else 0.0
    sparsity = 1.0 - (nz_count / elem_count) if elem_count > 0 else 0.0
    top = sorted(per_param, key=lambda x: x[1], reverse=True)[:topk]
    return {
        "total_grad_norm": total_norm,
        "max_abs_grad": max_abs,
        "grad_sparsity": sparsity,
        "top_grad_params": top,
    }


def _build_optimizer_param_groups(model: nn.Module, cfg: TrainConfig) -> List[Dict[str, object]]:
    groups = {
        "core": {"params": [], "weight_decay": cfg.weight_decay, "lr": cfg.lr},
        "bias": {"params": [], "weight_decay": 0.0, "lr": cfg.lr * cfg.bias_lr_mult},
        "norm": {"params": [], "weight_decay": 0.0, "lr": cfg.lr * cfg.norm_lr_mult},
        "emb": {"params": [], "weight_decay": 0.0, "lr": cfg.lr * cfg.emb_lr_mult},
        "k_logit": {"params": [], "weight_decay": 0.0, "lr": cfg.lr * cfg.k_logit_lr_mult},
    }

    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())

        if any(t in name for t in ("decay_logit", "alpha_logit", "k_base_gate_logit", "eta_logit")):
            groups["k_logit"]["params"].append(p)
        elif name.startswith("emb."):
            groups["emb"]["params"].append(p)
        elif name.endswith(".bias"):
            groups["bias"]["params"].append(p)
        elif ("norm" in name) or name.endswith("scale"):
            groups["norm"]["params"].append(p)
        else:
            groups["core"]["params"].append(p)

    param_groups = []
    for group_name, group in groups.items():
        if group["params"]:
            group["name"] = group_name
            param_groups.append(group)
    return param_groups


def _build_optimizer_param_groups_simple(model: nn.Module, cfg: TrainConfig) -> List[Dict[str, object]]:
    decay = []
    no_decay = []
    seen = set()

    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    groups = []
    if decay:
        groups.append({"name": "decay", "params": decay, "weight_decay": cfg.weight_decay, "lr": cfg.lr})
    if no_decay:
        groups.append({"name": "no_decay", "params": no_decay, "weight_decay": 0.0, "lr": cfg.lr})
    return groups


def _collect_k_layer_stats(model: nn.Module) -> str:
    model = _unwrap_model(model)
    if not hasattr(model, "k_stack") or not hasattr(model.k_stack, "layers"):
        return ""

    gates = []
    alphas = []
    gammas = []
    for layer in model.k_stack.layers:
        if hasattr(layer, "k_base_gate_logit"):
            gates.append(torch.sigmoid(layer.k_base_gate_logit).item())
        if hasattr(layer, "alpha_logit"):
            alpha_cap = float(getattr(layer, "alpha_cap", 1.0))
            alphas.append((alpha_cap * torch.sigmoid(layer.alpha_logit)).item())
        if hasattr(layer, "decay_logit"):
            gamma = GAMMA_FLOOR + (1.0 - GAMMA_FLOOR) * torch.sigmoid(layer.decay_logit)
            gammas.append((gamma.min().item(), gamma.mean().item(), gamma.max().item()))

    parts = []
    if gates:
        parts.append(f"gate[min/mean/max]={min(gates):.3f}/{sum(gates)/len(gates):.3f}/{max(gates):.3f}")
    if alphas:
        parts.append(f"alpha[min/mean/max]={min(alphas):.3f}/{sum(alphas)/len(alphas):.3f}/{max(alphas):.3f}")
    if gammas:
        g_min = min(x[0] for x in gammas)
        g_mean = sum(x[1] for x in gammas) / len(gammas)
        g_max = max(x[2] for x in gammas)
        parts.append(f"gamma[min/mean/max]={g_min:.3f}/{g_mean:.3f}/{g_max:.3f}")
    return " | ".join(parts)


def _collect_eval_refine_stats(model: nn.Module) -> str:
    model = _unwrap_model(model)
    if not hasattr(model, "get_eval_refine_diagnostics"):
        return ""

    diag = model.get_eval_refine_diagnostics()
    eta = float(diag.get("eta", float("nan")))
    eta_logit = float(getattr(model, "eta_logit", torch.tensor(float("nan"))).detach().cpu().item())
    delta_mean = diag.get("delta_mean", [])
    batches = int(diag.get("batches", 0))
    if not delta_mean:
        return f"eta={eta:.6f} | eta_logit={eta_logit:.6f} | loop_delta_mean=none | batches={batches}"
    delta_str = ", ".join(f"L{i+1}={v:.2e}" for i, v in enumerate(delta_mean))
    return f"eta={eta:.6f} | eta_logit={eta_logit:.6f} | loop_delta_mean={delta_str} | batches={batches}"


@torch.no_grad()
def _collect_update_weight_stats(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_now: float,
    topk: int = 5,
) -> Dict[str, object]:
    ratios = []
    by_layer = {}

    def layer_key(name: str) -> str:
        if name.startswith("emb."):
            return "emb"
        if name.startswith("head."):
            return "head"
        if name.startswith("norm."):
            return "final_norm"
        if name.startswith("k_stack.layers."):
            parts = name.split(".")
            if len(parts) >= 3:
                return ".".join(parts[:3])  # e.g. k_stack.layers.2
            return "k_stack"
        if name.startswith("k_stack."):
            return "k_stack"
        return "other"

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        w_norm = p.detach().norm().item()
        if w_norm == 0.0:
            continue

        state = optimizer.state.get(p, {})
        exp_avg = state.get("exp_avg", None)
        if exp_avg is None:
            continue

        upd_norm = (lr_now * exp_avg).norm().item()
        ratio = upd_norm / (w_norm + 1e-12)
        ratios.append((name, ratio))
        key = layer_key(name)
        by_layer.setdefault(key, []).append(ratio)

    if not ratios:
        return {
            "top": [],
            "min": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "by_layer": {},
        }

    top = sorted(ratios, key=lambda x: x[1], reverse=True)[:topk]
    vals = [x[1] for x in ratios]
    by_layer_summary = {}
    for key, layer_vals in by_layer.items():
        by_layer_summary[key] = {
            "min": min(layer_vals),
            "mean": sum(layer_vals) / len(layer_vals),
            "max": max(layer_vals),
        }

    return {
        "top": top,
        "min": min(vals),
        "mean": sum(vals) / len(vals),
        "max": max(vals),
        "by_layer": by_layer_summary,
    }


def train_model(model: KStackModel, train_data: torch.Tensor, val_data: torch.Tensor, cfg: TrainConfig, ckpt_path: Path | None) -> float:
    model = model.to(DEVICE)
    train_data = train_data.to(DEVICE, non_blocking=True)
    val_data = val_data.to(DEVICE, non_blocking=True)

    if cfg.optimizer_mode == "grouped":
        param_groups = _build_optimizer_param_groups(model, cfg)
    elif cfg.optimizer_mode == "simple":
        param_groups = _build_optimizer_param_groups_simple(model, cfg)
    else:
        raise ValueError(f"Unknown optimizer_mode: {cfg.optimizer_mode}")

    adamw_kwargs = {
        "lr": cfg.lr,
        "betas": (cfg.beta1, cfg.beta2),
    }
    fused_used = False
    if DEVICE == "cuda" and cfg.use_fused_adamw:
        try:
            optimizer = torch.optim.AdamW(param_groups, fused=True, **adamw_kwargs)
            fused_used = True
        except TypeError:
            optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)
            LOG.warning("fused_adamw requested but not supported by this PyTorch build; falling back to non-fused AdamW.")
    else:
        optimizer = torch.optim.AdamW(param_groups, **adamw_kwargs)

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return max(step, 1) / max(cfg.warmup_steps, 1)
        t = (step - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))
        floor_ratio = cfg.lr_floor / cfg.lr
        return floor_ratio + (1.0 - floor_ratio) * cosine

    start_step = 0
    best_ppl = float("inf")
    if ckpt_path and ckpt_path.exists():
        start_step, best_ppl = load_checkpoint(ckpt_path, model, optimizer)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
        last_epoch=start_step - 1,
    )
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    LOG.info(
        "Training start | device=%s | steps=%d | window=%d | batch=%d | lr=%.2e | betas=(%.3f, %.3f) | warmup=%d | opt_mode=%s | fused_adamw=%s",
        DEVICE,
        cfg.steps,
        cfg.window,
        cfg.batch_size,
        cfg.lr,
        cfg.beta1,
        cfg.beta2,
        cfg.warmup_steps,
        cfg.optimizer_mode,
        str(fused_used).lower(),
    )
    group_parts = []
    for g in optimizer.param_groups:
        n_params = sum(p.numel() for p in g["params"])
        group_parts.append(f"{g.get('name', 'group')}[n={n_params},lr={g['lr']:.2e},wd={g['weight_decay']:.2e}]")
    LOG.info("Optimizer groups | %s", " | ".join(group_parts))

    step_times: List[float] = []
    compile_warmup_pending = model is not _unwrap_model(model)
    if compile_warmup_pending:
        LOG.info("Compile warmup | first train step will include graph capture and is excluded from speed metrics.")
    train_loss_ema = None
    best_ce = float("inf")
    stale_evals = 0
    grad_norm_hist: deque[float] = deque(maxlen=cfg.plateau_patience_evals)
    val_ce_hist: deque[float] = deque(maxlen=cfg.plateau_patience_evals)

    for step in range(start_step, cfg.steps + 1):
        t0 = time.time()
        train_loss = float("nan")
        raw_grad_norm = float("nan")

        if step > 0:
            model.train()
            x, y = get_batch(train_data, cfg.window, cfg.batch_size, DEVICE)

            with _autocast_context():
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                raw_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm).item())
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                raw_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm).item())
                optimizer.step()

            scheduler.step()
            train_loss = float(loss.item())
            train_loss_ema = train_loss if train_loss_ema is None else 0.95 * train_loss_ema + 0.05 * train_loss
            if not math.isnan(raw_grad_norm):
                grad_norm_hist.append(raw_grad_norm)

        step_elapsed = time.time() - t0
        if step > 0:
            if compile_warmup_pending:
                LOG.info("Compile warmup done | step=%d | warmup_ms=%.1f", step, step_elapsed * 1000.0)
                compile_warmup_pending = False
            else:
                step_times.append(step_elapsed)

        if step % cfg.eval_interval == 0:
            ce, ppl = eval_deterministic(model, val_data, cfg.window, cfg.batch_size)
            val_ce_hist.append(ce)
            improved = ce < (best_ce - cfg.min_improve_ce)
            if improved:
                best_ce = ce
                stale_evals = 0
            else:
                stale_evals += 1

            if ppl < best_ppl:
                best_ppl = ppl
                if ckpt_path is not None:
                    save_checkpoint(ckpt_path, model, optimizer, step, best_ppl)

            if step_times:
                avg_ms = (sum(step_times) / len(step_times)) * 1000.0
                tok_s = (cfg.batch_size * cfg.window) / max(avg_ms / 1000.0, 1e-9)
                avg_ms_str = f"{avg_ms:.1f}"
                tok_s_str = f"{tok_s:.0f}"
            else:
                avg_ms_str = "N/A"
                tok_s_str = "N/A"
            lr_now = optimizer.param_groups[0]["lr"]
            train_str = "N/A" if math.isnan(train_loss) else f"{train_loss:.4f}"
            train_ema_str = "N/A" if train_loss_ema is None else f"{train_loss_ema:.4f}"
            grad_stats = _collect_grad_stats(model, topk=cfg.grad_topk)
            top_grad = ", ".join(f"{n}={v:.2e}" for n, v in grad_stats["top_grad_params"]) or "none"
            k_stats = _collect_k_layer_stats(model)
            eval_refine_stats = _collect_eval_refine_stats(model)
            clip_hit = (not math.isnan(raw_grad_norm)) and (raw_grad_norm > cfg.clip_grad_norm)
            grad_window = sum(grad_norm_hist) / len(grad_norm_hist) if grad_norm_hist else float("nan")
            val_window_drop = (val_ce_hist[0] - val_ce_hist[-1]) if len(val_ce_hist) >= 2 else float("nan")
            uw_stats = _collect_update_weight_stats(model, optimizer, lr_now, topk=5)
            top_uw = ", ".join(f"{n}={v:.2e}" for n, v in uw_stats["top"]) or "none"
            layer_order = sorted(
                uw_stats["by_layer"].keys(),
                key=lambda k: (
                    0 if k == "emb" else
                    1 if k.startswith("k_stack.layers.") else
                    2 if k == "k_stack" else
                    3 if k == "final_norm" else
                    4 if k == "head" else
                    5
                , k)
            )
            uw_by_layer = " | ".join(
                f"{k}[{uw_stats['by_layer'][k]['min']:.1e}/{uw_stats['by_layer'][k]['mean']:.1e}/{uw_stats['by_layer'][k]['max']:.1e}]"
                for k in layer_order
            ) or "none"

            LOG.info(
                "step=%5d | train_ce=%s | train_ce_ema=%s | val_ce=%.4f | val_ppl=%.2f | best_ppl=%.2f | lr=%.2e | %s ms/step | %s tok/s | gnorm=%.2e | gnorm_clip=%.2e | clip_hit=%s | gmax=%.2e | g_sparsity=%.2f%% | stale=%d | uw[min/mean/max]=%.2e/%.2e/%.2e",
                step,
                train_str,
                train_ema_str,
                ce,
                ppl,
                best_ppl,
                lr_now,
                avg_ms_str,
                tok_s_str,
                grad_stats["total_grad_norm"],
                raw_grad_norm,
                str(clip_hit).lower(),
                grad_stats["max_abs_grad"],
                grad_stats["grad_sparsity"] * 100.0,
                stale_evals,
                uw_stats["min"],
                uw_stats["mean"],
                uw_stats["max"],
            )
            LOG.info("grad_top_params | %s", top_grad)
            LOG.info("update_weight_top_params | %s", top_uw)
            LOG.info("update_weight_by_layer[min/mean/max] | %s", uw_by_layer)
            if k_stats:
                LOG.info("layer_stats | %s", k_stats)
            if eval_refine_stats:
                LOG.info("eval_refinement | %s", eval_refine_stats)
            if stale_evals >= cfg.plateau_patience_evals:
                LOG.warning(
                    "Plateau detected | stale_evals=%d | val_ce_delta_window=%.4f | avg_gnorm_window=%.2e | lr=%.2e | clip_hit=%s",
                    stale_evals,
                    val_window_drop,
                    grad_window,
                    lr_now,
                    str(clip_hit).lower(),
                )
            step_times.clear()

    return best_ppl


@torch.no_grad()
def sample_text(
    model: KStackModel,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    prompt: str,
    max_new_tokens: int,
    window: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int = 256,
    prompt_lock_chars: int = 0,
) -> str:
    model.eval()
    core_model = _unwrap_model(model)
    context = [stoi[ch] for ch in prompt if ch in stoi]
    if not context:
        context = [0]

    x = torch.tensor(context, dtype=torch.long, device=DEVICE).unsqueeze(0)
    for _ in range(max_new_tokens):
        if prompt_lock_chars > 0 and x.size(1) > window:
            lock_len = min(int(prompt_lock_chars), max(window - 1, 0), x.size(1))
            tail_len = window - lock_len
            if tail_len > 0:
                x_cond = torch.cat([x[:, :lock_len], x[:, -tail_len:]], dim=1)
            else:
                x_cond = x[:, :window]
        else:
            x_cond = x[:, -window:]
        logits = core_model(x_cond)[:, -1, :] / max(temperature, 1e-6)
        if repetition_penalty > 1.0:
            if repetition_window > 0:
                seen = x[:, -min(int(repetition_window), x.size(1)) :]
            else:
                seen = x
            for b in range(logits.size(0)):
                seen_ids = torch.unique(seen[b])
                seen_logits = logits[b, seen_ids]
                seen_logits = torch.where(
                    seen_logits > 0,
                    seen_logits / float(repetition_penalty),
                    seen_logits * float(repetition_penalty),
                )
                logits[b, seen_ids] = seen_logits
        if top_k is not None and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            top_vals, _ = torch.topk(logits, k, dim=-1)
            kth = top_vals[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove_mask = cumulative_probs > float(top_p)
            sorted_remove_mask[:, 0] = False
            remove_mask = torch.zeros_like(sorted_remove_mask, dtype=torch.bool)
            remove_mask.scatter_(1, sorted_indices, sorted_remove_mask)
            logits = logits.masked_fill(remove_mask, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    return "".join(itos[int(i)] for i in x[0].tolist())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a K-Stack character-level language model on Tiny Shakespeare.")
    p.add_argument("--steps", type=int, default=25000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--n-k2", type=int, default=4)
    p.add_argument("--head-mode", type=str, choices=["linear", "gelu"], default="linear")
    p.add_argument("--head-mult", type=int, default=6)
    p.add_argument("--head-dropout", type=float, default=0.10)
    p.add_argument(
        "--refine-steps",
        type=int,
        default=8,
        help="Iterative refinement steps at eval/inference. 0 runs one feedforward K-stack pass.",
    )
    p.add_argument(
        "--train-refine-steps",
        type=int,
        default=None,
        help="Refinement steps during training only. 0 runs one feedforward K-stack pass. Defaults to --refine-steps.",
    )
    p.add_argument(
        "--decay-impl",
        type=str,
        choices=["mask", "block"],
        default="mask",
        help="Gamma-decay backend: 'mask' is fastest, 'block' uses less memory.",
    )
    p.add_argument("--alpha-cap", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--lr-floor", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.8)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--warmup-steps", type=int, default=3000)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--bias-lr-mult", type=float, default=0.5)
    p.add_argument("--norm-lr-mult", type=float, default=0.5)
    p.add_argument("--emb-lr-mult", type=float, default=0.75)
    p.add_argument("--k-logit-lr-mult", type=float, default=0.5)
    p.add_argument("--optimizer-mode", type=str, choices=["simple", "grouped"], default="grouped")
    p.add_argument(
        "--fused-adamw",
        dest="fused_adamw",
        action="store_true",
        help="Use fused AdamW on CUDA (faster, may slightly change optimization trajectory).",
    )
    p.add_argument(
        "--no-fused-adamw",
        dest="fused_adamw",
        action="store_false",
        help="Disable fused AdamW on CUDA.",
    )
    p.set_defaults(fused_adamw=True)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable strict deterministic algorithms (disables TF32 and cudnn benchmark).",
    )
    p.add_argument(
        "--deterministic-warn-only",
        action="store_true",
        help="With --deterministic, warn instead of error on nondeterministic ops.",
    )
    p.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 kernels on CUDA (slower, sometimes more numerically stable).",
    )
    p.add_argument(
        "--run-manifest",
        type=str,
        default=None,
        help="Optional JSON path to store full run config/runtime metadata.",
    )
    p.add_argument(
        "--strict-repro",
        action="store_true",
        help="Force strict reproducibility (disables compile and fused AdamW, enables deterministic mode and disables TF32).",
    )
    p.add_argument(
        "--ckpt",
        "--checkpoint",
        dest="ckpt",
        type=str,
        default=None,
        help="Checkpoint file path. If omitted, checkpoints are not saved.",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Load a checkpoint (--checkpoint/--ckpt) and run deterministic evaluation only.",
    )
    p.add_argument(
        "--eval-refine-steps",
        type=int,
        default=None,
        help="Override refine steps for eval-only. If omitted, uses --refine-steps.",
    )
    p.add_argument("--compile", action="store_true", help="Enable torch.compile for faster steady-state training.")
    p.add_argument(
        "--compile-mode",
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="torch.compile mode: default starts quickly, max-autotune can be slower to warm up.",
    )
    p.add_argument("--sample", action="store_true", help="Generate a sample after training.")
    p.add_argument("--prompt", type=str, default="To be, or not to be")
    p.add_argument("--sample-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for --sample.")
    p.add_argument("--top-k", type=int, default=0, help="Top-k sampling for --sample. 0 disables top-k.")
    p.add_argument("--top-p", type=float, default=0.0, help="Top-p (nucleus) sampling for --sample. 0 disables top-p.")
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (>1 discourages repeated tokens). 1.0 disables.",
    )
    p.add_argument(
        "--repetition-window",
        type=int,
        default=256,
        help="Recent token window for repetition penalty. 0 means full generated context.",
    )
    p.add_argument(
        "--prompt-lock-chars",
        type=int,
        default=0,
        help="Keep first N prompt chars in conditioning window during long generation.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)
    if args.strict_repro:
        args.deterministic = True
        args.deterministic_warn_only = False
        args.no_tf32 = True
        args.fused_adamw = False
        if args.compile:
            LOG.warning("--strict-repro enabled: overriding --compile to disabled for exact run-to-run reproducibility.")
        args.compile = False
    if args.deterministic_warn_only and not args.deterministic:
        LOG.warning("--deterministic-warn-only has effect only with --deterministic.")
    configure_reproducibility(
        seed=args.seed,
        deterministic=args.deterministic,
        deterministic_warn_only=args.deterministic_warn_only,
        allow_tf32=not args.no_tf32,
    )

    LOG.info(
        "Runtime | device=%s | amp=%s | seed=%d | strict_repro=%s | deterministic=%s | deterministic_warn_only=%s | tf32=%s",
        DEVICE,
        USE_AMP,
        args.seed,
        str(args.strict_repro).lower(),
        str(args.deterministic).lower(),
        str(args.deterministic_warn_only).lower(),
        str(torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False).lower(),
    )
    LOG.info("Decay backend | impl=%s", args.decay_impl)
    LOG.info("Run config | hash=%s", _run_config_hash(args))
    LOG.info("Command | %s", _command_string())
    log_runtime_metadata()
    maybe_write_run_manifest(Path(args.run_manifest) if args.run_manifest else None, args)

    train_data, val_data, vocab_size, stoi, itos = load_shakespeare(val_frac=0.1)

    cfg = TrainConfig(
        window=args.window,
        d_model=args.d_model,
        rank=args.rank,
        n_k2=args.n_k2,
        alpha_cap=args.alpha_cap,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        lr_floor=args.lr_floor,
        beta1=args.beta1,
        beta2=args.beta2,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bias_lr_mult=args.bias_lr_mult,
        norm_lr_mult=args.norm_lr_mult,
        emb_lr_mult=args.emb_lr_mult,
        k_logit_lr_mult=args.k_logit_lr_mult,
        optimizer_mode=args.optimizer_mode,
        use_fused_adamw=args.fused_adamw,
        eval_interval=args.eval_interval,
    )

    model = KStackModel(
        vocab_size=vocab_size,
        window=cfg.window,
        d=cfg.d_model,
        rank=cfg.rank,
        n_k2=cfg.n_k2,
        emb_dropout=cfg.emb_dropout,
        mlp_dropout=cfg.mlp_dropout,
        residual_dropout=cfg.residual_dropout,
        head_mode=args.head_mode,
        head_mult=args.head_mult,
        head_dropout=args.head_dropout,
        refine_steps=args.refine_steps,
        train_refine_steps=args.train_refine_steps,
        alpha_cap=cfg.alpha_cap,
        decay_impl=args.decay_impl,
    )

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
        LOG.info("Compile config | mode=%s", args.compile_mode)
    elif not args.deterministic:
        LOG.warning(
            "Non-deterministic fast mode active: run-to-run CE can drift with AMP/fused/compile behavior. "
            "Use --strict-repro for exact reproducibility."
        )

    model_for_stats = _unwrap_model(model)
    params = model_for_stats.count_params() if hasattr(model_for_stats, "count_params") else {}
    if params:
        LOG.info(
            "Model params | total=%s | embedding=%s | k_stack=%s | head=%s",
            f"{params['total']:,}",
            f"{params['embedding']:,}",
            f"{params['k_stack']:,}",
            f"{params['head']:,}",
        )
        LOG.info(
            "Model config | head_mode=%s | head_mult=%d | head_dropout=%.2f | alpha_cap=%.2f | refine_steps[train/eval]=%d/%d",
            args.head_mode,
            args.head_mult,
            args.head_dropout,
            cfg.alpha_cap,
            model_for_stats.train_refine_steps,
            model_for_stats.refine_steps,
        )
        if model_for_stats.train_refine_steps == 0:
            LOG.warning("train_refine_steps=0: iterative refinement is disabled during training, so eta is not optimized.")

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if args.eval_only:
        if ckpt_path is None:
            raise ValueError("--eval-only requires --ckpt.")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = model.to(DEVICE)
        loaded_step, loaded_best_ppl = load_model_checkpoint(ckpt_path, model)
        core_model = _unwrap_model(model)
        if args.eval_refine_steps is not None:
            core_model.refine_steps = max(int(args.eval_refine_steps), 0)
            core_model.reset_eval_refine_diagnostics()

        ce, ppl = eval_deterministic(model, val_data, cfg.window, cfg.batch_size)
        loaded_step_str = "N/A" if loaded_step is None else str(loaded_step)
        loaded_best_ppl_str = "N/A" if loaded_best_ppl is None else f"{loaded_best_ppl:.2f}"
        LOG.info(
            "Eval only | step=%s | ckpt_best_ppl=%s | refine_steps=%d | val_ce=%.4f | val_ppl=%.2f",
            loaded_step_str,
            loaded_best_ppl_str,
            core_model.refine_steps,
            ce,
            ppl,
        )
        eval_refine_stats = _collect_eval_refine_stats(model)
        if eval_refine_stats:
            LOG.info("eval_refinement | %s", eval_refine_stats)

        if args.sample:
            top_k = args.top_k if args.top_k > 0 else None
            top_p = args.top_p if 0.0 < args.top_p < 1.0 else None
            text = sample_text(
                model,
                stoi,
                itos,
                args.prompt,
                args.sample_tokens,
                cfg.window,
                temperature=args.temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=max(args.repetition_penalty, 1.0),
                repetition_window=args.repetition_window,
                prompt_lock_chars=max(args.prompt_lock_chars, 0),
            )
            LOG.info("Sample:\n%s", text)
        return

    best_ppl = train_model(model, train_data, val_data, cfg, ckpt_path)
    LOG.info("Training complete | best_perplexity=%.2f", best_ppl)

    if args.sample:
        top_k = args.top_k if args.top_k > 0 else None
        top_p = args.top_p if 0.0 < args.top_p < 1.0 else None
        text = sample_text(
            model.to(DEVICE),
            stoi,
            itos,
            args.prompt,
            args.sample_tokens,
            cfg.window,
            temperature=args.temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=max(args.repetition_penalty, 1.0),
            repetition_window=args.repetition_window,
            prompt_lock_chars=max(args.prompt_lock_chars, 0),
        )
        LOG.info("Sample:\n%s", text)


if __name__ == "__main__":
    main()
