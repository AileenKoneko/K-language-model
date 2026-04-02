import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import load_checkpoint, load_checkpoint_metadata, save_checkpoint
from .data import get_batch, get_rollout_batch
from .model import KStackModel
from .runtime import AMP_DTYPE, DEVICE, LOG, USE_AMP, _autocast_context, _unwrap_model


def ce_to_bpc(ce: float) -> float:
    return float(ce) / math.log(2.0)


@torch.no_grad()
def eval_deterministic(model: nn.Module, data: torch.Tensor, window: int, batch_size: int) -> tuple[float, float]:
    model.eval()
    core_model = _unwrap_model(model)

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
    y_all = data[1: usable_tokens + 1].view(-1, window)

    for start in range(0, x_all.size(0), batch_size):
        x = x_all[start: start + batch_size]
        y = y_all[start: start + batch_size]

        with _autocast_context():
            # Use unwrapped model for deterministic eval behavior (historical baseline).
            loss = core_model(x, targets=y, reduction="sum")

        total_loss += loss.item()
        total_count += y.numel()

    ce = total_loss / max(total_count, 1)
    ppl = math.exp(min(ce, 20.0))
    return ce, ppl


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
    diagnostics: bool = False
    report_bpc: bool = False
    future_summary_horizons: tuple[int, ...] = ()
    future_summary_lambda: float = 0.05
    future_summary_lambda_min: float = 0.0
    future_summary_ce_target: float | None = None
    future_summary_ce_anchor: float | None = None
    future_summary_start_step: int = 0
    future_summary_eval_batches: int = 16
    rollout_horizon: int = 0
    rollout_lambda: float = 0.2
    rollout_start_step: int = 0
    rollout_mode: str = "argmax"
    semantic_lambda: float = 0.05
    semantic_start_step: int = 0
    rollout_eval_batches: int = 16
    rollout_useful_ce_tol: float = 0.05


@dataclass
class StepLossStats:
    total: float
    ce: float
    future: float
    future_lambda: float
    rollout: float
    semantic: float
    phase: str


@dataclass
class CheckpointTargets:
    best_ppl: Path
    best_rollout: Path
    best_useful: Path


def _checkpoint_variant_path(base: Path, variant: str) -> Path:
    suffix = "".join(base.suffixes)
    stem = base.name[: -len(suffix)] if suffix else base.name
    if not stem:
        stem = base.name
    filename = f"{stem}.{variant}{suffix}" if suffix else f"{stem}.{variant}"
    return base.with_name(filename)


def _resolve_checkpoint_targets(ckpt_path: Path | None) -> CheckpointTargets | None:
    if ckpt_path is None:
        return None
    return CheckpointTargets(
        best_ppl=ckpt_path,
        best_rollout=_checkpoint_variant_path(ckpt_path, "best_rollout"),
        best_useful=_checkpoint_variant_path(ckpt_path, "best_useful"),
    )


def _metric_from_checkpoint(path: Path, key: str) -> float:
    meta = load_checkpoint_metadata(path)
    value = meta.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _resolve_loss_phase(cfg: TrainConfig, step: int) -> tuple[bool, bool, bool, str]:
    future_active = (
        bool(cfg.future_summary_horizons)
        and cfg.future_summary_lambda > 0.0
        and step >= max(int(cfg.future_summary_start_step), 0)
    )
    rollout_active = (
        cfg.rollout_horizon > 0
        and cfg.rollout_lambda > 0.0
        and step >= max(int(cfg.rollout_start_step), 0)
    )
    semantic_active = (
        rollout_active
        and cfg.semantic_lambda > 0.0
        and step >= max(int(cfg.semantic_start_step), 0)
    )
    phase_parts: List[str] = []
    if rollout_active:
        phase_parts.append("rollout")
    if semantic_active:
        phase_parts.append("semantic")
    if future_active:
        phase_parts.append("future")
    phase = "+".join(phase_parts) if phase_parts else "ce"
    return rollout_active, semantic_active, future_active, phase


def _resolve_future_lambda(
    cfg: TrainConfig,
    ce_reference: float | None,
    anchor: float | None,
) -> tuple[float, float | None]:
    lambda_max = max(float(cfg.future_summary_lambda), 0.0)
    if not cfg.future_summary_horizons or lambda_max <= 0.0:
        return 0.0, anchor

    lambda_min = min(max(float(cfg.future_summary_lambda_min), 0.0), lambda_max)
    ce_target = cfg.future_summary_ce_target
    if ce_target is None or not math.isfinite(float(ce_target)):
        return lambda_max, anchor
    ce_target = float(ce_target)

    resolved_anchor = anchor
    if resolved_anchor is None and cfg.future_summary_ce_anchor is not None and math.isfinite(float(cfg.future_summary_ce_anchor)):
        resolved_anchor = float(cfg.future_summary_ce_anchor)

    ce_value = None
    if ce_reference is not None:
        try:
            ce_candidate = float(ce_reference)
        except (TypeError, ValueError):
            ce_candidate = float("nan")
        if math.isfinite(ce_candidate):
            ce_value = ce_candidate

    if resolved_anchor is None and ce_value is not None:
        resolved_anchor = max(ce_value, ce_target + 1e-6)

    if resolved_anchor is None:
        return lambda_max, None

    resolved_anchor = max(float(resolved_anchor), ce_target + 1e-6)
    if ce_value is None:
        return lambda_max, resolved_anchor

    progress = (ce_value - ce_target) / max(resolved_anchor - ce_target, 1e-6)
    progress = min(max(progress, 0.0), 1.0)
    lambda_eff = lambda_min + (lambda_max - lambda_min) * progress
    return lambda_eff, resolved_anchor


def _compute_future_summary_loss(
    core_model: KStackModel,
    hidden: torch.Tensor,
    horizons: tuple[int, ...],
) -> torch.Tensor:
    if not horizons or hidden.ndim != 3 or hidden.size(1) <= 1:
        return torch.zeros((), device=hidden.device, dtype=torch.float32)

    predictors = core_model.predict_future_summaries(hidden)
    if not predictors:
        return torch.zeros((), device=hidden.device, dtype=torch.float32)

    hidden_float = hidden.float()
    future = hidden_float.detach()[:, 1:, :]
    if future.size(1) <= 0:
        return hidden_float.new_zeros(())

    zero = torch.zeros(future.size(0), 1, future.size(2), device=future.device, dtype=future.dtype)
    cumsum = torch.cat((zero, future.cumsum(dim=1)), dim=1)
    losses = []
    for horizon in horizons:
        if horizon <= 0 or hidden.size(1) <= horizon or horizon not in predictors:
            continue
        target_sum = cumsum[:, horizon:, :] - cumsum[:, :-horizon, :]
        target_avg = target_sum / float(horizon)
        current = hidden_float.detach()[:, : hidden.size(1) - horizon, :]
        target_direction = target_avg - current
        pred_direction = predictors[horizon][:, : hidden.size(1) - horizon, :].float()

        pred_direction = F.normalize(pred_direction, dim=-1, eps=1e-8)
        target_direction = F.normalize(target_direction, dim=-1, eps=1e-8)
        losses.append(1.0 - (pred_direction * target_direction).sum(dim=-1).mean())

    if not losses:
        return hidden_float.new_zeros(())
    return torch.stack(losses, dim=0).mean()


def _select_rollout_tokens(scores: torch.Tensor, mode: str) -> torch.Tensor:
    mode = str(mode).strip().lower()
    detached = scores.detach()
    if mode == "argmax":
        return detached.argmax(dim=-1, keepdim=True)
    if mode == "sample":
        probs = F.softmax(detached, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    raise ValueError(f"Unsupported rollout_mode: {mode}")


@torch.no_grad()
def _teacher_rollout_pooled_hidden(core_model: KStackModel, rollout_prefix: torch.Tensor, true_chunk: torch.Tensor) -> torch.Tensor:
    was_training = core_model.training
    core_model.eval()
    context = rollout_prefix
    pooled_steps = []
    try:
        for idx in range(true_chunk.size(1)):
            context_window = context[:, -core_model.window:]
            hidden = core_model.hidden_states(context_window)
            pooled_steps.append(hidden[:, -1, :])
            context = torch.cat([context, true_chunk[:, idx: idx + 1]], dim=1)
    finally:
        core_model.train(was_training)
    return torch.stack(pooled_steps, dim=1).mean(dim=1)


def _compute_rollout_losses(
    core_model: KStackModel,
    rollout_prefix: torch.Tensor,
    true_chunk: torch.Tensor,
    *,
    rollout_mode: str,
    include_semantic: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if true_chunk.ndim != 2 or true_chunk.size(1) <= 0:
        zero = torch.zeros((), device=rollout_prefix.device, dtype=torch.float32)
        return zero, zero

    rollout_losses = []
    generated_hidden = []
    context = rollout_prefix
    for idx in range(true_chunk.size(1)):
        context_window = context[:, -core_model.window:]
        hidden = core_model.hidden_states(context_window)
        scores = core_model.scores_from_hidden(hidden)[:, -1, :]
        rollout_losses.append(F.cross_entropy(scores.float(), true_chunk[:, idx], reduction="mean"))
        generated_hidden.append(hidden[:, -1, :])
        next_token = _select_rollout_tokens(scores, rollout_mode)
        context = torch.cat([context, next_token], dim=1)

    rollout_ce = torch.stack(rollout_losses, dim=0).mean()
    semantic_loss = rollout_ce.new_zeros(())
    if include_semantic:
        target_pool = _teacher_rollout_pooled_hidden(core_model, rollout_prefix, true_chunk).detach()
        gen_pool = torch.stack(generated_hidden, dim=1).mean(dim=1)
        gen_pool = F.normalize(gen_pool.float(), dim=-1, eps=1e-8)
        target_pool = F.normalize(target_pool.float(), dim=-1, eps=1e-8)
        semantic_loss = (1.0 - (gen_pool * target_pool).sum(dim=-1)).mean()
    return rollout_ce, semantic_loss


def _compute_train_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    cfg: TrainConfig,
    step: int,
    *,
    future_lambda: float,
) -> tuple[torch.Tensor, StepLossStats]:
    rollout_active, semantic_active, future_active, phase = _resolve_loss_phase(cfg, step)
    if not rollout_active and not future_active:
        x, y = get_batch(train_data, cfg.window, cfg.batch_size, DEVICE)
        ce_loss = model(x, targets=y)
        ce_value = float(ce_loss.detach().item())
        return ce_loss, StepLossStats(
            total=ce_value,
            ce=ce_value,
            future=float("nan"),
            future_lambda=float("nan"),
            rollout=float("nan"),
            semantic=float("nan"),
            phase=phase,
        )

    core_model = _unwrap_model(model)
    if rollout_active:
        x, y, rollout_prefix, true_chunk = get_rollout_batch(
            train_data,
            cfg.window,
            cfg.rollout_horizon,
            cfg.batch_size,
            DEVICE,
        )
    else:
        x, y = get_batch(train_data, cfg.window, cfg.batch_size, DEVICE)
        rollout_prefix = x
        true_chunk = x[:, :0]

    hidden = core_model.hidden_states(x)
    ce_loss = core_model.loss_from_hidden(hidden, y)
    future_summary_loss = (
        _compute_future_summary_loss(core_model, hidden, cfg.future_summary_horizons)
        if future_active
        else ce_loss.new_zeros(())
    )

    rollout_ce, semantic_loss = _compute_rollout_losses(
        core_model,
        rollout_prefix,
        true_chunk,
        rollout_mode=cfg.rollout_mode,
        include_semantic=semantic_active,
    ) if rollout_active else (ce_loss.new_zeros(()), ce_loss.new_zeros(()))

    total_loss = ce_loss + future_lambda * future_summary_loss
    if rollout_active:
        total_loss = total_loss + cfg.rollout_lambda * rollout_ce
    if semantic_active:
        total_loss = total_loss + cfg.semantic_lambda * semantic_loss
    return total_loss, StepLossStats(
        total=float(total_loss.detach().item()),
        ce=float(ce_loss.detach().item()),
        future=float(future_summary_loss.detach().item()) if future_active else float("nan"),
        future_lambda=float(future_lambda) if future_active else float("nan"),
        rollout=float(rollout_ce.detach().item()) if rollout_active else float("nan"),
        semantic=float(semantic_loss.detach().item()) if semantic_active else float("nan"),
        phase=phase,
    )


@torch.no_grad()
def eval_rollout_deterministic(
    model: nn.Module,
    data: torch.Tensor,
    window: int,
    batch_size: int,
    horizon: int,
    max_batches: int,
) -> tuple[float, float, int]:
    if horizon <= 0 or max_batches <= 0:
        return float("nan"), float("nan"), 0

    model.eval()
    core_model = _unwrap_model(model)

    device_obj = torch.device(DEVICE)
    if data.device != device_obj:
        data = data.to(device_obj, non_blocking=True)
    if not data.is_contiguous():
        data = data.contiguous()

    total_len = window + horizon + 1
    available = (len(data) - 1) // total_len
    if available <= 0:
        return float("nan"), float("nan"), 0

    target_examples = min(int(available), int(batch_size) * int(max_batches))
    if target_examples <= 0:
        return float("nan"), float("nan"), 0

    if target_examples == available:
        example_ids = torch.arange(available, device=data.device, dtype=torch.long)
    else:
        example_ids = torch.linspace(
            0,
            available - 1,
            steps=target_examples,
            device=data.device,
            dtype=torch.float32,
        ).round().to(dtype=torch.long)
    starts = example_ids * total_len
    offsets = torch.arange(total_len, device=data.device, dtype=torch.long)
    seq = data.index_select(0, (starts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)).view(target_examples, total_len)

    total_rollout = 0.0
    total_semantic = 0.0
    total_count = 0
    for start in range(0, target_examples, batch_size):
        batch_seq = seq[start: start + batch_size]
        rollout_prefix = batch_seq[:, 1: window + 1]
        true_chunk = batch_seq[:, window + 1:]
        with _autocast_context():
            rollout_ce, semantic_loss = _compute_rollout_losses(
                core_model,
                rollout_prefix,
                true_chunk,
                rollout_mode="argmax",
                include_semantic=True,
            )
        count = int(batch_seq.size(0))
        total_rollout += float(rollout_ce.item()) * count
        total_semantic += float(semantic_loss.item()) * count
        total_count += count

    if total_count <= 0:
        return float("nan"), float("nan"), 0
    return total_rollout / total_count, total_semantic / total_count, total_count


@torch.no_grad()
def eval_future_summary_deterministic(
    model: nn.Module,
    data: torch.Tensor,
    window: int,
    batch_size: int,
    horizons: tuple[int, ...],
    max_batches: int,
) -> tuple[float, int]:
    if not horizons or max_batches <= 0 or window <= min(horizons):
        return float("nan"), 0

    model.eval()
    core_model = _unwrap_model(model)

    device_obj = torch.device(DEVICE)
    if data.device != device_obj:
        data = data.to(device_obj, non_blocking=True)
    if not data.is_contiguous():
        data = data.contiguous()

    n_tokens = len(data) - 1
    usable_tokens = (n_tokens // window) * window
    if usable_tokens <= 0:
        return float("nan"), 0

    x_all = data[:usable_tokens].view(-1, window)
    total_examples = int(x_all.size(0))
    target_examples = min(total_examples, int(batch_size) * int(max_batches))
    if target_examples <= 0:
        return float("nan"), 0

    if target_examples == total_examples:
        example_ids = torch.arange(total_examples, device=x_all.device, dtype=torch.long)
    else:
        example_ids = torch.linspace(
            0,
            total_examples - 1,
            steps=target_examples,
            device=x_all.device,
            dtype=torch.float32,
        ).round().to(dtype=torch.long)
    x_eval = x_all.index_select(0, example_ids)

    total_future = 0.0
    total_count = 0
    for start in range(0, x_eval.size(0), batch_size):
        x = x_eval[start: start + batch_size]
        with _autocast_context():
            hidden = core_model.hidden_states(x)
            future_loss = _compute_future_summary_loss(core_model, hidden, horizons)
        count = int(x.size(0))
        total_future += float(future_loss.item()) * count
        total_count += count

    if total_count <= 0:
        return float("nan"), 0
    return total_future / total_count, total_count


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
    def normalize_name(name: str) -> str:
        # torch.compile / wrappers can prefix names (e.g. "_orig_mod.").
        while True:
            if name.startswith("_orig_mod."):
                name = name[len("_orig_mod."):]
                continue
            if name.startswith("module."):
                name = name[len("module."):]
                continue
            break
        return name

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
        norm_name = normalize_name(name)

        if any(t in norm_name for t in ("decay_logit", "alpha_logit", "rho_logit", "k_base_gate_logit")):
            groups["k_logit"]["params"].append(p)
        elif (
            norm_name.startswith("emb.")
            or norm_name.startswith("emb_to_model")
            or norm_name.startswith("rosa_emb.")
            or norm_name.startswith("rosa_to_model")
        ):
            groups["emb"]["params"].append(p)
        elif norm_name.endswith(".bias"):
            groups["bias"]["params"].append(p)
        elif ("norm" in norm_name) or norm_name.endswith("scale"):
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
    rhos = []
    for layer in model.k_stack.layers:
        if hasattr(layer, "k_base_gate_logit"):
            gates.append(torch.sigmoid(layer.k_base_gate_logit).item())
        if hasattr(layer, "alpha_logit"):
            alpha_cap = float(getattr(layer, "alpha_cap", 1.0))
            alpha_vec = alpha_cap * torch.sigmoid(layer.alpha_logit)
            alphas.append((alpha_vec.min().item(), alpha_vec.mean().item(), alpha_vec.max().item()))
        if hasattr(layer, "decay_logit"):
            gamma_min = float(getattr(layer, "gamma_min", 0.0))
            gamma_max = float(getattr(layer, "gamma_max", 1.0))
            gamma = gamma_min + (gamma_max - gamma_min) * torch.sigmoid(layer.decay_logit)
            gammas.append((gamma.min().item(), gamma.mean().item(), gamma.max().item()))
        if hasattr(layer, "rho_logit"):
            rhos.append(torch.sigmoid(layer.rho_logit).item())

    parts = []
    if gates:
        parts.append(f"gate[min/mean/max]={min(gates):.3f}/{sum(gates) / len(gates):.3f}/{max(gates):.3f}")
    if alphas:
        a_min = min(x[0] for x in alphas)
        a_mean = sum(x[1] for x in alphas) / len(alphas)
        a_max = max(x[2] for x in alphas)
        parts.append(f"alpha[min/mean/max]={a_min:.3f}/{a_mean:.3f}/{a_max:.3f}")
    if gammas:
        g_min = min(x[0] for x in gammas)
        g_mean = sum(x[1] for x in gammas) / len(gammas)
        g_max = max(x[2] for x in gammas)
        parts.append(f"gamma[min/mean/max]={g_min:.3f}/{g_mean:.3f}/{g_max:.3f}")
    if rhos:
        parts.append(f"rho[min/mean/max]={min(rhos):.3f}/{sum(rhos) / len(rhos):.3f}/{max(rhos):.3f}")
    return " | ".join(parts)


@torch.no_grad()
def _collect_update_weight_stats(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_now: float,
    topk: int = 5,
) -> Dict[str, object]:
    """Estimate Adam update/weight ratios using exp_avg as a proxy (diagnostic only)."""
    ratios = []
    by_layer = {}

    def normalize_name(name: str) -> str:
        while True:
            if name.startswith("_orig_mod."):
                name = name[len("_orig_mod."):]
                continue
            if name.startswith("module."):
                name = name[len("module."):]
                continue
            break
        return name

    def layer_key(name: str) -> str:
        name = normalize_name(name)
        if name.startswith("emb."):
            return "emb"
        if name.startswith("emb_to_model"):
            return "emb"
        if name.startswith("head."):
            return "head"
        if name.startswith("head_to_emb"):
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

    if cfg.rollout_mode not in {"argmax", "sample"}:
        raise ValueError(f"Unsupported rollout_mode: {cfg.rollout_mode}")
    if cfg.rollout_horizon < 0:
        raise ValueError(f"rollout_horizon must be >= 0, got {cfg.rollout_horizon}")
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
    ckpt_targets = _resolve_checkpoint_targets(ckpt_path)
    best_rollout_ce = float("inf")
    best_useful_rollout_ce = float("inf")
    if ckpt_targets is not None:
        best_rollout_ce = _metric_from_checkpoint(ckpt_targets.best_rollout, "val_rollout_ce")
        best_useful_rollout_ce = _metric_from_checkpoint(ckpt_targets.best_useful, "val_rollout_ce")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
        last_epoch=start_step - 1,
    )
    use_grad_scaler = bool(USE_AMP and AMP_DTYPE == torch.float16)
    scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None

    LOG.info(
        "Training start | device=%s | steps=%d | window=%d | batch=%d | lr=%.2e | betas=(%.3f, %.3f) | warmup=%d | opt_mode=%s | fused_adamw=%s | grad_scaler=%s",
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
        str(use_grad_scaler).lower(),
    )
    group_parts = []
    for g in optimizer.param_groups:
        n_params = sum(p.numel() for p in g["params"])
        group_parts.append(f"{g.get('name', 'group')}[n={n_params},lr={g['lr']:.2e},wd={g['weight_decay']:.2e}]")
    LOG.info("Optimizer groups | %s", " | ".join(group_parts))
    if cfg.rollout_horizon > 0 and cfg.rollout_lambda > 0.0:
        LOG.info(
            "Aux objective | horizon=%d | rollout_lambda=%.3f | rollout_start=%d | rollout_mode=%s | semantic_lambda=%.3f | semantic_start=%d",
            cfg.rollout_horizon,
            cfg.rollout_lambda,
            cfg.rollout_start_step,
            cfg.rollout_mode,
            cfg.semantic_lambda,
            cfg.semantic_start_step,
        )
    future_horizons_str = ",".join(str(h) for h in cfg.future_summary_horizons) if cfg.future_summary_horizons else ""
    if cfg.future_summary_horizons and cfg.future_summary_lambda > 0.0:
        LOG.info(
            "Future summary | horizons=%s | future_lambda_max=%.3f | future_lambda_min=%.3f | future_start=%d",
            future_horizons_str,
            cfg.future_summary_lambda,
            cfg.future_summary_lambda_min,
            cfg.future_summary_start_step,
        )
        if cfg.future_summary_ce_target is not None:
            anchor_str = "auto" if cfg.future_summary_ce_anchor is None else f"{cfg.future_summary_ce_anchor:.4f}"
            LOG.info(
                "Future lambda schedule | ce_target=%.4f | ce_anchor=%s",
                cfg.future_summary_ce_target,
                anchor_str,
            )
    if cfg.rollout_horizon > 0 and cfg.rollout_eval_batches > 0:
        LOG.info(
            "Rollout eval | horizon=%d | batches=%d | mode=argmax",
            cfg.rollout_horizon,
            cfg.rollout_eval_batches,
        )
    if cfg.future_summary_horizons and cfg.future_summary_eval_batches > 0:
        LOG.info(
            "Future eval | horizons=%s | batches=%d",
            future_horizons_str,
            cfg.future_summary_eval_batches,
        )

    step_times: List[float] = []
    compile_warmup_pending = model is not _unwrap_model(model)
    if compile_warmup_pending:
        LOG.info("Compile warmup | first train step will include graph capture and is excluded from speed metrics.")
    train_loss_ema = None
    future_lambda_anchor = cfg.future_summary_ce_anchor
    best_ce = float("inf")
    best_val_ce = float("inf")
    stale_evals = 0
    grad_norm_hist: deque[float] = deque(maxlen=cfg.plateau_patience_evals)
    val_ce_hist: deque[float] = deque(maxlen=cfg.plateau_patience_evals)

    for step in range(start_step, cfg.steps + 1):
        t0 = time.time()
        train_total = float("nan")
        train_ce = float("nan")
        train_future = float("nan")
        train_future_lambda = float("nan")
        train_rollout = float("nan")
        train_semantic = float("nan")
        loss_phase = "ce"
        raw_grad_norm = float("nan")

        if step > 0:
            model.train()
            future_lambda_now, future_lambda_anchor = _resolve_future_lambda(cfg, train_loss_ema, future_lambda_anchor)

            with _autocast_context():
                loss, loss_stats = _compute_train_loss(
                    model,
                    train_data,
                    cfg,
                    step,
                    future_lambda=future_lambda_now,
                )

            optimizer.zero_grad(set_to_none=True)
            step_applied = False
            if not bool(torch.isfinite(loss).all().item()):
                LOG.warning("Non-finite loss at step=%d (loss=%s); skipping optimizer step.", step, float(loss.detach().item()))
            else:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    raw_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm).item())
                    if math.isfinite(raw_grad_norm):
                        scaler.step(optimizer)
                        step_applied = True
                    else:
                        LOG.warning("Non-finite gradient norm at step=%d (gnorm=%s); skipping optimizer step.", step, raw_grad_norm)
                    scaler.update()
                else:
                    loss.backward()
                    raw_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm).item())
                    if math.isfinite(raw_grad_norm):
                        optimizer.step()
                        step_applied = True
                    else:
                        LOG.warning("Non-finite gradient norm at step=%d (gnorm=%s); skipping optimizer step.", step, raw_grad_norm)

            if step_applied:
                scheduler.step()
                train_total = loss_stats.total
                train_ce = loss_stats.ce
                train_future = loss_stats.future
                train_future_lambda = loss_stats.future_lambda
                train_rollout = loss_stats.rollout
                train_semantic = loss_stats.semantic
                loss_phase = loss_stats.phase
                train_loss_ema = train_ce if train_loss_ema is None else 0.95 * train_loss_ema + 0.05 * train_ce
                if (
                    future_lambda_anchor is None
                    and cfg.future_summary_ce_target is not None
                    and train_loss_ema is not None
                    and math.isfinite(train_loss_ema)
                ):
                    future_lambda_anchor = max(float(train_loss_ema), float(cfg.future_summary_ce_target) + 1e-6)
                if cfg.diagnostics and not math.isnan(raw_grad_norm):
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
            best_val_ce = min(best_val_ce, ce)
            val_future_summary = float("nan")
            val_future_examples = 0
            if cfg.future_summary_horizons and cfg.future_summary_eval_batches > 0:
                val_future_summary, val_future_examples = eval_future_summary_deterministic(
                    model,
                    val_data,
                    cfg.window,
                    cfg.batch_size,
                    cfg.future_summary_horizons,
                    cfg.future_summary_eval_batches,
                )
            val_rollout_ce = float("nan")
            val_semantic = float("nan")
            val_rollout_examples = 0
            if cfg.rollout_horizon > 0 and cfg.rollout_eval_batches > 0:
                val_rollout_ce, val_semantic, val_rollout_examples = eval_rollout_deterministic(
                    model,
                    val_data,
                    cfg.window,
                    cfg.batch_size,
                    cfg.rollout_horizon,
                    cfg.rollout_eval_batches,
                )
            if cfg.diagnostics:
                val_ce_hist.append(ce)
                improved = ce < (best_ce - cfg.min_improve_ce)
                if improved:
                    best_ce = ce
                    stale_evals = 0
                else:
                    stale_evals += 1

            if ppl < best_ppl:
                best_ppl = ppl
                if ckpt_targets is not None:
                    save_checkpoint(
                        ckpt_targets.best_ppl,
                        model,
                        optimizer,
                        step,
                        best_ppl,
                        extra_metadata={
                            "selection_metric": "best_ppl",
                            "val_ce": ce,
                            "val_ppl": ppl,
                            "val_future_summary": val_future_summary,
                            "val_rollout_ce": val_rollout_ce,
                            "val_semantic": val_semantic,
                        },
                    )
            if ckpt_targets is not None and val_rollout_examples > 0 and val_rollout_ce < best_rollout_ce:
                best_rollout_ce = val_rollout_ce
                save_checkpoint(
                    ckpt_targets.best_rollout,
                    model,
                    optimizer,
                    step,
                    best_ppl,
                    extra_metadata={
                        "selection_metric": "best_rollout",
                        "val_ce": ce,
                        "val_ppl": ppl,
                        "val_future_summary": val_future_summary,
                        "val_rollout_ce": val_rollout_ce,
                        "val_semantic": val_semantic,
                    },
                )
            useful_ok = val_rollout_examples > 0 and ce <= (best_val_ce + cfg.rollout_useful_ce_tol)
            if ckpt_targets is not None and useful_ok and val_rollout_ce < best_useful_rollout_ce:
                best_useful_rollout_ce = val_rollout_ce
                save_checkpoint(
                    ckpt_targets.best_useful,
                    model,
                    optimizer,
                    step,
                    best_ppl,
                    extra_metadata={
                        "selection_metric": "best_useful",
                        "val_ce": ce,
                        "val_ppl": ppl,
                        "val_future_summary": val_future_summary,
                        "val_rollout_ce": val_rollout_ce,
                        "val_semantic": val_semantic,
                        "guardrail_best_val_ce": best_val_ce,
                        "guardrail_ce_tol": cfg.rollout_useful_ce_tol,
                    },
                )

            if step_times:
                avg_ms = (sum(step_times) / len(step_times)) * 1000.0
                tok_s = (cfg.batch_size * cfg.window) / max(avg_ms / 1000.0, 1e-9)
                avg_ms_str = f"{avg_ms:.1f}"
                tok_s_str = f"{tok_s:.0f}"
            else:
                avg_ms_str = "N/A"
                tok_s_str = "N/A"
            lr_now = optimizer.param_groups[0]["lr"]
            aux_enabled = (
                (bool(cfg.future_summary_horizons) and cfg.future_summary_lambda > 0.0)
                or (cfg.rollout_horizon > 0 and (cfg.rollout_lambda > 0.0 or cfg.semantic_lambda > 0.0))
            )
            if cfg.report_bpc and not aux_enabled:
                train_str = "N/A" if math.isnan(train_ce) else f"{ce_to_bpc(train_ce):.4f}"
                train_ema_str = "N/A" if train_loss_ema is None else f"{ce_to_bpc(train_loss_ema):.4f}"
                LOG.info(
                    "step=%5d | train_bpc=%s | train_bpc_ema=%s | val_bpc=%.4f | val_ppl=%.2f | best_ppl=%.2f | lr=%.2e | %s ms/step | %s tok/s",
                    step,
                    train_str,
                    train_ema_str,
                    ce_to_bpc(ce),
                    ppl,
                    best_ppl,
                    lr_now,
                    avg_ms_str,
                    tok_s_str,
                )
            elif cfg.report_bpc and aux_enabled:
                train_total_str = "N/A" if math.isnan(train_total) else f"{train_total:.4f}"
                train_bpc_str = "N/A" if math.isnan(train_ce) else f"{ce_to_bpc(train_ce):.4f}"
                train_ema_str = "N/A" if train_loss_ema is None else f"{ce_to_bpc(train_loss_ema):.4f}"
                future_str = "N/A" if math.isnan(train_future) else f"{train_future:.4f}"
                future_lambda_str = "N/A" if math.isnan(train_future_lambda) else f"{train_future_lambda:.4f}"
                rollout_str = "N/A" if math.isnan(train_rollout) else f"{train_rollout:.4f}"
                semantic_str = "N/A" if math.isnan(train_semantic) else f"{train_semantic:.4f}"
                LOG.info(
                    "step=%5d | phase=%s | train_total=%s | train_bpc=%s | train_bpc_ema=%s | train_future=%s | future_lambda=%s | train_rollout=%s | train_sem=%s | val_bpc=%.4f | val_ppl=%.2f | best_ppl=%.2f | lr=%.2e | %s ms/step | %s tok/s",
                    step,
                    loss_phase,
                    train_total_str,
                    train_bpc_str,
                    train_ema_str,
                    future_str,
                    future_lambda_str,
                    rollout_str,
                    semantic_str,
                    ce_to_bpc(ce),
                    ppl,
                    best_ppl,
                    lr_now,
                    avg_ms_str,
                    tok_s_str,
                )
            elif not aux_enabled:
                train_str = "N/A" if math.isnan(train_ce) else f"{train_ce:.4f}"
                train_ema_str = "N/A" if train_loss_ema is None else f"{train_loss_ema:.4f}"
                LOG.info(
                    "step=%5d | train_ce=%s | train_ce_ema=%s | val_ce=%.4f | val_ppl=%.2f | best_ppl=%.2f | lr=%.2e | %s ms/step | %s tok/s",
                    step,
                    train_str,
                    train_ema_str,
                    ce,
                    ppl,
                    best_ppl,
                    lr_now,
                    avg_ms_str,
                    tok_s_str,
                )
            else:
                train_total_str = "N/A" if math.isnan(train_total) else f"{train_total:.4f}"
                train_ce_str = "N/A" if math.isnan(train_ce) else f"{train_ce:.4f}"
                train_ema_str = "N/A" if train_loss_ema is None else f"{train_loss_ema:.4f}"
                future_str = "N/A" if math.isnan(train_future) else f"{train_future:.4f}"
                future_lambda_str = "N/A" if math.isnan(train_future_lambda) else f"{train_future_lambda:.4f}"
                rollout_str = "N/A" if math.isnan(train_rollout) else f"{train_rollout:.4f}"
                semantic_str = "N/A" if math.isnan(train_semantic) else f"{train_semantic:.4f}"
                LOG.info(
                    "step=%5d | phase=%s | train_total=%s | train_ce=%s | train_ce_ema=%s | train_future=%s | future_lambda=%s | train_rollout=%s | train_sem=%s | val_ce=%.4f | val_ppl=%.2f | best_ppl=%.2f | lr=%.2e | %s ms/step | %s tok/s",
                    step,
                    loss_phase,
                    train_total_str,
                    train_ce_str,
                    train_ema_str,
                    future_str,
                    future_lambda_str,
                    rollout_str,
                    semantic_str,
                    ce,
                    ppl,
                    best_ppl,
                    lr_now,
                    avg_ms_str,
                    tok_s_str,
                )
            if val_future_examples > 0:
                LOG.info(
                    "future_eval | step=%5d | val_future=%.4f | examples=%d | horizons=%s",
                    step,
                    val_future_summary,
                    val_future_examples,
                    future_horizons_str,
                )
            if val_rollout_examples > 0:
                LOG.info(
                    "rollout_eval | step=%5d | val_rollout_ce=%.4f | val_sem=%.4f | examples=%d | mode=argmax",
                    step,
                    val_rollout_ce,
                    val_semantic,
                    val_rollout_examples,
                )
                if ckpt_targets is not None:
                    LOG.info(
                        "checkpoint_metrics | best_rollout_ce=%.4f | best_useful_rollout_ce=%.4f | best_val_ce=%.4f | useful_ce_tol=%.4f",
                        best_rollout_ce,
                        best_useful_rollout_ce,
                        best_val_ce,
                        cfg.rollout_useful_ce_tol,
                    )
            if cfg.diagnostics:
                grad_stats = _collect_grad_stats(model, topk=cfg.grad_topk)
                top_grad = ", ".join(f"{n}={v:.2e}" for n, v in grad_stats["top_grad_params"]) or "none"
                k_stats = _collect_k_layer_stats(model)
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
                        5,
                        k,
                    ),
                )
                uw_by_layer = " | ".join(
                    f"{k}[{uw_stats['by_layer'][k]['min']:.1e}/{uw_stats['by_layer'][k]['mean']:.1e}/{uw_stats['by_layer'][k]['max']:.1e}]"
                    for k in layer_order
                ) or "none"
                LOG.info(
                    "diagnostics | gnorm=%.2e | gnorm_clip=%.2e | clip_hit=%s | gmax=%.2e | g_sparsity=%.2f%% | stale=%d | adam_expavg_weight[min/mean/max]=%.2e/%.2e/%.2e",
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
                LOG.info("adam_expavg_weight_top_params | %s", top_uw)
                LOG.info("adam_expavg_weight_by_layer[min/mean/max] | %s", uw_by_layer)
                if k_stats:
                    LOG.info("layer_stats | %s", k_stats)
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
