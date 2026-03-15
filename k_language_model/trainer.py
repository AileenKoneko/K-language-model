import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from .checkpoint import load_checkpoint, save_checkpoint
from .data import get_batch
from .model import KStackModel
from .runtime import DEVICE, GAMMA_FLOOR, LOG, USE_AMP, _autocast_context, _unwrap_model


def ce_to_bpc(ce: float) -> float:
    return float(ce) / math.log(2.0)


@torch.no_grad()
def eval_deterministic(model: nn.Module, data: torch.Tensor, window: int, batch_size: int) -> tuple[float, float]:
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
        elif name.startswith("emb.") or name.startswith("emb_to_model"):
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
        parts.append(f"gate[min/mean/max]={min(gates):.3f}/{sum(gates) / len(gates):.3f}/{max(gates):.3f}")
    if alphas:
        parts.append(f"alpha[min/mean/max]={min(alphas):.3f}/{sum(alphas) / len(alphas):.3f}/{max(alphas):.3f}")
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
    delta_str = ", ".join(f"L{i + 1}={v:.2e}" for i, v in enumerate(delta_mean))
    return f"eta={eta:.6f} | eta_logit={eta_logit:.6f} | loop_delta_mean={delta_str} | batches={batches}"


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

    def layer_key(name: str) -> str:
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
                loss = model(x, targets=y)

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
            if cfg.report_bpc:
                train_str = "N/A" if math.isnan(train_loss) else f"{ce_to_bpc(train_loss):.4f}"
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
            else:
                train_str = "N/A" if math.isnan(train_loss) else f"{train_loss:.4f}"
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
            if cfg.diagnostics:
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
