from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import load_model_checkpoint
from .cli_args import add_dataset_args, add_dynamics_args, add_model_args, add_repro_runtime_args
from .configs import DatasetConfig
from .data import load_dataset_bundle
from .kstack import K2Layer
from .model import resolve_adaptive_cutoffs
from .model_factory import build_model, model_config_from_args, parse_adaptive_cutoffs
from .runtime import (
    DEVICE,
    LOG,
    _command_string,
    _run_config_hash,
    _unwrap_model,
    configure_reproducibility,
    log_runtime_metadata,
    setup_logging,
)


@dataclass
class ProbeStats:
    overall_total: int = 0
    overall_correct: int = 0
    assistant_total: int = 0
    assistant_correct: int = 0
    repeat_total: int = 0
    repeat_correct: int = 0
    nonrepeat_total: int = 0
    nonrepeat_correct: int = 0
    copy_total: int = 0
    copy_correct: int = 0
    novel_total: int = 0
    novel_correct: int = 0

    def ratio(self, num: int, den: int) -> float:
        if den <= 0:
            return float("nan")
        return float(num) / float(den)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "overall_top1_acc": self.ratio(self.overall_correct, self.overall_total),
            "assistant_top1_acc": self.ratio(self.assistant_correct, self.assistant_total),
            "repeat_target_rate": self.ratio(self.repeat_total, self.assistant_total if self.assistant_total > 0 else self.overall_total),
            "repeat_top1_acc": self.ratio(self.repeat_correct, self.repeat_total),
            "nonrepeat_top1_acc": self.ratio(self.nonrepeat_correct, self.nonrepeat_total),
            "copy_target_rate": self.ratio(self.copy_total, self.assistant_total if self.assistant_total > 0 else self.overall_total),
            "copy_top1_acc": self.ratio(self.copy_correct, self.copy_total),
            "novel_top1_acc": self.ratio(self.novel_correct, self.novel_total),
            "overall_total_tokens": self.overall_total,
            "assistant_total_tokens": self.assistant_total,
            "copy_total_tokens": self.copy_total,
            "novel_total_tokens": self.novel_total,
        }


def _build_parser(description: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description or "Probe a K-Stack checkpoint with structure-oriented metrics.")
    add_dataset_args(
        p,
        tokenizer_help="Tokenizer used to rebuild the checkpoint vocabulary.",
        sp_model_help="Existing SentencePiece model path used by the checkpoint.",
    )
    p.add_argument("--ckpt", "--checkpoint", dest="ckpt", type=str, required=True, help="Checkpoint path to load.")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size used for deterministic probing.")
    add_model_args(
        p,
        include_dropouts=False,
        adaptive_cutoffs_help="Comma-separated adaptive softmax cutoffs used by the checkpoint.",
    )
    add_dynamics_args(
        p,
        decay_help="Gamma-decay backend used by the checkpoint architecture.",
    )
    add_repro_runtime_args(
        p,
        include_compile=False,
        include_run_manifest=False,
        strict_repro_help="Force deterministic probing (deterministic ops and TF32 off).",
    )
    p.add_argument(
        "--copy-window",
        type=int,
        default=64,
        help="Context window used to classify targets as copy-vs-novel.",
    )
    p.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Optional cap on number of deterministic eval batches (0 = full validation split).",
    )
    p.add_argument(
        "--ce-stride",
        type=int,
        default=64,
        help="Sliding-window CE stride in tokens (1 = strict sliding, window = non-overlap).",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to save structured probe output as JSON.",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def _iter_eval_batches(
    data: torch.Tensor,
    loss_mask: torch.Tensor | None,
    window: int,
    batch_size: int,
    max_eval_batches: int,
):
    n_tokens = len(data) - 1
    usable_tokens = (n_tokens // window) * window
    if usable_tokens <= 0:
        return
    x_all = data[:usable_tokens].view(-1, window)
    y_all = data[1: usable_tokens + 1].view(-1, window)
    keep_all = None if loss_mask is None else loss_mask[1: usable_tokens + 1].view(-1, window)

    batch_idx = 0
    for start in range(0, x_all.size(0), batch_size):
        if max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break
        x = x_all[start: start + batch_size]
        y = y_all[start: start + batch_size]
        keep = None if keep_all is None else keep_all[start: start + batch_size]
        yield x, y, keep
        batch_idx += 1


def _update_copy_and_repeat_stats(
    stats: ProbeStats,
    x: torch.Tensor,
    y: torch.Tensor,
    pred: torch.Tensor,
    keep: torch.Tensor | None,
    copy_window: int,
) -> None:
    if keep is None:
        target_mask = torch.ones_like(y, dtype=torch.bool)
    else:
        target_mask = keep.bool()
    compare = pred.eq(y)

    repeat_mask = target_mask & y.eq(x)
    nonrepeat_mask = target_mask & ~y.eq(x)
    stats.repeat_total += int(repeat_mask.sum().item())
    stats.repeat_correct += int((compare & repeat_mask).sum().item())
    stats.nonrepeat_total += int(nonrepeat_mask.sum().item())
    stats.nonrepeat_correct += int((compare & nonrepeat_mask).sum().item())

    x_cpu = x.detach().cpu()
    y_cpu = y.detach().cpu()
    pred_cpu = pred.detach().cpu()
    mask_cpu = target_mask.detach().cpu()
    batch, window = y_cpu.shape
    for b in range(batch):
        for t in range(window):
            if not bool(mask_cpu[b, t].item()):
                continue
            left = max(0, t - copy_window + 1)
            context = x_cpu[b, left: t + 1]
            target_token = int(y_cpu[b, t].item())
            correct = bool(pred_cpu[b, t].item() == target_token)
            is_copy = bool((context == target_token).any().item()) if context.numel() > 0 else False
            if is_copy:
                stats.copy_total += 1
                stats.copy_correct += 1 if correct else 0
            else:
                stats.novel_total += 1
                stats.novel_correct += 1 if correct else 0


def _collect_layer_probe_stats(model: torch.nn.Module) -> list[dict[str, float | int]]:
    core = _unwrap_model(model)
    if not hasattr(core, "k_stack") or not hasattr(core.k_stack, "layers"):
        return []

    layer_rows: list[dict[str, float | int]] = []
    stack_gamma = core.k_stack.decay_gamma() if hasattr(core.k_stack, "decay_gamma") else None
    stack_kappa = None
    if hasattr(core.k_stack, "kappa_proj"):
        bias = getattr(core.k_stack.kappa_proj, "bias", None)
        if isinstance(bias, torch.Tensor):
            stack_kappa = torch.sigmoid(bias.detach().float())
    stack_epsilon = None
    if hasattr(core.k_stack, "epsilon_proj"):
        bias = getattr(core.k_stack.epsilon_proj, "bias", None)
        if isinstance(bias, torch.Tensor):
            stack_epsilon = torch.tanh(bias.detach().float())
    k2_idx = 0
    for layer in core.k_stack.layers:
        if not isinstance(layer, K2Layer):
            continue
        if stack_gamma is not None:
            gamma_vec = stack_gamma
        elif hasattr(layer, "decay_gamma") and hasattr(layer, "decay_logit"):
            gamma_vec = layer.decay_gamma()
        elif hasattr(layer, "decay_logit"):
            gamma_vec = torch.sigmoid(layer.decay_logit).clamp(min=layer.gamma_min, max=layer.gamma_max)
        else:
            gamma_vec = torch.zeros_like(layer.alpha_logit)
        if stack_kappa is not None:
            kappa_vec = stack_kappa
        elif hasattr(layer, "kappa_logit"):
            kappa_vec = torch.sigmoid(layer.kappa_logit.detach().float()).view(1)
        else:
            kappa_vec = torch.zeros(1, dtype=layer.alpha_logit.dtype, device=layer.alpha_logit.device)
        if stack_epsilon is not None:
            epsilon_vec = stack_epsilon
        else:
            epsilon_vec = torch.zeros(1, dtype=layer.alpha_logit.dtype, device=layer.alpha_logit.device)
        row = {
            "k2_index": int(k2_idx),
            "gate": float(torch.sigmoid(layer.k_base_gate_logit).item()),
            "alpha_mean": float((layer.alpha_cap * torch.sigmoid(layer.alpha_logit)).mean().item()),
            "alpha_min": float((layer.alpha_cap * torch.sigmoid(layer.alpha_logit)).min().item()),
            "alpha_max": float((layer.alpha_cap * torch.sigmoid(layer.alpha_logit)).max().item()),
            "gamma_mean": float(gamma_vec.mean().item()),
            "gamma_min": float(gamma_vec.min().item()),
            "gamma_max": float(gamma_vec.max().item()),
            "kappa_mean": float(kappa_vec.mean().item()),
            "kappa_min": float(kappa_vec.min().item()),
            "kappa_max": float(kappa_vec.max().item()),
            "epsilon_mean": float(epsilon_vec.mean().item()),
            "epsilon_min": float(epsilon_vec.min().item()),
            "epsilon_max": float(epsilon_vec.max().item()),
            "rho": float(torch.sigmoid(layer.rho_logit).item()),
        }
        kernel = core.k_stack.shared_k_base_kernel if core.k_stack.share_k_base else layer.k_base_kernel
        if isinstance(kernel, torch.Tensor):
            row["k_base_l2"] = float(kernel.norm().item())
            row["k_base_l1"] = float(kernel.abs().sum().item())
        layer_rows.append(row)
        k2_idx += 1
    return layer_rows


@torch.no_grad()
def compute_sliding_window_ce(
    model: torch.nn.Module,
    data: torch.Tensor,
    loss_mask: torch.Tensor | None,
    *,
    window: int,
    stride: int,
    ignore_index: int = -100,
) -> tuple[float, float, int]:
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")
    if stride > window:
        raise ValueError(f"stride must be <= window ({window}), got {stride}")

    # MPS can be unstable for many sequential CE forwards in probe mode; run CE pass on CPU there.
    device = torch.device("cpu") if DEVICE == "mps" else torch.device(DEVICE)
    first_param = next(model.parameters(), None)
    if first_param is not None and first_param.device != device:
        model = model.to(device)
    model.eval()
    if data.device != device:
        data = data.to(device, non_blocking=True)
    if loss_mask is not None:
        if len(loss_mask) != len(data):
            raise ValueError(f"loss_mask length ({len(loss_mask)}) must match data length ({len(data)}).")
        if loss_mask.device != device:
            loss_mask = loss_mask.to(device, non_blocking=True)

    n_tokens = len(data) - 1
    if n_tokens <= 0:
        return 0.0, 1.0, 0

    total_loss = 0.0
    total_count = 0
    next_target_pos = 1  # global target index into `data` space

    for start in range(0, n_tokens, stride):
        end = min(start + window, n_tokens)
        if end <= start:
            break

        x = data[start:end].unsqueeze(0)
        y = data[start + 1: end + 1].unsqueeze(0)
        if y.numel() == 0:
            if end == n_tokens:
                break
            continue

        global_pos = torch.arange(start + 1, end + 1, device=device)
        include = global_pos.ge(next_target_pos)
        if loss_mask is not None:
            include = include & loss_mask[start + 1: end + 1].bool()
        if not bool(include.any().item()):
            next_target_pos = max(next_target_pos, end + 1)
            if end == n_tokens:
                break
            continue

        y_masked = y.masked_fill(~include.unsqueeze(0), int(ignore_index))
        loss = model(x, targets=y_masked, reduction="sum")
        total_loss += float(loss.item())
        total_count += int(include.sum().item())
        next_target_pos = max(next_target_pos, end + 1)
        if end == n_tokens:
            break

    ce = total_loss / max(total_count, 1)
    ppl = math.exp(min(ce, 20.0))
    return ce, ppl, total_count


@torch.no_grad()
def _run_probe(
    model: torch.nn.Module,
    val_data: torch.Tensor,
    val_loss_mask: torch.Tensor | None,
    *,
    window: int,
    batch_size: int,
    copy_window: int,
    max_eval_batches: int,
) -> ProbeStats:
    stats = ProbeStats()
    model.eval()

    device = torch.device(DEVICE)
    if val_data.device != device:
        val_data = val_data.to(device, non_blocking=True)
    if val_loss_mask is not None and val_loss_mask.device != device:
        val_loss_mask = val_loss_mask.to(device, non_blocking=True)

    for x, y, keep in _iter_eval_batches(val_data, val_loss_mask, window, batch_size, max_eval_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        keep_device = None if keep is None else keep.to(device, non_blocking=True)

        logits = model(x)
        pred = logits.argmax(dim=-1)

        stats.overall_total += y.numel()
        stats.overall_correct += int(pred.eq(y).sum().item())
        if keep_device is not None:
            keep_bool = keep_device.bool()
            stats.assistant_total += int(keep_bool.sum().item())
            stats.assistant_correct += int((pred.eq(y) & keep_bool).sum().item())

        _update_copy_and_repeat_stats(
            stats=stats,
            x=x,
            y=y,
            pred=pred,
            keep=keep_device,
            copy_window=copy_window,
        )

    return stats


def main() -> None:
    args = _build_parser().parse_args()
    setup_logging(verbose=args.verbose)

    if args.strict_repro:
        args.deterministic = True
        args.deterministic_warn_only = False
        args.no_tf32 = True

    configure_reproducibility(
        seed=args.seed,
        deterministic=args.deterministic,
        deterministic_warn_only=args.deterministic_warn_only,
        allow_tf32=not args.no_tf32,
    )
    LOG.info("Runtime | device=%s | seed=%d | deterministic=%s", DEVICE, args.seed, str(args.deterministic).lower())
    LOG.info("Run config | hash=%s", _run_config_hash(args))
    LOG.info("Command | %s", _command_string())
    log_runtime_metadata()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if args.copy_window <= 0:
        raise ValueError(f"--copy-window must be > 0, got {args.copy_window}.")
    if args.max_eval_batches < 0:
        raise ValueError(f"--max-eval-batches must be >= 0, got {args.max_eval_batches}.")
    if args.ce_stride <= 0:
        raise ValueError(f"--ce-stride must be > 0, got {args.ce_stride}.")
    if args.ce_stride > args.window:
        raise ValueError(f"--ce-stride must be <= window ({args.window}), got {args.ce_stride}.")

    remap_by_frequency = args.head_mode == "adaptive"
    dataset_bundle = load_dataset_bundle(
        DatasetConfig(
            dataset=args.dataset,
            val_frac=args.val_frac,
            data_path=args.data_path,
            val_path=args.val_path,
            tokenizer_type=args.tokenizer,
            sp_model=args.sp_model,
            sp_vocab_size=args.sp_vocab_size,
            sp_model_type=args.sp_model_type,
            sp_character_coverage=args.sp_character_coverage,
            sp_split_digits=args.sp_split_digits,
            sp_byte_fallback=args.sp_byte_fallback,
            allow_training_tokenizer=False,
            remap_by_frequency=remap_by_frequency,
        )
    )
    tokenizer = dataset_bundle.tokenizer
    val_data = dataset_bundle.val_data
    val_loss_mask = dataset_bundle.val_loss_mask
    vocab_size = tokenizer.vocab_size

    adaptive_cutoffs = None
    if args.head_mode == "adaptive":
        adaptive_cutoffs = resolve_adaptive_cutoffs(vocab_size, parse_adaptive_cutoffs(args.adaptive_cutoffs))
        LOG.info("Adaptive head | cutoffs=%s | div_value=%.2f", adaptive_cutoffs, args.adaptive_div_value)

    model = build_model(
        model_config_from_args(
            args,
            vocab_size=vocab_size,
            adaptive_cutoffs=adaptive_cutoffs,
            emb_dropout=0.0,
            mlp_dropout=0.0,
            residual_dropout=0.0,
        )
    ).to(DEVICE)
    loaded_step, loaded_best_ppl = load_model_checkpoint(ckpt_path, model)

    core_model = _unwrap_model(model)
    params = core_model.count_params() if hasattr(core_model, "count_params") else {}
    if params:
        LOG.info(
            "Model params | total=%s | embedding=%s | k_stack=%s | head=%s | other=%s",
            f"{params['total']:,}",
            f"{params['embedding']:,}",
            f"{params['k_stack']:,}",
            f"{params['head']:,}",
            f"{params['other']:,}",
        )

    LOG.info("Probe pass start | mode=structure_top1")
    probe_stats = _run_probe(
        model=model,
        val_data=val_data,
        val_loss_mask=val_loss_mask,
        window=args.window,
        batch_size=args.batch_size,
        copy_window=args.copy_window,
        max_eval_batches=args.max_eval_batches,
    )
    LOG.info("Probe pass done | mode=structure_top1")
    metric_row = probe_stats.as_dict()
    LOG.info("Probe pass start | mode=sliding_ce | stride=%d", args.ce_stride)
    ce_value, ppl_value, ce_count = compute_sliding_window_ce(
        model=model,
        data=val_data,
        loss_mask=val_loss_mask,
        window=args.window,
        stride=args.ce_stride,
    )
    LOG.info("Probe pass done | mode=sliding_ce")
    metric_row["sliding_ce"] = ce_value
    metric_row["sliding_ppl"] = ppl_value
    metric_row["sliding_eval_tokens"] = ce_count
    metric_row["sliding_stride"] = int(args.ce_stride)
    layer_rows = _collect_layer_probe_stats(model)

    LOG.info(
        "Probe metrics | sliding_ce=%.4f | sliding_ppl=%.2f | overall_top1=%.4f | assistant_top1=%.4f | repeat_acc=%.4f | nonrepeat_acc=%.4f | copy_acc=%.4f | novel_acc=%.4f | copy_rate=%.4f",
        metric_row["sliding_ce"],
        metric_row["sliding_ppl"],
        metric_row["overall_top1_acc"],
        metric_row["assistant_top1_acc"],
        metric_row["repeat_top1_acc"],
        metric_row["nonrepeat_top1_acc"],
        metric_row["copy_top1_acc"],
        metric_row["novel_top1_acc"],
        metric_row["copy_target_rate"],
    )
    for row in layer_rows:
        LOG.info(
            "Layer probe | k2=%d | gate=%.3f | alpha[min/mean/max]=%.3f/%.3f/%.3f | gamma[min/mean/max]=%.3f/%.3f/%.3f | kappa[min/mean/max]=%.3f/%.3f/%.3f | epsilon[min/mean/max]=%.3f/%.3f/%.3f | rho=%.3f | k_base[l1/l2]=%.3e/%.3e",
            row["k2_index"],
            row["gate"],
            row["alpha_min"],
            row["alpha_mean"],
            row["alpha_max"],
            row["gamma_min"],
            row["gamma_mean"],
            row["gamma_max"],
            row.get("kappa_min", float("nan")),
            row.get("kappa_mean", float("nan")),
            row.get("kappa_max", float("nan")),
            row.get("epsilon_min", float("nan")),
            row.get("epsilon_mean", float("nan")),
            row.get("epsilon_max", float("nan")),
            row["rho"],
            row.get("k_base_l1", float("nan")),
            row.get("k_base_l2", float("nan")),
        )

    payload = {
        "checkpoint": str(ckpt_path),
        "step": loaded_step,
        "best_ppl": loaded_best_ppl,
        "tokenizer": tokenizer.describe(),
        "window": int(args.window),
        "batch_size": int(args.batch_size),
        "copy_window": int(args.copy_window),
        "ce_stride": int(args.ce_stride),
        "max_eval_batches": int(args.max_eval_batches),
        "metrics": metric_row,
        "layers": layer_rows,
        "assistant_supervised_tokens": int(val_loss_mask.sum().item()) if val_loss_mask is not None else None,
    }
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        LOG.info("Probe report written | path=%s", out_path)


if __name__ == "__main__":
    main()
