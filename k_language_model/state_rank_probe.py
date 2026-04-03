from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

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


_STATE_KIND_CHOICES = ("q_raw", "q", "q_alpha", "k_raw", "k")


@dataclass
class _RankStateAccumulator:
    rank: int
    count: int = 0
    sum_vec: torch.Tensor = field(init=False)
    sum_outer: torch.Tensor = field(init=False)
    min_vec: torch.Tensor = field(init=False)
    max_vec: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        r = int(self.rank)
        self.sum_vec = torch.zeros(r, dtype=torch.float64)
        self.sum_outer = torch.zeros(r, r, dtype=torch.float64)
        self.min_vec = torch.full((r,), float("inf"), dtype=torch.float64)
        self.max_vec = torch.full((r,), float("-inf"), dtype=torch.float64)

    def update(self, states: torch.Tensor) -> None:
        if states.ndim != 3:
            raise ValueError(f"Expected [B, T, R] states, got {tuple(states.shape)}.")
        if states.size(-1) != self.rank:
            raise ValueError(f"Expected rank={self.rank}, got {states.size(-1)}.")
        flat = states.detach().to(device="cpu", dtype=torch.float64).reshape(-1, self.rank)
        n = int(flat.size(0))
        if n <= 0:
            return
        self.count += n
        self.sum_vec += flat.sum(dim=0)
        self.sum_outer += flat.transpose(0, 1) @ flat
        self.min_vec = torch.minimum(self.min_vec, flat.min(dim=0).values)
        self.max_vec = torch.maximum(self.max_vec, flat.max(dim=0).values)

    def finalize(self) -> dict[str, object]:
        r = int(self.rank)
        if self.count <= 0:
            nan_vec = [float("nan")] * r
            nan_mat = [[float("nan")] * r for _ in range(r)]
            return {
                "count": 0,
                "rank": r,
                "mean": nan_vec,
                "std": nan_vec,
                "min": nan_vec,
                "max": nan_vec,
                "correlation": nan_mat,
                "cosine_similarity": nan_mat,
                "cov_eigenvalues": nan_vec,
                "singular_values": nan_vec,
                "effective_rank": float("nan"),
                "corr_offdiag_mean": float("nan"),
                "corr_offdiag_abs_mean": float("nan"),
                "cosine_offdiag_mean": float("nan"),
                "cosine_offdiag_abs_mean": float("nan"),
            }

        count_f = float(self.count)
        mean = self.sum_vec / count_f
        second_moment = self.sum_outer / count_f
        cov = second_moment - torch.outer(mean, mean)
        cov = 0.5 * (cov + cov.transpose(0, 1))
        var = torch.diagonal(cov).clamp(min=0.0)
        std = torch.sqrt(var)

        denom_corr = torch.outer(std, std)
        corr = torch.full((r, r), float("nan"), dtype=torch.float64)
        corr_valid = denom_corr > 1e-16
        corr[corr_valid] = cov[corr_valid] / denom_corr[corr_valid]
        diag_ok = std > 1e-8
        diag_idx = torch.arange(r)
        corr[diag_idx[diag_ok], diag_idx[diag_ok]] = 1.0

        raw_norm = torch.diagonal(self.sum_outer).clamp(min=0.0)
        denom_cos = torch.sqrt(torch.outer(raw_norm, raw_norm))
        cosine = torch.full((r, r), float("nan"), dtype=torch.float64)
        cos_valid = denom_cos > 1e-16
        cosine[cos_valid] = self.sum_outer[cos_valid] / denom_cos[cos_valid]
        diag_nonzero = raw_norm > 1e-16
        cosine[diag_idx[diag_nonzero], diag_idx[diag_nonzero]] = 1.0

        eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
        eigvals = torch.sort(eigvals, descending=True).values
        singular_values = torch.sqrt(eigvals * count_f)

        eig_sum = float(eigvals.sum().item())
        if eig_sum <= 1e-16:
            effective_rank = 0.0
        else:
            p = eigvals / eig_sum
            nz = p > 0
            entropy = -torch.sum(p[nz] * torch.log(p[nz]))
            effective_rank = float(torch.exp(entropy).item())

        offdiag_mask = ~torch.eye(r, dtype=torch.bool)

        def _offdiag_mean(mat: torch.Tensor) -> float:
            vals = mat[offdiag_mask]
            vals = vals[torch.isfinite(vals)]
            if vals.numel() <= 0:
                return float("nan")
            return float(vals.mean().item())

        def _offdiag_abs_mean(mat: torch.Tensor) -> float:
            vals = mat[offdiag_mask]
            vals = vals[torch.isfinite(vals)]
            if vals.numel() <= 0:
                return float("nan")
            return float(vals.abs().mean().item())

        return {
            "count": int(self.count),
            "rank": r,
            "mean": [float(v) for v in mean.tolist()],
            "std": [float(v) for v in std.tolist()],
            "min": [float(v) for v in self.min_vec.tolist()],
            "max": [float(v) for v in self.max_vec.tolist()],
            "correlation": [[float(v) for v in row] for row in corr.tolist()],
            "cosine_similarity": [[float(v) for v in row] for row in cosine.tolist()],
            "cov_eigenvalues": [float(v) for v in eigvals.tolist()],
            "singular_values": [float(v) for v in singular_values.tolist()],
            "effective_rank": float(effective_rank),
            "corr_offdiag_mean": _offdiag_mean(corr),
            "corr_offdiag_abs_mean": _offdiag_abs_mean(corr),
            "cosine_offdiag_mean": _offdiag_mean(cosine),
            "cosine_offdiag_abs_mean": _offdiag_abs_mean(cosine),
        }


def _resolve_probe_device(mode: str) -> torch.device:
    choice = str(mode).strip().lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unsupported probe_device: {mode}")


def _build_parser(description: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description or "Probe K2 per-rank state geometry (corr/cosine/effective-rank).")
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
        "--state-kind",
        type=str,
        choices=list(_STATE_KIND_CHOICES),
        default="q_alpha",
        help="K2 per-rank state variant to probe.",
    )
    p.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Optional cap on number of deterministic eval batches (0 = full validation split).",
    )
    p.add_argument(
        "--probe-device",
        type=str,
        choices=["cpu", "auto"],
        default="cpu",
        help="Device for probing. cpu is slower but stable on MPS setups.",
    )
    p.add_argument("--json-out", type=str, default=None, help="Optional path to save full probe JSON.")
    p.add_argument(
        "--rank-stats-csv",
        type=str,
        default=None,
        help="Optional CSV path for per-layer, per-rank summary stats.",
    )
    p.add_argument(
        "--pairwise-csv",
        type=str,
        default=None,
        help="Optional CSV path for correlation/cosine pairwise matrices.",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def _extract_rank_states(layer: K2Layer, h: torch.Tensor, *, state_kind: str, k_stack: torch.nn.Module | None = None) -> torch.Tensor:
    if hasattr(layer, "u") and hasattr(layer, "v"):
        h_norm = layer.norm1(h)
        qk = h_norm @ torch.cat((layer.u, layer.v), dim=1)
        q_raw, k_raw = qk.split(layer.u.size(1), dim=-1)
    else:
        if (
            k_stack is None
            or not hasattr(k_stack, "decay_norm")
            or not hasattr(k_stack, "decay_u")
            or not hasattr(k_stack, "decay_v")
        ):
            raise RuntimeError("Shared decay parameters are unavailable for rank-state extraction.")
        h_norm = k_stack.decay_norm(h)
        qk = h_norm @ torch.cat((k_stack.decay_u, k_stack.decay_v), dim=1)
        q_raw, k_raw = qk.split(k_stack.decay_u.size(1), dim=-1)
    if state_kind == "q_raw":
        return q_raw
    if state_kind == "k_raw":
        return k_raw
    q = F.normalize(q_raw, dim=-1, eps=1e-8)
    k = F.normalize(k_raw, dim=-1, eps=1e-8)
    if state_kind == "q":
        return q
    if state_kind == "k":
        return k
    if state_kind == "q_alpha":
        alpha = (layer.alpha_cap * torch.sigmoid(layer.alpha_logit)).to(dtype=q.dtype, device=q.device)
        return q * alpha.view(1, 1, -1)
    raise ValueError(f"Unsupported state_kind: {state_kind}")


def _iter_token_windows(data: torch.Tensor, *, window: int, batch_size: int, max_eval_batches: int):
    n_tokens = len(data) - 1
    usable_tokens = (n_tokens // window) * window
    if usable_tokens <= 0:
        return
    x_all = data[:usable_tokens].view(-1, window)
    batch_idx = 0
    for start in range(0, x_all.size(0), batch_size):
        if max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break
        yield x_all[start: start + batch_size]
        batch_idx += 1


@torch.no_grad()
def _run_rank_state_probe(
    model: torch.nn.Module,
    val_data: torch.Tensor,
    *,
    window: int,
    batch_size: int,
    max_eval_batches: int,
    state_kind: str,
    probe_device: torch.device,
) -> dict[str, object]:
    core = _unwrap_model(model)
    if not hasattr(core, "k_stack") or not hasattr(core.k_stack, "layers"):
        raise RuntimeError("Model does not expose k_stack layers required for rank-state probe.")

    if next(core.parameters(), None) is not None and next(core.parameters()).device != probe_device:
        core = core.to(probe_device)
    core.eval()

    if val_data.device != probe_device:
        val_data = val_data.to(probe_device, non_blocking=True)

    layers: list[tuple[int, int, K2Layer]] = []
    for module_idx, layer in enumerate(core.k_stack.layers):
        if isinstance(layer, K2Layer):
            layers.append((len(layers), int(module_idx), layer))
    if not layers:
        raise RuntimeError("No K2 layers found in model.")

    acc_map: Dict[int, _RankStateAccumulator] = {
        k2_idx: _RankStateAccumulator(rank=int(layer.alpha_logit.numel()))
        for k2_idx, _, layer in layers
    }

    hooks = []
    for k2_idx, _, layer in layers:
        def _pre_hook(mod: K2Layer, args, _k2_idx: int = k2_idx):
            if not args:
                return
            h = args[0]
            states = _extract_rank_states(mod, h, state_kind=state_kind, k_stack=core.k_stack)
            acc_map[_k2_idx].update(states)

        hooks.append(layer.register_forward_pre_hook(_pre_hook))

    try:
        token_windows = 0
        for x in _iter_token_windows(val_data, window=window, batch_size=batch_size, max_eval_batches=max_eval_batches):
            token_windows += int(x.numel())
            _ = core.hidden_states(x.to(probe_device, non_blocking=True))
    finally:
        for hook in hooks:
            hook.remove()

    layer_payloads: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    for k2_idx, module_idx, _ in layers:
        stats = acc_map[k2_idx].finalize()
        layer_payload = {
            "k2_index": int(k2_idx),
            "module_index": int(module_idx),
            **stats,
        }
        layer_payloads.append(layer_payload)

        means = list(stats["mean"])
        stds = list(stats["std"])
        mins = list(stats["min"])
        maxs = list(stats["max"])
        for rank_idx in range(int(stats["rank"])):
            rank_rows.append(
                {
                    "k2_index": int(k2_idx),
                    "module_index": int(module_idx),
                    "rank": int(rank_idx),
                    "state_kind": state_kind,
                    "count": int(stats["count"]),
                    "mean": float(means[rank_idx]),
                    "std": float(stds[rank_idx]),
                    "min": float(mins[rank_idx]),
                    "max": float(maxs[rank_idx]),
                }
            )

        corr = list(stats["correlation"])
        cos = list(stats["cosine_similarity"])
        for i in range(int(stats["rank"])):
            for j in range(int(stats["rank"])):
                pairwise_rows.append(
                    {
                        "k2_index": int(k2_idx),
                        "module_index": int(module_idx),
                        "metric": "correlation",
                        "rank_i": int(i),
                        "rank_j": int(j),
                        "value": float(corr[i][j]),
                    }
                )
                pairwise_rows.append(
                    {
                        "k2_index": int(k2_idx),
                        "module_index": int(module_idx),
                        "metric": "cosine_similarity",
                        "rank_i": int(i),
                        "rank_j": int(j),
                        "value": float(cos[i][j]),
                    }
                )

    return {
        "token_windows_processed": int(token_windows),
        "layers": layer_payloads,
        "rank_rows": rank_rows,
        "pairwise_rows": pairwise_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    probe_device = _resolve_probe_device(args.probe_device)
    LOG.info(
        "Runtime | train_device=%s | probe_device=%s | seed=%d | deterministic=%s",
        DEVICE,
        str(probe_device),
        args.seed,
        str(args.deterministic).lower(),
    )
    LOG.info("Run config | hash=%s", _run_config_hash(args))
    LOG.info("Command | %s", _command_string())
    log_runtime_metadata()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if args.max_eval_batches < 0:
        raise ValueError(f"--max-eval-batches must be >= 0, got {args.max_eval_batches}.")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}.")

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
    ).to(probe_device)
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

    LOG.info(
        "Rank probe start | state_kind=%s | window=%d | batch=%d | max_eval_batches=%d",
        args.state_kind,
        args.window,
        args.batch_size,
        args.max_eval_batches,
    )
    probe_out = _run_rank_state_probe(
        model=model,
        val_data=val_data,
        window=args.window,
        batch_size=args.batch_size,
        max_eval_batches=args.max_eval_batches,
        state_kind=args.state_kind,
        probe_device=probe_device,
    )
    LOG.info("Rank probe done | token_windows=%d", probe_out["token_windows_processed"])

    for layer in probe_out["layers"]:
        LOG.info(
            "Rank layer | k2=%d | module_idx=%d | samples=%d | rank=%d | eff_rank=%.3f | corr_offdiag_abs_mean=%.4f | cosine_offdiag_abs_mean=%.4f",
            layer["k2_index"],
            layer["module_index"],
            layer["count"],
            layer["rank"],
            layer["effective_rank"],
            layer["corr_offdiag_abs_mean"],
            layer["cosine_offdiag_abs_mean"],
        )
        sv = list(layer["singular_values"])
        if sv:
            LOG.info(
                "Rank layer svd | k2=%d | singular_values=%s",
                layer["k2_index"],
                ",".join(f"{float(v):.6f}" for v in sv),
            )

    payload = {
        "checkpoint": str(ckpt_path),
        "step": loaded_step,
        "best_ppl": loaded_best_ppl,
        "tokenizer": tokenizer.describe(),
        "window": int(args.window),
        "batch_size": int(args.batch_size),
        "max_eval_batches": int(args.max_eval_batches),
        "state_kind": str(args.state_kind),
        "probe_device": str(probe_device),
        "token_windows_processed": int(probe_out["token_windows_processed"]),
        "layers": probe_out["layers"],
    }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        LOG.info("Rank probe JSON written | path=%s", out_path)

    if args.rank_stats_csv:
        out_path = Path(args.rank_stats_csv)
        _write_csv(
            out_path,
            probe_out["rank_rows"],
            fieldnames=["k2_index", "module_index", "rank", "state_kind", "count", "mean", "std", "min", "max"],
        )
        LOG.info("Rank probe CSV written | type=rank_stats | path=%s", out_path)

    if args.pairwise_csv:
        out_path = Path(args.pairwise_csv)
        _write_csv(
            out_path,
            probe_out["pairwise_rows"],
            fieldnames=["k2_index", "module_index", "metric", "rank_i", "rank_j", "value"],
        )
        LOG.info("Rank probe CSV written | type=pairwise | path=%s", out_path)


if __name__ == "__main__":
    main()
