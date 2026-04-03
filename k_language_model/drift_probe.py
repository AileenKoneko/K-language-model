from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import load_model_checkpoint
from .cli_args import add_dataset_args, add_dynamics_args, add_model_args, add_repro_runtime_args, add_sampling_args
from .configs import DatasetConfig
from .data import load_dataset_bundle
from .model import resolve_adaptive_cutoffs
from .model_factory import build_model, model_config_from_args, parse_adaptive_cutoffs
from .runtime import (
    LOG,
    _command_string,
    _run_config_hash,
    configure_reproducibility,
    log_runtime_metadata,
    setup_logging,
)


def _build_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description or "Probe held-out continuation CE/PPL against sampled drifted continuations."
    )
    add_dataset_args(
        parser,
        tokenizer_help="Tokenizer used to rebuild the checkpoint vocabulary.",
        sp_model_help="Existing SentencePiece model path used by the checkpoint.",
    )
    parser.add_argument("--ckpt", "--checkpoint", dest="ckpt", type=str, required=True, help="Checkpoint path to load.")
    parser.add_argument("--batch-size", type=int, default=8, help="Probe batch size for generation and CE scoring.")
    add_model_args(
        parser,
        include_dropouts=False,
        adaptive_cutoffs_help="Comma-separated adaptive softmax cutoffs used by the checkpoint.",
    )
    add_dynamics_args(
        parser,
        decay_help="Gamma-decay backend used by the checkpoint architecture.",
    )
    add_repro_runtime_args(
        parser,
        include_compile=False,
        include_run_manifest=False,
        strict_repro_help="Force deterministic probing (deterministic ops and TF32 off).",
    )
    add_sampling_args(parser, include_sample_flag=False, include_skip_flags=False)
    parser.add_argument(
        "--probe-mode",
        type=str,
        choices=["heldout", "prompt"],
        default="heldout",
        help="heldout: evenly spaced prompts from validation split. prompt: locate --prompt in validation split.",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=256,
        help="Prompt length for heldout mode.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=32,
        help="Number of held-out prompt/continuation pairs for heldout mode.",
    )
    parser.add_argument(
        "--probe-device",
        type=str,
        choices=["cpu", "auto"],
        default="cpu",
        help="Device for probing. cpu is slower but stable on MPS setups.",
    )
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save a JSON report.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _resolve_probe_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _top_p_top_k(scores: torch.Tensor, *, top_k: int | None, top_p: float | None) -> torch.Tensor:
    filtered = scores
    if top_k is not None and top_k > 0:
        k = min(int(top_k), filtered.size(-1))
        top_vals, _ = torch.topk(filtered, k, dim=-1)
        kth = top_vals[:, -1].unsqueeze(-1)
        filtered = filtered.masked_fill(filtered < kth, float("-inf"))
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_scores, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_scores, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > float(top_p)
        sorted_remove[:, 0] = False
        remove = torch.zeros_like(sorted_remove, dtype=torch.bool)
        remove.scatter_(1, sorted_indices, sorted_remove)
        filtered = filtered.masked_fill(remove, float("-inf"))
    return filtered


def _build_heldout_pairs(
    val_data: torch.Tensor,
    *,
    prompt_len: int,
    continuation_len: int,
    num_examples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_len = int(prompt_len) + int(continuation_len)
    available = int(val_data.numel()) - total_len
    if available <= 0:
        raise ValueError(
            f"Validation split too short for prompt_len={prompt_len} and continuation_len={continuation_len}."
        )
    n = min(max(int(num_examples), 1), max(1, available))
    if n == 1:
        starts = torch.zeros(1, dtype=torch.long)
    else:
        starts = torch.linspace(0, available - 1, steps=n, dtype=torch.float32).round().to(dtype=torch.long)
    offsets = torch.arange(total_len, dtype=torch.long)
    seq = val_data.index_select(0, (starts.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)).view(n, total_len)
    prompts = seq[:, :prompt_len].contiguous()
    true_cont = seq[:, prompt_len:].contiguous()
    return prompts, true_cont, starts


def _build_prompt_pair(
    val_data: torch.Tensor,
    *,
    prompt_tokens: list[int],
    continuation_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not prompt_tokens:
        raise ValueError("Prompt tokenized to an empty sequence.")
    m = len(prompt_tokens)
    n = int(val_data.numel())
    val_list = val_data.tolist()
    for idx in range(0, n - m - continuation_len):
        if val_list[idx: idx + m] != prompt_tokens:
            continue
        prompt = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0)
        true_cont = torch.tensor(
            val_list[idx + m: idx + m + continuation_len],
            dtype=torch.long,
        ).unsqueeze(0)
        start = torch.tensor([idx], dtype=torch.long)
        return prompt, true_cont, start
    raise RuntimeError("Prompt token sequence not found in held-out validation split.")


@torch.no_grad()
def _sample_drifted_continuations(
    model: nn.Module,
    prompts: torch.Tensor,
    *,
    continuation_len: int,
    window: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    repetition_window: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, prompts.size(0), batch_size):
        prompt = prompts[start: start + batch_size].to(device, non_blocking=True)
        context = prompt
        generated = []
        for _ in range(int(continuation_len)):
            x_cond = context[:, -int(window):]
            scores = model(x_cond)[:, -1, :].float() / max(float(temperature), 1e-6)
            if repetition_penalty > 1.0 and scores.size(-1) > 0:
                seen = context[:, -min(int(repetition_window), context.size(1)):] if repetition_window > 0 else context
                for b in range(scores.size(0)):
                    seen_ids = torch.unique(seen[b])
                    seen_scores = scores[b, seen_ids]
                    seen_scores = torch.where(
                        seen_scores > 0,
                        seen_scores / float(repetition_penalty),
                        seen_scores * float(repetition_penalty),
                    )
                    scores[b, seen_ids] = seen_scores
            scores = _top_p_top_k(scores, top_k=top_k, top_p=top_p)
            probs = F.softmax(scores, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)
            generated.append(next_ids)
            context = torch.cat((context, next_ids), dim=1)
        chunks.append(torch.cat(generated, dim=1).cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _continuation_ce_per_example(
    model: nn.Module,
    prompts: torch.Tensor,
    continuation: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    model.eval()
    prompt_len = int(prompts.size(1))
    per_example = []
    for start in range(0, prompts.size(0), batch_size):
        prompt = prompts[start: start + batch_size].to(device, non_blocking=True)
        cont = continuation[start: start + batch_size].to(device, non_blocking=True)
        seq = torch.cat((prompt, cont), dim=1)
        x = seq[:, :-1]
        y = seq[:, 1:].clone()
        if prompt_len > 1:
            y[:, : prompt_len - 1] = -100
        logits = model(x).float()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(y)
        counts = (y != -100).sum(dim=1).clamp(min=1)
        per_example.append((loss.sum(dim=1) / counts).cpu())
    ce_values = torch.cat(per_example, dim=0)
    mean_ce = float(ce_values.mean().item())
    ppl = math.exp(min(mean_ce, 20.0))
    return ce_values, mean_ce, ppl


def _build_model_from_args(args: argparse.Namespace, *, vocab_size: int):
    adaptive_cutoffs = None
    if args.head_mode == "adaptive":
        adaptive_cutoffs = resolve_adaptive_cutoffs(vocab_size, parse_adaptive_cutoffs(args.adaptive_cutoffs))
    return build_model(
        model_config_from_args(
            args,
            vocab_size=vocab_size,
            adaptive_cutoffs=adaptive_cutoffs,
            emb_dropout=0.0,
            mlp_dropout=0.0,
            residual_dropout=0.0,
        )
    )


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
    probe_device = _resolve_probe_device(str(args.probe_device).strip().lower())
    LOG.info("Runtime | probe_device=%s | seed=%d | deterministic=%s", probe_device, args.seed, str(args.deterministic).lower())
    LOG.info("Run config | hash=%s", _run_config_hash(args))
    LOG.info("Command | %s", _command_string())
    log_runtime_metadata()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if int(args.sample_tokens) <= 0:
        raise ValueError(f"--sample-tokens must be > 0, got {args.sample_tokens}.")
    if int(args.batch_size) <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}.")
    if args.probe_mode == "heldout" and int(args.prompt_len) <= 0:
        raise ValueError(f"--prompt-len must be > 0 in heldout mode, got {args.prompt_len}.")

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
            remap_by_frequency=args.head_mode == "adaptive",
        )
    )
    tokenizer = dataset_bundle.tokenizer
    val_data = dataset_bundle.val_data.cpu()

    model = _build_model_from_args(args, vocab_size=tokenizer.vocab_size).to(probe_device)
    loaded_step, loaded_best_ppl = load_model_checkpoint(ckpt_path, model)
    LOG.info("Checkpoint loaded | step=%s | best_ppl=%s | path=%s", loaded_step, loaded_best_ppl, ckpt_path)

    continuation_len = int(args.sample_tokens)
    if args.probe_mode == "prompt":
        prompt_tokens = tokenizer.encode(args.prompt)
        prompts, true_cont, starts = _build_prompt_pair(
            val_data,
            prompt_tokens=prompt_tokens,
            continuation_len=continuation_len,
        )
    else:
        prompts, true_cont, starts = _build_heldout_pairs(
            val_data,
            prompt_len=int(args.prompt_len),
            continuation_len=continuation_len,
            num_examples=int(args.num_examples),
        )

    top_k = args.top_k if args.top_k > 0 else None
    top_p = args.top_p if 0.0 < args.top_p < 1.0 else None
    drift_cont = _sample_drifted_continuations(
        model,
        prompts,
        continuation_len=continuation_len,
        window=int(args.window),
        temperature=float(args.temperature),
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=max(float(args.repetition_penalty), 1.0),
        repetition_window=int(args.repetition_window),
        batch_size=int(args.batch_size),
        device=probe_device,
    )
    ce_true_ex, ce_true, ppl_true = _continuation_ce_per_example(
        model,
        prompts,
        true_cont,
        batch_size=int(args.batch_size),
        device=probe_device,
    )
    ce_drift_ex, ce_drift, ppl_drift = _continuation_ce_per_example(
        model,
        prompts,
        drift_cont,
        batch_size=int(args.batch_size),
        device=probe_device,
    )
    delta = ce_drift_ex - ce_true_ex
    overlap = (drift_cont == true_cont).float().mean(dim=1)
    drift_better_rate = float((delta < 0).float().mean().item())

    result: Dict[str, object] = {
        "checkpoint": str(ckpt_path),
        "step": loaded_step,
        "best_ppl": loaded_best_ppl,
        "probe_mode": args.probe_mode,
        "probe_device": str(probe_device),
        "num_examples": int(prompts.size(0)),
        "prompt_len": int(prompts.size(1)),
        "continuation_len": continuation_len,
        "starts_min": int(starts.min().item()),
        "starts_max": int(starts.max().item()),
        "true_ce": ce_true,
        "true_ppl": ppl_true,
        "drift_ce": ce_drift,
        "drift_ppl": ppl_drift,
        "delta_ce": ce_drift - ce_true,
        "ppl_ratio": ppl_drift / max(ppl_true, 1e-9),
        "delta_mean": float(delta.mean().item()),
        "delta_median": float(delta.median().item()),
        "delta_p10": float(torch.quantile(delta, 0.10).item()),
        "delta_p90": float(torch.quantile(delta, 0.90).item()),
        "token_overlap_mean": float(overlap.mean().item()),
        "token_overlap_median": float(overlap.median().item()),
        "drift_better_rate": drift_better_rate,
    }

    print("Drift probe | held-out continuation likelihood")
    print(f"  ckpt={result['checkpoint']}")
    print(f"  loaded_step={result['step']} best_ppl={result['best_ppl']}")
    print(
        "  mode={probe_mode} | examples={num_examples} | prompt_len={prompt_len} | continuation_len={continuation_len}".format(
            **result
        )
    )
    print(f"  starts[min,max]=[{result['starts_min']},{result['starts_max']}]")
    print(f"  true_ce={result['true_ce']:.4f} true_ppl={result['true_ppl']:.2f}")
    print(f"  drift_ce={result['drift_ce']:.4f} drift_ppl={result['drift_ppl']:.2f}")
    print(f"  delta_ce(drift-true)={result['delta_ce']:+.4f} | ppl_ratio={result['ppl_ratio']:.3f}")
    print(
        "  delta_per_example mean={delta_mean:.4f} median={delta_median:.4f} p10={delta_p10:.4f} p90={delta_p90:.4f}".format(
            **result
        )
    )
    print(
        "  token_overlap mean={token_overlap_mean:.3f} median={token_overlap_median:.3f} | drift_better_rate={drift_better_rate:.3f}".format(
            **result
        )
    )

    idx = 0
    prompt_text = tokenizer.decode(prompts[idx].tolist()).replace("\n", "\\n")
    true_text = tokenizer.decode(true_cont[idx].tolist()).replace("\n", "\\n")
    drift_text = tokenizer.decode(drift_cont[idx].tolist()).replace("\n", "\\n")
    print(f"  sample_prompt={prompt_text[:180]}")
    print(f"  sample_true={true_text[:180]}")
    print(f"  sample_drift={drift_text[:180]}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        LOG.info("Drift probe report written | path=%s", out_path)


if __name__ == "__main__":
    main()
