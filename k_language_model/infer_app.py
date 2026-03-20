import argparse
import time
from pathlib import Path

import torch

from .checkpoint import load_model_checkpoint
from .data import load_dataset
from .generation import sample_text
from .model import KStackModel, resolve_adaptive_cutoffs
from .runtime import (
    DEVICE,
    LOG,
    USE_AMP,
    _command_string,
    _run_config_hash,
    _unwrap_model,
    configure_reproducibility,
    log_runtime_metadata,
    maybe_enable_compile,
    setup_logging,
)
from .trainer import _collect_eval_refine_stats, ce_to_bpc, eval_deterministic


def _parse_adaptive_cutoffs(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",")]
    parsed = [int(value) for value in values if value]
    return parsed or None


def _usable_eval_tokens(data: torch.Tensor, window: int) -> int:
    n_tokens = len(data) - 1
    if n_tokens <= 0:
        return 0
    return (n_tokens // window) * window


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference/evaluation with a trained K-Stack checkpoint.")
    p.add_argument(
        "--dataset",
        type=str,
        choices=["shakespeare", "wikitext2", "wikitext2_raw"],
        default="shakespeare",
        help="Dataset preset used to build eval data/vocabulary.",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional train/source text path. If omitted, built-in dataset files are used/downloaded.",
    )
    p.add_argument(
        "--val-path",
        type=str,
        default=None,
        help="Optional validation text path. If omitted, preset val split is used or val_frac split is applied.",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation fraction used only when a separate validation file is not available/provided.",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        choices=["char", "sentencepiece"],
        default="char",
        help="Tokenizer used to rebuild the training/eval token vocabulary.",
    )
    p.add_argument(
        "--sp-model",
        type=str,
        default=None,
        help="Existing SentencePiece model path used for the checkpoint.",
    )
    p.add_argument("--sp-vocab-size", type=int, default=4096)
    p.add_argument(
        "--sp-model-type",
        type=str,
        choices=["unigram", "bpe", "char", "word"],
        default="unigram",
    )
    p.add_argument("--sp-character-coverage", type=float, default=1.0)
    p.add_argument("--sp-split-digits", action="store_true")
    p.add_argument("--sp-byte-fallback", action="store_true")
    p.add_argument("--ckpt", "--checkpoint", dest="ckpt", type=str, required=True, help="Checkpoint path to load.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size used for deterministic validation eval.")
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument(
        "--emb-dim",
        type=int,
        default=None,
        help="Optional token embedding dimension used by the checkpoint.",
    )
    p.add_argument("--rank", type=int, default=32)
    p.add_argument(
        "--k-base-rank",
        type=int,
        default=2,
        help="Low-rank factorization rank for causal k_base path. Set <=0 to use dense k_base.",
    )
    p.add_argument(
        "--k-base-impl",
        type=str,
        choices=["auto", "fused", "scan"],
        default="auto",
        help="Low-rank k_base kernel: auto picks fused on accelerators when temporary tensors fit; scan minimizes memory.",
    )
    p.add_argument(
        "--share-k-base",
        dest="share_k_base",
        action="store_true",
        help="Use one shared dense learnable k_base matrix across all K2 layers (requires --k-base-rank <= 0).",
    )
    p.add_argument(
        "--no-share-k-base",
        dest="share_k_base",
        action="store_false",
        help="Use per-layer dense k_base matrices when --k-base-rank <= 0.",
    )
    p.set_defaults(share_k_base=False)
    p.add_argument("--n-k2", type=int, default=4)
    p.add_argument("--head-mode", type=str, choices=["linear", "gelu", "adaptive"], default="linear")
    p.add_argument("--head-mult", type=int, default=6)
    p.add_argument("--head-dropout", type=float, default=0.10)
    p.add_argument(
        "--adaptive-cutoffs",
        type=str,
        default=None,
        help="Comma-separated adaptive softmax cutoffs used by the checkpoint.",
    )
    p.add_argument("--adaptive-div-value", type=float, default=4.0)
    p.add_argument(
        "--refine-steps",
        type=int,
        default=8,
        help="Iterative refinement steps used during eval/inference.",
    )
    p.add_argument(
        "--train-refine-steps",
        type=int,
        default=None,
        help="Train-time refinement setting for checkpoint shape compatibility.",
    )
    p.add_argument(
        "--eval-refine-steps",
        type=int,
        default=None,
        help="Override refine steps for eval/inference after loading checkpoint.",
    )
    p.add_argument("--alpha-cap", type=float, default=0.8)
    p.add_argument(
        "--decay-impl",
        type=str,
        choices=["mask", "block", "kernel"],
        default="mask",
        help="Gamma-decay backend used by the checkpoint architecture (kernel is experimental Triton CUDA path).",
    )
    p.add_argument("--gamma-min", type=float, default=0.85, help="Lower bound for per-rank gamma decay values.")
    p.add_argument("--gamma-max", type=float, default=1.0, help="Upper bound for per-rank gamma decay values.")
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
        "--strict-repro",
        action="store_true",
        help="Force strict reproducibility (disables compile, enables deterministic mode and disables TF32).",
    )
    p.add_argument("--compile", action="store_true", help="Enable torch.compile for model forward pass.")
    p.add_argument(
        "--compile-mode",
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    p.add_argument("--skip-eval", action="store_true", help="Skip deterministic validation CE/PPL evaluation.")
    p.add_argument("--skip-sample", action="store_true", help="Skip text sampling.")
    p.add_argument("--prompt", type=str, default="To be, or not to be")
    p.add_argument("--sample-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--repetition-window", type=int, default=256)
    p.add_argument("--prompt-lock-tokens", type=int, dest="prompt_lock_tokens", default=0)
    p.add_argument("--prompt-lock-chars", type=int, dest="prompt_lock_tokens", help=argparse.SUPPRESS)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    if args.strict_repro:
        args.deterministic = True
        args.deterministic_warn_only = False
        args.no_tf32 = True
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
    LOG.info("Run config | hash=%s", _run_config_hash(args))
    LOG.info("Command | %s", _command_string())
    log_runtime_metadata()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    remap_by_frequency = args.head_mode == "adaptive"
    _, val_data, tokenizer = load_dataset(
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
    vocab_size = tokenizer.vocab_size
    adaptive_cutoffs = None
    if args.head_mode == "adaptive":
        adaptive_cutoffs = resolve_adaptive_cutoffs(vocab_size, _parse_adaptive_cutoffs(args.adaptive_cutoffs))
        LOG.info(
            "Adaptive head | cutoffs=%s | div_value=%.2f | vocab_frequency_remap=%s",
            adaptive_cutoffs,
            args.adaptive_div_value,
            str(remap_by_frequency).lower(),
        )

    model = KStackModel(
        vocab_size=vocab_size,
        window=args.window,
        d=args.d_model,
        emb_dim=args.emb_dim,
        rank=args.rank,
        n_k2=args.n_k2,
        k_base_rank=args.k_base_rank,
        k_base_impl=args.k_base_impl,
        share_k_base=args.share_k_base,
        emb_dropout=0.0,
        mlp_dropout=0.0,
        residual_dropout=0.0,
        head_mode=args.head_mode,
        head_mult=args.head_mult,
        head_dropout=args.head_dropout,
        adaptive_cutoffs=adaptive_cutoffs,
        adaptive_div_value=args.adaptive_div_value,
        refine_steps=args.refine_steps,
        train_refine_steps=args.train_refine_steps,
        alpha_cap=args.alpha_cap,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        decay_impl=args.decay_impl,
    )

    if args.compile:
        model = maybe_enable_compile(model, enabled=True, mode=args.compile_mode)

    model = model.to(DEVICE)
    loaded_step, loaded_best_ppl = load_model_checkpoint(ckpt_path, model)
    core_model = _unwrap_model(model)

    if args.eval_refine_steps is not None:
        core_model.refine_steps = max(int(args.eval_refine_steps), 0)
        core_model.reset_eval_refine_diagnostics()

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
            "Model config | tokenizer=%s | emb_dim=%d | d_model=%d | tied_weights=%s | head_mode=%s | k_base_rank=%d | k_base_impl=%s | share_k_base=%s | gamma[min/max]=%.3f/%.3f",
            tokenizer.describe(),
            core_model.emb_dim,
            core_model.d_model,
            str(getattr(core_model, "tie_weights", False)).lower(),
            args.head_mode,
            core_model.k_base_rank,
            core_model.k_base_impl,
            str(getattr(core_model, "share_k_base", False)).lower(),
            args.gamma_min,
            args.gamma_max,
        )
        if getattr(core_model, "adaptive_cutoffs", []):
            LOG.info("Adaptive config | cutoffs=%s | div_value=%.2f", core_model.adaptive_cutoffs, core_model.adaptive_div_value)

    if not args.skip_eval:
        eval_t0 = time.perf_counter()
        ce, ppl = eval_deterministic(model, val_data, args.window, args.batch_size)
        eval_elapsed = max(time.perf_counter() - eval_t0, 1e-9)
        eval_tokens = _usable_eval_tokens(val_data, args.window)
        eval_tok_s = eval_tokens / eval_elapsed if eval_tokens > 0 else 0.0
        loaded_step_str = "N/A" if loaded_step is None else str(loaded_step)
        loaded_best_ppl_str = "N/A" if loaded_best_ppl is None else f"{loaded_best_ppl:.2f}"
        if args.dataset in {"wikitext2", "wikitext2_raw"} and tokenizer.is_character_level:
            LOG.info(
                "Eval | step=%s | ckpt_best_ppl=%s | refine_steps=%d | val_bpc=%.4f | val_ppl=%.2f | eval_tok_s=%.0f",
                loaded_step_str,
                loaded_best_ppl_str,
                core_model.refine_steps,
                ce_to_bpc(ce),
                ppl,
                eval_tok_s,
            )
        else:
            LOG.info(
                "Eval | step=%s | ckpt_best_ppl=%s | refine_steps=%d | val_ce=%.4f | val_ppl=%.2f | eval_tok_s=%.0f",
                loaded_step_str,
                loaded_best_ppl_str,
                core_model.refine_steps,
                ce,
                ppl,
                eval_tok_s,
            )
        eval_refine_stats = _collect_eval_refine_stats(model)
        if eval_refine_stats:
            LOG.info("eval_refinement | %s", eval_refine_stats)

    if not args.skip_sample:
        top_k = args.top_k if args.top_k > 0 else None
        top_p = args.top_p if 0.0 < args.top_p < 1.0 else None
        prompt_tokens = len(tokenizer.encode(args.prompt))
        sample_t0 = time.perf_counter()
        text = sample_text(
            model,
            tokenizer,
            args.prompt,
            args.sample_tokens,
            args.window,
            temperature=args.temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=max(args.repetition_penalty, 1.0),
            repetition_window=args.repetition_window,
            prompt_lock_tokens=max(args.prompt_lock_tokens, 0),
        )
        sample_elapsed = max(time.perf_counter() - sample_t0, 1e-9)
        sample_tok_s = max(int(args.sample_tokens), 0) / sample_elapsed if args.sample_tokens > 0 else 0.0
        LOG.info(
            "Sample speed | prompt_tokens=%d | generated_tokens=%d | sample_tok_s=%.0f",
            prompt_tokens,
            max(int(args.sample_tokens), 0),
            sample_tok_s,
        )
        LOG.info("Sample:\n%s", text)


if __name__ == "__main__":
    main()
