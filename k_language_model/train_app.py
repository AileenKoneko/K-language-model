import argparse
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
    maybe_write_run_manifest,
    setup_logging,
)
from .trainer import TrainConfig, _collect_eval_refine_stats, ce_to_bpc, eval_deterministic, train_model


def _parse_adaptive_cutoffs(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",")]
    parsed = [int(value) for value in values if value]
    return parsed or None


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description or "Train and evaluate a K-Stack token-level language model."
    )
    p.add_argument(
        "--dataset",
        type=str,
        choices=["shakespeare", "wikitext2", "wikitext2_raw"],
        default="shakespeare",
        help="Dataset preset to use.",
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
        help="Tokenizer used to build the training and validation sequences.",
    )
    p.add_argument(
        "--sp-model",
        type=str,
        default=None,
        help="SentencePiece model path. If omitted during training, a model is trained into data/tokenizers/.",
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
    p.add_argument("--steps", type=int, default=25000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument(
        "--emb-dim",
        type=int,
        default=None,
        help="Optional token embedding dimension. Defaults to --d-model when omitted.",
    )
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--n-k2", type=int, default=4)
    p.add_argument("--head-mode", type=str, choices=["linear", "gelu", "adaptive"], default="linear")
    p.add_argument("--head-mult", type=int, default=6)
    p.add_argument("--head-dropout", type=float, default=0.10)
    p.add_argument(
        "--adaptive-cutoffs",
        type=str,
        default=None,
        help="Comma-separated class cutoffs for adaptive softmax. If omitted, a heuristic is used from vocab size.",
    )
    p.add_argument("--adaptive-div-value", type=float, default=4.0)
    p.add_argument("--emb-dropout", type=float, default=0.08)
    p.add_argument("--mlp-dropout", type=float, default=0.10)
    p.add_argument("--residual-dropout", type=float, default=0.05)
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
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable verbose diagnostic logging (grad stats, layer stats, update/weight ratios) at eval intervals.",
    )
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
        "--prompt-lock-tokens",
        type=int,
        dest="prompt_lock_tokens",
        default=0,
        help="Keep first N prompt tokens in the conditioning window during long generation.",
    )
    p.add_argument("--prompt-lock-chars", type=int, dest="prompt_lock_tokens", help=argparse.SUPPRESS)
    p.add_argument("--verbose", action="store_true")
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


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

    remap_by_frequency = args.head_mode == "adaptive"
    train_data, val_data, tokenizer = load_dataset(
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
        allow_training_tokenizer=True,
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

    cfg = TrainConfig(
        window=args.window,
        d_model=args.d_model,
        rank=args.rank,
        n_k2=args.n_k2,
        alpha_cap=args.alpha_cap,
        emb_dropout=args.emb_dropout,
        mlp_dropout=args.mlp_dropout,
        residual_dropout=args.residual_dropout,
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
        diagnostics=args.diagnostics,
        report_bpc=(args.dataset in {"wikitext2", "wikitext2_raw"} and tokenizer.is_character_level),
    )

    model = KStackModel(
        vocab_size=vocab_size,
        window=cfg.window,
        d=cfg.d_model,
        emb_dim=args.emb_dim,
        rank=cfg.rank,
        n_k2=cfg.n_k2,
        emb_dropout=cfg.emb_dropout,
        mlp_dropout=cfg.mlp_dropout,
        residual_dropout=cfg.residual_dropout,
        head_mode=args.head_mode,
        head_mult=args.head_mult,
        head_dropout=args.head_dropout,
        adaptive_cutoffs=adaptive_cutoffs,
        adaptive_div_value=args.adaptive_div_value,
        refine_steps=args.refine_steps,
        train_refine_steps=args.train_refine_steps,
        alpha_cap=cfg.alpha_cap,
        decay_impl=args.decay_impl,
    )

    if args.compile:
        model = maybe_enable_compile(model, enabled=True, mode=args.compile_mode)
    elif not args.deterministic:
        LOG.warning(
            "Non-deterministic fast mode active: run-to-run CE can drift with AMP/fused/compile behavior. "
            "Use --strict-repro for exact reproducibility."
        )

    model_for_stats = _unwrap_model(model)
    params = model_for_stats.count_params() if hasattr(model_for_stats, "count_params") else {}
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
            "Model config | tokenizer=%s | emb_dim=%d | d_model=%d | tied_weights=%s | head_mode=%s | head_mult=%d | head_dropout=%.2f | alpha_cap=%.2f | refine_steps[train/eval]=%d/%d",
            tokenizer.describe(),
            model_for_stats.emb_dim,
            model_for_stats.d_model,
            str(getattr(model_for_stats, "tie_weights", False)).lower(),
            args.head_mode,
            args.head_mult,
            args.head_dropout,
            cfg.alpha_cap,
            model_for_stats.train_refine_steps,
            model_for_stats.refine_steps,
        )
        if getattr(model_for_stats, "adaptive_cutoffs", []):
            LOG.info("Adaptive config | cutoffs=%s | div_value=%.2f", model_for_stats.adaptive_cutoffs, model_for_stats.adaptive_div_value)
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
        if cfg.report_bpc:
            LOG.info(
                "Eval only | step=%s | ckpt_best_ppl=%s | refine_steps=%d | val_bpc=%.4f | val_ppl=%.2f",
                loaded_step_str,
                loaded_best_ppl_str,
                core_model.refine_steps,
                ce_to_bpc(ce),
                ppl,
            )
        else:
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
                tokenizer,
                args.prompt,
                args.sample_tokens,
                cfg.window,
                temperature=args.temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=max(args.repetition_penalty, 1.0),
                repetition_window=args.repetition_window,
                prompt_lock_tokens=max(args.prompt_lock_tokens, 0),
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
            tokenizer,
            args.prompt,
            args.sample_tokens,
            cfg.window,
            temperature=args.temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=max(args.repetition_penalty, 1.0),
            repetition_window=args.repetition_window,
            prompt_lock_tokens=max(args.prompt_lock_tokens, 0),
        )
        LOG.info("Sample:\n%s", text)


if __name__ == "__main__":
    main()
