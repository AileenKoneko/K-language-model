import argparse
from pathlib import Path

import torch

from .checkpoint import load_model_checkpoint
from .cli_args import (
    add_dataset_args,
    add_dynamics_args,
    add_experimental_objective_args,
    add_model_args,
    add_repro_runtime_args,
    add_sampling_args,
)
from .configs import DatasetConfig
from .data import load_dataset_bundle
from .generation import sample_text
from .model import resolve_adaptive_cutoffs
from .model_factory import build_model, model_config_from_args, parse_adaptive_cutoffs, parse_int_list
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
from .trainer import (
    TrainConfig,
    _collect_k_layer_stats,
    ce_to_bpc,
    eval_deterministic,
    train_model,
)


def _build_model(
    args: argparse.Namespace,
    *,
    vocab_size: int,
    adaptive_cutoffs: list[int] | None,
    emb_dropout: float,
    mlp_dropout: float,
    residual_dropout: float,
):
    return build_model(
        model_config_from_args(
            args,
            vocab_size=vocab_size,
            adaptive_cutoffs=adaptive_cutoffs,
            emb_dropout=emb_dropout,
            mlp_dropout=mlp_dropout,
            residual_dropout=residual_dropout,
        )
    )


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description or "Train and evaluate a K-Stack token-level language model."
    )
    add_dataset_args(
        p,
        tokenizer_help="Tokenizer used to build the training and validation sequences.",
        sp_model_help="SentencePiece model path. If omitted during training, a model is trained into data/tokenizers/.",
    )
    p.add_argument("--steps", type=int, default=25000)
    p.add_argument("--batch-size", type=int, default=256)
    add_model_args(
        p,
        include_dropouts=True,
        adaptive_cutoffs_help="Comma-separated class cutoffs for adaptive softmax. If omitted, a heuristic is used from vocab size.",
    )
    add_dynamics_args(
        p,
        decay_help="Gamma-decay backend: mask (baseline), block (lower memory), kernel (experimental Triton CUDA path).",
    )
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
        "--diagnostic",
        action="store_true",
        help="Enable verbose diagnostic logging (grad stats, layer stats, update/weight ratios) at eval intervals.",
    )
    add_experimental_objective_args(p)
    add_repro_runtime_args(
        p,
        include_compile=True,
        include_run_manifest=True,
        strict_repro_help="Force strict reproducibility (disables compile and fused AdamW, enables deterministic mode and disables TF32).",
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
    add_sampling_args(
        p,
        include_sample_flag=True,
        include_skip_flags=False,
        temperature_help="Sampling temperature for --sample.",
        top_k_help="Top-k sampling for --sample. 0 disables top-k.",
        top_p_help="Top-p (nucleus) sampling for --sample. 0 disables top-p.",
        repetition_penalty_help="Repetition penalty (>1 discourages repeated tokens). 1.0 disables.",
        repetition_window_help="Recent token window for repetition penalty. 0 means full generated context.",
        prompt_lock_help="Keep first N prompt tokens in the conditioning window during long generation.",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


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
    dataset_config = DatasetConfig(
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
    dataset_bundle = load_dataset_bundle(dataset_config)
    train_data = dataset_bundle.train_data
    val_data = dataset_bundle.val_data
    tokenizer = dataset_bundle.tokenizer
    vocab_size = tokenizer.vocab_size
    adaptive_cutoffs = None
    if args.head_mode == "adaptive":
        adaptive_cutoffs = resolve_adaptive_cutoffs(vocab_size, parse_adaptive_cutoffs(args.adaptive_cutoffs))
        LOG.info(
            "Adaptive head | cutoffs=%s | div_value=%.2f | vocab_frequency_remap=%s",
            adaptive_cutoffs,
            args.adaptive_div_value,
            str(remap_by_frequency).lower(),
        )

    future_horizons = parse_int_list(args.future_summary_horizons)
    if not future_horizons and int(args.future_summary_horizon) > 0:
        future_horizons = (int(args.future_summary_horizon),)

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
        future_summary_horizons=future_horizons,
        future_summary_lambda=max(float(args.future_summary_lambda), 0.0),
        future_summary_lambda_min=max(float(args.future_summary_lambda_min), 0.0),
        future_summary_ce_target=(
            float(args.future_summary_ce_target) if args.future_summary_ce_target is not None else None
        ),
        future_summary_ce_anchor=(
            float(args.future_summary_ce_anchor) if args.future_summary_ce_anchor is not None else None
        ),
        future_summary_start_step=0,
        future_summary_eval_batches=max(int(args.future_summary_eval_batches), 0),
        rollout_horizon=max(int(args.rollout_horizon), 0),
        rollout_lambda=max(float(args.rollout_lambda), 0.0),
        rollout_start_step=0,
        rollout_mode=str(args.rollout_mode).strip().lower(),
        semantic_lambda=max(float(args.semantic_lambda), 0.0),
        semantic_start_step=0,
        rollout_eval_batches=max(int(args.rollout_eval_batches), 0),
        rollout_useful_ce_tol=max(float(args.rollout_useful_ce_tol), 0.0),
    )
    if cfg.future_summary_horizons and cfg.future_summary_lambda > 0.0:
        cfg.future_summary_start_step = (
            args.future_summary_start_step if args.future_summary_start_step is not None else args.warmup_steps
        )
    if cfg.rollout_horizon > 0 and cfg.rollout_lambda > 0.0:
        cfg.rollout_start_step = args.rollout_start_step if args.rollout_start_step is not None else args.warmup_steps
    if cfg.rollout_horizon > 0 and cfg.rollout_lambda > 0.0 and cfg.semantic_lambda > 0.0:
        default_semantic_start = max(cfg.rollout_start_step, int(0.75 * max(args.steps, 1)))
        cfg.semantic_start_step = (
            args.semantic_start_step if args.semantic_start_step is not None else default_semantic_start
        )

    LOG.info("Model family | version=v2")
    model = build_model(
        model_config_from_args(
            args,
            vocab_size=vocab_size,
            adaptive_cutoffs=adaptive_cutoffs,
            emb_dropout=cfg.emb_dropout,
            mlp_dropout=cfg.mlp_dropout,
            residual_dropout=cfg.residual_dropout,
        )
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
            "Model config | tokenizer=%s | emb_dim=%d | d_model=%d | tied_weights=%s | head_mode=%s | head_mult=%d | head_dropout=%.2f | k_base_rank=%d | k_base_impl=%s | share_k_base=%s | rosa_impl=%s | rosa_layers=%s | alpha_cap=%.2f | gamma[min/max]=%.3f/%.3f",
            tokenizer.describe(),
            model_for_stats.emb_dim,
            model_for_stats.d_model,
            str(getattr(model_for_stats, "tie_weights", False)).lower(),
            args.head_mode,
            args.head_mult,
            args.head_dropout,
            model_for_stats.k_base_rank,
            model_for_stats.k_base_impl,
            str(getattr(model_for_stats, "share_k_base", False)).lower(),
            model_for_stats.rosa_impl,
            model_for_stats.describe_rosa_layers(),
            cfg.alpha_cap,
            args.gamma_min,
            args.gamma_max,
        )
        if getattr(model_for_stats, "adaptive_cutoffs", []):
            LOG.info("Adaptive config | cutoffs=%s | div_value=%.2f", model_for_stats.adaptive_cutoffs, model_for_stats.adaptive_div_value)

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if args.eval_only:
        if ckpt_path is None:
            raise ValueError("--eval-only requires --ckpt.")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = model.to(DEVICE)
        loaded_step, loaded_best_ppl = load_model_checkpoint(ckpt_path, model)
        core_model = _unwrap_model(model)
        ce, ppl = eval_deterministic(model, val_data, cfg.window, cfg.batch_size)
        loaded_step_str = "N/A" if loaded_step is None else str(loaded_step)
        loaded_best_ppl_str = "N/A" if loaded_best_ppl is None else f"{loaded_best_ppl:.2f}"
        if cfg.report_bpc:
            LOG.info(
                "Eval only | step=%s | ckpt_best_ppl=%s | val_bpc=%.4f | val_ppl=%.2f",
                loaded_step_str,
                loaded_best_ppl_str,
                ce_to_bpc(ce),
                ppl,
            )
        else:
            LOG.info(
                "Eval only | step=%s | ckpt_best_ppl=%s | val_ce=%.4f | val_ppl=%.2f",
                loaded_step_str,
                loaded_best_ppl_str,
                ce,
                ppl,
            )
        if args.diagnostics:
            k_stats = _collect_k_layer_stats(model)
            if k_stats:
                LOG.info("layer_stats | %s", k_stats)

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
