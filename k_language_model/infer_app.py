import argparse
from pathlib import Path

import torch

from .checkpoint import load_model_checkpoint
from .data import load_dataset
from .generation import sample_text
from .model import KStackModel
from .runtime import (
    DEVICE,
    LOG,
    USE_AMP,
    _command_string,
    _run_config_hash,
    _unwrap_model,
    configure_reproducibility,
    log_runtime_metadata,
    setup_logging,
)
from .trainer import _collect_eval_refine_stats, ce_to_bpc, eval_deterministic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference/evaluation with a trained K-Stack checkpoint.")
    p.add_argument(
        "--dataset",
        type=str,
        choices=["shakespeare", "wikitext2"],
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
    p.add_argument("--ckpt", "--checkpoint", dest="ckpt", type=str, required=True, help="Checkpoint path to load.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size used for deterministic validation eval.")
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
        choices=["mask", "block"],
        default="mask",
        help="Gamma-decay backend used by the checkpoint architecture.",
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
    p.add_argument("--prompt-lock-chars", type=int, default=0)
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

    _, val_data, vocab_size, stoi, itos = load_dataset(
        dataset=args.dataset,
        val_frac=args.val_frac,
        data_path=args.data_path,
        val_path=args.val_path,
    )

    model = KStackModel(
        vocab_size=vocab_size,
        window=args.window,
        d=args.d_model,
        rank=args.rank,
        n_k2=args.n_k2,
        emb_dropout=0.0,
        mlp_dropout=0.0,
        residual_dropout=0.0,
        head_mode=args.head_mode,
        head_mult=args.head_mult,
        head_dropout=args.head_dropout,
        refine_steps=args.refine_steps,
        train_refine_steps=args.train_refine_steps,
        alpha_cap=args.alpha_cap,
        decay_impl=args.decay_impl,
    )

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
        LOG.info("Compile config | mode=%s", args.compile_mode)

    model = model.to(DEVICE)
    loaded_step, loaded_best_ppl = load_model_checkpoint(ckpt_path, model)
    core_model = _unwrap_model(model)

    if args.eval_refine_steps is not None:
        core_model.refine_steps = max(int(args.eval_refine_steps), 0)
        core_model.reset_eval_refine_diagnostics()

    params = core_model.count_params() if hasattr(core_model, "count_params") else {}
    if params:
        LOG.info(
            "Model params | total=%s | embedding=%s | k_stack=%s | head=%s",
            f"{params['total']:,}",
            f"{params['embedding']:,}",
            f"{params['k_stack']:,}",
            f"{params['head']:,}",
        )

    if not args.skip_eval:
        ce, ppl = eval_deterministic(model, val_data, args.window, args.batch_size)
        loaded_step_str = "N/A" if loaded_step is None else str(loaded_step)
        loaded_best_ppl_str = "N/A" if loaded_best_ppl is None else f"{loaded_best_ppl:.2f}"
        if args.dataset == "wikitext2":
            LOG.info(
                "Eval | step=%s | ckpt_best_ppl=%s | refine_steps=%d | val_bpc=%.4f | val_ppl=%.2f",
                loaded_step_str,
                loaded_best_ppl_str,
                core_model.refine_steps,
                ce_to_bpc(ce),
                ppl,
            )
        else:
            LOG.info(
                "Eval | step=%s | ckpt_best_ppl=%s | refine_steps=%d | val_ce=%.4f | val_ppl=%.2f",
                loaded_step_str,
                loaded_best_ppl_str,
                core_model.refine_steps,
                ce,
                ppl,
            )
        eval_refine_stats = _collect_eval_refine_stats(model)
        if eval_refine_stats:
            LOG.info("eval_refinement | %s", eval_refine_stats)

    if not args.skip_sample:
        top_k = args.top_k if args.top_k > 0 else None
        top_p = args.top_p if 0.0 < args.top_p < 1.0 else None
        text = sample_text(
            model,
            stoi,
            itos,
            args.prompt,
            args.sample_tokens,
            args.window,
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
