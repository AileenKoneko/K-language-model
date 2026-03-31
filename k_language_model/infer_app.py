import argparse
import time
from pathlib import Path

import torch

from .checkpoint import load_model_checkpoint
from .cli_args import add_dataset_args, add_dynamics_args, add_model_args, add_repro_runtime_args, add_sampling_args
from .configs import DatasetConfig
from .data import load_dataset_bundle
from .generation import sample_text
from .model import resolve_adaptive_cutoffs
from .model_factory import build_model, model_config_from_args, parse_adaptive_cutoffs
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
from .trainer import ce_to_bpc, eval_deterministic


def _usable_eval_tokens(data: torch.Tensor, window: int) -> int:
    n_tokens = len(data) - 1
    if n_tokens <= 0:
        return 0
    return (n_tokens // window) * window



def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description or "Run inference/evaluation with a trained K-Stack checkpoint.")
    add_dataset_args(
        p,
        tokenizer_help="Tokenizer used to rebuild the training/eval token vocabulary.",
        sp_model_help="Existing SentencePiece model path used for the checkpoint.",
    )
    p.add_argument("--ckpt", "--checkpoint", dest="ckpt", type=str, required=True, help="Checkpoint path to load.")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size used for deterministic validation eval.")
    add_model_args(
        p,
        include_dropouts=False,
        adaptive_cutoffs_help="Comma-separated adaptive softmax cutoffs used by the checkpoint.",
    )
    add_dynamics_args(
        p,
        decay_help="Gamma-decay backend used by the checkpoint architecture (kernel is experimental Triton CUDA path).",
    )
    add_repro_runtime_args(
        p,
        include_compile=True,
        include_run_manifest=False,
        strict_repro_help="Force strict reproducibility (disables compile, enables deterministic mode and disables TF32).",
    )
    add_sampling_args(p, include_sample_flag=False, include_skip_flags=True)
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
        allow_training_tokenizer=False,
        remap_by_frequency=remap_by_frequency,
    )
    dataset_bundle = load_dataset_bundle(dataset_config)
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

    LOG.info("Model family | version=v2")
    model = build_model(
        model_config_from_args(
            args,
            vocab_size=vocab_size,
            adaptive_cutoffs=adaptive_cutoffs,
            emb_dropout=0.0,
            mlp_dropout=0.0,
            residual_dropout=0.0,
        )
    )

    if args.compile:
        model = maybe_enable_compile(model, enabled=True, mode=args.compile_mode)

    model = model.to(DEVICE)
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
            "Model config | tokenizer=%s | emb_dim=%d | d_model=%d | tied_weights=%s | head_mode=%s | k_base_rank=%d | k_base_impl=%s | share_k_base=%s | rosa_impl=%s | rosa_layers=%s | gamma[min/max]=%.3f/%.3f",
            tokenizer.describe(),
            core_model.emb_dim,
            core_model.d_model,
            str(getattr(core_model, "tie_weights", False)).lower(),
            args.head_mode,
            core_model.k_base_rank,
            core_model.k_base_impl,
            str(getattr(core_model, "share_k_base", False)).lower(),
            core_model.rosa_impl,
            core_model.describe_rosa_layers(),
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
                "Eval | step=%s | ckpt_best_ppl=%s | val_bpc=%.4f | val_ppl=%.2f | eval_tok_s=%.0f",
                loaded_step_str,
                loaded_best_ppl_str,
                ce_to_bpc(ce),
                ppl,
                eval_tok_s,
            )
        else:
            LOG.info(
                "Eval | step=%s | ckpt_best_ppl=%s | val_ce=%.4f | val_ppl=%.2f | eval_tok_s=%.0f",
                loaded_step_str,
                loaded_best_ppl_str,
                ce,
                ppl,
                eval_tok_s,
            )

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
