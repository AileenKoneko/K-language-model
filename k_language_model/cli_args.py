from __future__ import annotations

import argparse

from .kbase import DEFAULT_K_BASE_KERNEL_SIZE


def add_dataset_args(parser: argparse.ArgumentParser, *, tokenizer_help: str, sp_model_help: str) -> None:
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["shakespeare", "full-shakespeare", "full-shakespeare-clean", "wikitext2", "wikitext2_raw"],
        default="shakespeare",
        help="Dataset preset to use.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional train/source text path. If omitted, built-in dataset files are used/downloaded.",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        help="Optional validation text path. If omitted, preset val split is used or val_frac split is applied.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation fraction used only when a separate validation file is not available/provided.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["char", "byte", "sentencepiece"],
        default="char",
        help=tokenizer_help,
    )
    parser.add_argument(
        "--sp-model",
        type=str,
        default=None,
        help=sp_model_help,
    )
    parser.add_argument("--sp-vocab-size", type=int, default=4096)
    parser.add_argument(
        "--sp-model-type",
        type=str,
        choices=["unigram", "bpe", "char", "word"],
        default="unigram",
    )
    parser.add_argument("--sp-character-coverage", type=float, default=1.0)
    parser.add_argument("--sp-split-digits", action="store_true")
    parser.add_argument("--sp-byte-fallback", action="store_true")


def add_model_args(parser: argparse.ArgumentParser, *, include_dropouts: bool, adaptive_cutoffs_help: str) -> None:
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=None,
        help="Optional token embedding dimension. Defaults to --d-model when omitted.",
    )
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument(
        "--k-base-rank",
        type=int,
        default=2,
        help="Reserved for future k_base variants. The current conv implementation ignores this knob.",
    )
    parser.add_argument(
        "--k-base-impl",
        type=str,
        choices=["conv", "auto", "fused", "scan"],
        default="conv",
        help="k_base implementation name. auto/fused/scan are currently aliases for conv.",
    )
    parser.add_argument(
        "--k-base-kernel-size",
        type=int,
        default=DEFAULT_K_BASE_KERNEL_SIZE,
        help="Local k_base lag width for the conv implementation.",
    )
    parser.add_argument(
        "--share-k-base",
        dest="share_k_base",
        action="store_true",
        help="Use one shared learnable k_base state across all K2 layers.",
    )
    parser.add_argument(
        "--no-share-k-base",
        dest="share_k_base",
        action="store_false",
        help="Use per-layer k_base state.",
    )
    parser.set_defaults(share_k_base=False)
    parser.add_argument("--n-k2", type=int, default=4)
    parser.add_argument("--head-mode", type=str, choices=["linear", "gelu", "adaptive"], default="linear")
    parser.add_argument("--head-mult", type=int, default=6)
    parser.add_argument("--head-dropout", type=float, default=0.10)
    parser.add_argument(
        "--adaptive-cutoffs",
        type=str,
        default=None,
        help=adaptive_cutoffs_help,
    )
    parser.add_argument("--adaptive-div-value", type=float, default=4.0)
    if include_dropouts:
        parser.add_argument("--emb-dropout", type=float, default=0.08)
        parser.add_argument("--mlp-dropout", type=float, default=0.10)
        parser.add_argument("--residual-dropout", type=float, default=0.05)


def add_dynamics_args(parser: argparse.ArgumentParser, *, decay_help: str) -> None:
    parser.add_argument(
        "--decay-impl",
        type=str,
        choices=["mask", "block", "kernel"],
        default="mask",
        help=decay_help,
    )
    parser.add_argument(
        "--rosa-impl",
        type=str,
        choices=["off", "exact", "gpu_approx", "auto"],
        default="off",
        help=(
            "ROSA backend: off (disable ROSA path), exact (CPU suffix-automaton reference), "
            "gpu_approx (experimental tensorized approximation; primarily for CUDA), "
            "auto (CUDA-only parity check, then choose gpu_approx or exact)."
        ),
    )
    parser.add_argument(
        "--rosa-layers",
        type=str,
        default="all",
        help="ROSA-enabled K2 layers: all, final, none, or comma-separated 1-based ids (for example: 4,5,6).",
    )
    parser.add_argument("--gamma-min", type=float, default=0.85, help="Lower bound for per-rank gamma decay values.")
    parser.add_argument("--gamma-max", type=float, default=1.0, help="Upper bound for per-rank gamma decay values.")
    parser.add_argument("--alpha-cap", type=float, default=0.8)


def add_repro_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    include_compile: bool,
    include_run_manifest: bool,
    strict_repro_help: str,
) -> None:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable strict deterministic algorithms (disables TF32 and cudnn benchmark).",
    )
    parser.add_argument(
        "--deterministic-warn-only",
        action="store_true",
        help="With --deterministic, warn instead of error on nondeterministic ops.",
    )
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 kernels on CUDA (slower, sometimes more numerically stable).",
    )
    parser.add_argument(
        "--strict-repro",
        action="store_true",
        help=strict_repro_help,
    )
    if include_run_manifest:
        parser.add_argument(
            "--run-manifest",
            type=str,
            default=None,
            help="Optional JSON path to store full run config/runtime metadata.",
        )
    if include_compile:
        parser.add_argument("--compile", action="store_true", help="Enable torch.compile.")
        parser.add_argument(
            "--compile-mode",
            type=str,
            choices=["default", "reduce-overhead", "max-autotune"],
            default="default",
        )


def add_sampling_args(
    parser: argparse.ArgumentParser,
    *,
    include_sample_flag: bool,
    include_skip_flags: bool,
    temperature_help: str | None = None,
    top_k_help: str | None = None,
    top_p_help: str | None = None,
    repetition_penalty_help: str | None = None,
    repetition_window_help: str | None = None,
    prompt_lock_help: str | None = None,
) -> None:
    if include_sample_flag:
        parser.add_argument("--sample", action="store_true", help="Generate a sample after training.")
    if include_skip_flags:
        parser.add_argument("--skip-eval", action="store_true", help="Skip deterministic validation CE/PPL evaluation.")
        parser.add_argument("--skip-sample", action="store_true", help="Skip text sampling.")
    parser.add_argument("--prompt", type=str, default="To be, or not to be")
    parser.add_argument("--sample-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=1.0, help=temperature_help)
    parser.add_argument("--top-k", type=int, default=0, help=top_k_help)
    parser.add_argument("--top-p", type=float, default=0.0, help=top_p_help)
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help=repetition_penalty_help)
    parser.add_argument("--repetition-window", type=int, default=256, help=repetition_window_help)
    parser.add_argument(
        "--prompt-lock-tokens",
        type=int,
        dest="prompt_lock_tokens",
        default=0,
        help=prompt_lock_help,
    )
    parser.add_argument("--prompt-lock-chars", type=int, dest="prompt_lock_tokens", help=argparse.SUPPRESS)
