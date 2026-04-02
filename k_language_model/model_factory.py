from __future__ import annotations

from argparse import Namespace

from .configs import ModelConfig
from .kstack import KStackModel
from .runtime import LOG


def parse_adaptive_cutoffs(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [part.strip() for part in raw.split(",")]
    parsed = [int(value) for value in values if value]
    return parsed or None


def parse_int_list(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return ()
    values = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value > 0:
            values.append(value)
    return tuple(sorted(set(values)))


def build_model(config: ModelConfig) -> KStackModel:
    if config.k_base_rank > 0:
        LOG.warning("k_base_rank is currently reserved on the V2 branch; the conv k_base implementation ignores it.")
    return KStackModel(
        vocab_size=config.vocab_size,
        window=config.window,
        d=config.d_model,
        emb_dim=config.emb_dim,
        rank=config.rank,
        n_k2=config.n_k2,
        k_base_rank=config.k_base_rank,
        k_base_impl=config.k_base_impl,
        share_k_base=config.share_k_base,
        emb_dropout=config.emb_dropout,
        mlp_dropout=config.mlp_dropout,
        residual_dropout=config.residual_dropout,
        head_mode=config.head_mode,
        head_mult=config.head_mult,
        head_dropout=config.head_dropout,
        future_summary_horizons=list(config.future_summary_horizons),
        adaptive_cutoffs=list(config.adaptive_cutoffs),
        adaptive_div_value=config.adaptive_div_value,
        alpha_cap=config.alpha_cap,
        gamma_min=config.gamma_min,
        gamma_max=config.gamma_max,
        decay_impl=config.decay_impl,
        rosa_impl=config.rosa_impl,
        rosa_layers=config.rosa_layers,
        k_base_kernel_size=config.k_base_kernel_size,
    )


def model_config_from_args(
    args: Namespace,
    *,
    vocab_size: int,
    adaptive_cutoffs: list[int] | None,
    emb_dropout: float,
    mlp_dropout: float,
    residual_dropout: float,
) -> ModelConfig:
    future_summary_horizons = parse_int_list(getattr(args, "future_summary_horizons", None))
    if not future_summary_horizons:
        single_horizon = int(getattr(args, "future_summary_horizon", 0) or 0)
        if single_horizon > 0:
            future_summary_horizons = (single_horizon,)
    return ModelConfig(
        future_summary_horizons=future_summary_horizons,
        vocab_size=vocab_size,
        window=args.window,
        d_model=args.d_model,
        emb_dim=args.emb_dim,
        rank=args.rank,
        n_k2=args.n_k2,
        k_base_rank=args.k_base_rank,
        k_base_impl=args.k_base_impl,
        share_k_base=args.share_k_base,
        emb_dropout=emb_dropout,
        mlp_dropout=mlp_dropout,
        residual_dropout=residual_dropout,
        head_mode=args.head_mode,
        head_mult=args.head_mult,
        head_dropout=args.head_dropout,
        adaptive_cutoffs=tuple(() if adaptive_cutoffs is None else adaptive_cutoffs),
        adaptive_div_value=args.adaptive_div_value,
        alpha_cap=args.alpha_cap,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        decay_impl=args.decay_impl,
        rosa_impl=args.rosa_impl,
        rosa_layers=args.rosa_layers,
        k_base_kernel_size=args.k_base_kernel_size,
    )
