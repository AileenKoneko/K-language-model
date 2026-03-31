from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str = "shakespeare"
    val_frac: float = 0.1
    data_path: str | None = None
    val_path: str | None = None
    tokenizer_type: str = "char"
    sp_model: str | None = None
    sp_vocab_size: int = 4096
    sp_model_type: str = "unigram"
    sp_character_coverage: float = 1.0
    sp_split_digits: bool = False
    sp_byte_fallback: bool = False
    allow_training_tokenizer: bool = True
    remap_by_frequency: bool = False


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    window: int
    d_model: int
    emb_dim: int | None
    rank: int
    n_k2: int
    k_base_rank: int
    k_base_impl: str
    share_k_base: bool
    emb_dropout: float
    mlp_dropout: float
    residual_dropout: float
    head_mode: str
    head_mult: int
    head_dropout: float
    adaptive_cutoffs: tuple[int, ...] = ()
    adaptive_div_value: float = 4.0
    alpha_cap: float = 1.0
    gamma_min: float = 0.85
    gamma_max: float = 1.0
    decay_impl: str = "mask"
    rosa_impl: str = "off"
    rosa_layers: str | None = "all"
    k_base_kernel_size: int = 8
