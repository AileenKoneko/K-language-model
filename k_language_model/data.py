from __future__ import annotations

from typing import Tuple

import torch

from .configs import DatasetConfig
from .dataset_loaders import (
    DATA_DIR,
    FULL_SHAKESPEARE_CLEAN_PATH,
    FULL_SHAKESPEARE_CLEAN_VERSION,
    FULL_SHAKESPEARE_DIR,
    FULL_SHAKESPEARE_PATH,
    FULL_SHAKESPEARE_RAW_DIR,
    SHAKESPEARE_PATH,
    WIKITEXT2_DIR,
    WIKITEXT2_RAW_DIR,
    WIKITEXT2_RAW_TRAIN_PATH,
    WIKITEXT2_RAW_VALID_PATH,
    WIKITEXT2_TRAIN_PATH,
    WIKITEXT2_VALID_PATH,
    download_shakespeare,
    download_wikitext2,
    download_wikitext2_raw,
    read_bytes,
    read_text,
)
from .dataset_pipeline import DatasetBundle, get_batch, load_dataset_bundle
from .tokenizers import ByteTokenizer, CharTokenizer, FrequencyRemappedTokenizer, SentencePieceTokenizer, TextTokenizer


def load_dataset(
    dataset: str = "shakespeare",
    val_frac: float = 0.1,
    data_path: str | None = None,
    val_path: str | None = None,
    tokenizer_type: str = "char",
    sp_model: str | None = None,
    sp_vocab_size: int = 4096,
    sp_model_type: str = "unigram",
    sp_character_coverage: float = 1.0,
    sp_split_digits: bool = False,
    sp_byte_fallback: bool = False,
    allow_training_tokenizer: bool = True,
    remap_by_frequency: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, TextTokenizer]:
    bundle = load_dataset_bundle(
        DatasetConfig(
            dataset=dataset,
            val_frac=val_frac,
            data_path=data_path,
            val_path=val_path,
            tokenizer_type=tokenizer_type,
            sp_model=sp_model,
            sp_vocab_size=sp_vocab_size,
            sp_model_type=sp_model_type,
            sp_character_coverage=sp_character_coverage,
            sp_split_digits=sp_split_digits,
            sp_byte_fallback=sp_byte_fallback,
            allow_training_tokenizer=allow_training_tokenizer,
            remap_by_frequency=remap_by_frequency,
        )
    )
    return bundle.train_data, bundle.val_data, bundle.tokenizer


def load_shakespeare(val_frac: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, TextTokenizer]:
    return load_dataset(dataset="shakespeare", val_frac=val_frac)


def load_full_shakespeare(val_frac: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, TextTokenizer]:
    return load_dataset(dataset="full-shakespeare", val_frac=val_frac)


def load_full_shakespeare_clean(val_frac: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, TextTokenizer]:
    return load_dataset(dataset="full-shakespeare-clean", val_frac=val_frac)


__all__ = [
    "ByteTokenizer",
    "CharTokenizer",
    "DATA_DIR",
    "DatasetBundle",
    "FULL_SHAKESPEARE_CLEAN_PATH",
    "FULL_SHAKESPEARE_CLEAN_VERSION",
    "FULL_SHAKESPEARE_DIR",
    "FULL_SHAKESPEARE_PATH",
    "FULL_SHAKESPEARE_RAW_DIR",
    "FrequencyRemappedTokenizer",
    "SHAKESPEARE_PATH",
    "SentencePieceTokenizer",
    "TextTokenizer",
    "WIKITEXT2_DIR",
    "WIKITEXT2_RAW_DIR",
    "WIKITEXT2_RAW_TRAIN_PATH",
    "WIKITEXT2_RAW_VALID_PATH",
    "WIKITEXT2_TRAIN_PATH",
    "WIKITEXT2_VALID_PATH",
    "download_shakespeare",
    "download_wikitext2",
    "download_wikitext2_raw",
    "get_batch",
    "load_dataset",
    "load_dataset_bundle",
    "load_full_shakespeare",
    "load_full_shakespeare_clean",
    "load_shakespeare",
    "read_bytes",
    "read_text",
]
