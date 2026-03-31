from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

from .configs import DatasetConfig
from .dataset_loaders import DATA_DIR, build_dataset_loader, read_bytes, read_text
from .runtime import LOG
from .tokenizers import (
    ByteTokenizer,
    CharTokenizer,
    FrequencyRemappedTokenizer,
    SentencePieceTokenizer,
    TextTokenizer,
    sentencepiece_module,
)


TOKENIZER_DIR = DATA_DIR / "tokenizers"


@dataclass(frozen=True)
class DatasetBundle:
    train_data: torch.Tensor
    val_data: torch.Tensor
    tokenizer: TextTokenizer


def _split_text(text: str, val_frac: float) -> Tuple[str, str]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")
    if len(text) < 2:
        raise ValueError("Dataset text must contain at least 2 characters for train/val split.")

    split = int(len(text) * (1.0 - val_frac))
    split = max(1, min(split, len(text) - 1))
    return text[:split], text[split:]


def _split_bytes(data: bytes, val_frac: float) -> Tuple[bytes, bytes]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")
    if len(data) < 2:
        raise ValueError("Dataset bytes must contain at least 2 bytes for train/val split.")

    split = int(len(data) * (1.0 - val_frac))
    split = max(1, min(split, len(data) - 1))
    return data[:split], data[split:]


def _default_sentencepiece_model_path(dataset_key: str, data_path: str | None, vocab_size: int, model_type: str) -> Path:
    if data_path:
        base_name = Path(data_path).stem
    else:
        base_name = dataset_key.replace("-", "_")
    return TOKENIZER_DIR / f"{base_name}_{model_type}_{vocab_size}.model"


def _ensure_sentencepiece_model(
    model_path: Path,
    train_text: str,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
    split_digits: bool,
    byte_fallback: bool,
    allow_training: bool,
) -> Path:
    model_path = model_path if model_path.suffix == ".model" else model_path.with_suffix(".model")
    if model_path.exists():
        return model_path
    if not allow_training:
        raise FileNotFoundError(
            f"SentencePiece model not found: {model_path}. "
            "Provide an existing --sp-model that matches the checkpoint/tokenizer used for training."
        )

    spm = sentencepiece_module()
    model_prefix = model_path.with_suffix("")
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(train_text)
        input_path = Path(handle.name)

    LOG.info(
        "Training SentencePiece model | input=%s | model=%s | vocab=%d | type=%s",
        input_path,
        model_path,
        vocab_size,
        model_type,
    )
    try:
        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=str(model_prefix),
            vocab_size=int(vocab_size),
            model_type=str(model_type),
            character_coverage=float(character_coverage),
            split_digits=bool(split_digits),
            byte_fallback=bool(byte_fallback),
            hard_vocab_limit=False,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            unk_id=0,
        )
    finally:
        input_path.unlink(missing_ok=True)
    if not model_path.exists():
        raise FileNotFoundError(f"SentencePiece training did not produce expected model file: {model_path}")
    return model_path


def _build_tokenizer(config: DatasetConfig, *, train_text: str | None, val_text: str | None) -> TextTokenizer:
    tokenizer_key = config.tokenizer_type.strip().lower()
    if tokenizer_key == "char":
        if train_text is None or val_text is None:
            raise ValueError("Char tokenizer requires decoded text inputs.")
        return CharTokenizer(train_text + val_text)
    if tokenizer_key == "byte":
        return ByteTokenizer()
    if tokenizer_key != "sentencepiece":
        raise ValueError(f"Unknown tokenizer '{config.tokenizer_type}'. Expected one of: char, byte, sentencepiece.")
    if train_text is None:
        raise ValueError("SentencePiece tokenizer requires decoded text inputs.")

    model_path = Path(config.sp_model) if config.sp_model else _default_sentencepiece_model_path(
        dataset_key=config.dataset.strip().lower(),
        data_path=config.data_path,
        vocab_size=config.sp_vocab_size,
        model_type=config.sp_model_type,
    )
    model_path = _ensure_sentencepiece_model(
        model_path=model_path,
        train_text=train_text,
        vocab_size=config.sp_vocab_size,
        model_type=config.sp_model_type,
        character_coverage=config.sp_character_coverage,
        split_digits=config.sp_split_digits,
        byte_fallback=config.sp_byte_fallback,
        allow_training=config.allow_training_tokenizer,
    )
    return SentencePieceTokenizer(model_path)


def _remap_by_frequency(
    tokenizer: TextTokenizer,
    train_ids: List[int],
    val_ids: List[int],
) -> Tuple[List[int], List[int], TextTokenizer]:
    if tokenizer.vocab_size <= 1:
        return train_ids, val_ids, tokenizer

    counts = [0 for _ in range(tokenizer.vocab_size)]
    for idx in train_ids:
        counts[int(idx)] += 1

    new_to_old = sorted(range(tokenizer.vocab_size), key=lambda idx: (-counts[idx], idx))
    old_to_new = [0 for _ in range(tokenizer.vocab_size)]
    for new_idx, old_idx in enumerate(new_to_old):
        old_to_new[old_idx] = new_idx

    remapped_train = [old_to_new[idx] for idx in train_ids]
    remapped_val = [old_to_new[idx] for idx in val_ids]
    return remapped_train, remapped_val, FrequencyRemappedTokenizer(tokenizer, old_to_new, new_to_old)


def load_dataset_bundle(config: DatasetConfig) -> DatasetBundle:
    dataset_key = config.dataset.strip().lower()
    tokenizer_key = config.tokenizer_type.strip().lower()
    source_reader = read_bytes if tokenizer_key == "byte" else read_text
    source_splitter = _split_bytes if tokenizer_key == "byte" else _split_text

    sources = build_dataset_loader(dataset_key).load(
        data_path=config.data_path,
        val_path=config.val_path,
        val_frac=config.val_frac,
        source_reader=source_reader,
        source_splitter=source_splitter,
    )

    train_text = sources.train_source if isinstance(sources.train_source, str) else None
    val_text = sources.val_source if isinstance(sources.val_source, str) else None
    tokenizer = _build_tokenizer(config, train_text=train_text, val_text=val_text)

    train_ids = tokenizer.encode(sources.train_source)
    val_ids = tokenizer.encode(sources.val_source)
    if config.remap_by_frequency:
        train_ids, val_ids, tokenizer = _remap_by_frequency(tokenizer, train_ids, val_ids)

    train_tensor = torch.tensor(train_ids, dtype=torch.long)
    val_tensor = torch.tensor(val_ids, dtype=torch.long)

    LOG.info(
        "Dataset ready | name=%s | source=%s | tokenizer=%s | vocab=%d | train_tokens=%d | val_tokens=%d",
        dataset_key,
        sources.source_desc,
        tokenizer.describe(),
        tokenizer.vocab_size,
        len(train_tensor),
        len(val_tensor),
    )
    return DatasetBundle(train_data=train_tensor, val_data=val_tensor, tokenizer=tokenizer)


def get_batch(data: torch.Tensor, window: int, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= window + 1:
        raise ValueError(f"Data length ({len(data)}) must be > window+1 ({window+1}).")

    src_device = data.device
    ix = torch.randint(len(data) - window - 1, (batch_size,), device=src_device)
    offsets = torch.arange(window, device=src_device)
    pos = ix.unsqueeze(1) + offsets.unsqueeze(0)
    flat = pos.reshape(-1)

    x = data.index_select(0, flat).view(batch_size, window)
    y = data.index_select(0, flat + 1).view(batch_size, window)

    dst_device = torch.device(device)
    if x.device != dst_device:
        x = x.to(dst_device, non_blocking=True)
        y = y.to(dst_device, non_blocking=True)
    return x, y
