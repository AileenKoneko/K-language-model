from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class TextTokenizer(ABC):
    tokenizer_type = "unknown"

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def is_character_level(self) -> bool:
        return False

    @abstractmethod
    def encode(self, text: str | bytes) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    def describe(self) -> str:
        return self.tokenizer_type


class CharTokenizer(TextTokenizer):
    tokenizer_type = "char"

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def is_character_level(self) -> bool:
        return True

    def encode(self, text: str | bytes) -> List[int]:
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids if int(i) in self.itos)


class ByteTokenizer(TextTokenizer):
    tokenizer_type = "byte"

    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str | bytes) -> List[int]:
        if isinstance(text, str):
            text = text.encode("utf-8")
        return list(text)

    def decode(self, ids: List[int]) -> str:
        raw = bytes(int(i) for i in ids if 0 <= int(i) < 256)
        return raw.decode("utf-8", errors="backslashreplace")


def sentencepiece_module():
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            "SentencePiece support requires the 'sentencepiece' package. "
            "Install project dependencies again after pulling this change."
        ) from exc
    return spm


class SentencePieceTokenizer(TextTokenizer):
    tokenizer_type = "sentencepiece"

    def __init__(self, model_path: Path):
        spm = sentencepiece_module()
        model_path = model_path if model_path.suffix == ".model" else model_path.with_suffix(".model")
        if not model_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
        self.model_path = model_path
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    def encode(self, text: str | bytes) -> List[int]:
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        return list(self.processor.encode(text, out_type=int))

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        return str(self.processor.decode([int(i) for i in ids]))

    def describe(self) -> str:
        return f"{self.tokenizer_type}:{self.model_path}"


class FrequencyRemappedTokenizer(TextTokenizer):
    tokenizer_type = "frequency-remap"

    def __init__(self, base: TextTokenizer, old_to_new: List[int], new_to_old: List[int]):
        if len(old_to_new) != base.vocab_size or len(new_to_old) != base.vocab_size:
            raise ValueError("Frequency remap tables must match tokenizer vocab size.")
        self.base = base
        self.old_to_new = old_to_new
        self.new_to_old = new_to_old

    @property
    def vocab_size(self) -> int:
        return self.base.vocab_size

    @property
    def is_character_level(self) -> bool:
        return self.base.is_character_level

    def encode(self, text: str | bytes) -> List[int]:
        return [self.old_to_new[idx] for idx in self.base.encode(text)]

    def decode(self, ids: List[int]) -> str:
        return self.base.decode([self.new_to_old[int(i)] for i in ids])

    def describe(self) -> str:
        return f"{self.base.describe()}+freq"
