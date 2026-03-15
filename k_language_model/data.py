import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple
import zipfile

import torch

from .runtime import LOG

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TOKENIZER_DIR = DATA_DIR / "tokenizers"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_PATH = DATA_DIR / "input.txt"
WIKITEXT2_DIR = DATA_DIR / "wikitext-2"
WIKITEXT2_TRAIN_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
WIKITEXT2_VALID_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
WIKITEXT2_TRAIN_PATH = WIKITEXT2_DIR / "train.txt"
WIKITEXT2_VALID_PATH = WIKITEXT2_DIR / "valid.txt"
WIKITEXT2_RAW_DIR = DATA_DIR / "wikitext-2-raw"
WIKITEXT2_RAW_ARCHIVE_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
WIKITEXT2_RAW_ARCHIVE_PATH = DATA_DIR / "wikitext-2-raw-v1.zip"
WIKITEXT2_RAW_TRAIN_PATH = WIKITEXT2_RAW_DIR / "wiki.train.raw"
WIKITEXT2_RAW_VALID_PATH = WIKITEXT2_RAW_DIR / "wiki.valid.raw"


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
    def encode(self, text: str) -> List[int]:
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

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids if int(i) in self.itos)


class SentencePieceTokenizer(TextTokenizer):
    tokenizer_type = "sentencepiece"

    def __init__(self, model_path: Path):
        spm = _sentencepiece_module()
        model_path = model_path if model_path.suffix == ".model" else model_path.with_suffix(".model")
        if not model_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
        self.model_path = model_path
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))

    @property
    def vocab_size(self) -> int:
        return int(self.processor.vocab_size())

    def encode(self, text: str) -> List[int]:
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

    def encode(self, text: str) -> List[int]:
        return [self.old_to_new[idx] for idx in self.base.encode(text)]

    def decode(self, ids: List[int]) -> str:
        return self.base.decode([self.new_to_old[int(i)] for i in ids])

    def describe(self) -> str:
        return f"{self.base.describe()}+freq"


def _sentencepiece_module():
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            "SentencePiece support requires the 'sentencepiece' package. "
            "Install project dependencies again after pulling this change."
        ) from exc
    return spm


def _download_file(url: str, path: Path, label: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading %s to %s", label, path)
    urllib.request.urlretrieve(url, path)


def download_shakespeare() -> None:
    _download_file(SHAKESPEARE_URL, SHAKESPEARE_PATH, "Tiny Shakespeare")


def download_wikitext2() -> None:
    _download_file(WIKITEXT2_TRAIN_URL, WIKITEXT2_TRAIN_PATH, "WikiText-2 train split")
    _download_file(WIKITEXT2_VALID_URL, WIKITEXT2_VALID_PATH, "WikiText-2 validation split")


def _extract_zip_members(archive_path: Path, target_dir: Path, required_members: List[str], label: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        archive_names = {Path(name).name: name for name in zf.namelist()}
        missing = [name for name in required_members if name not in archive_names]
        if missing:
            raise FileNotFoundError(
                f"{label} archive {archive_path} is missing expected members: {', '.join(missing)}"
            )

        for member_name in required_members:
            target_path = target_dir / member_name
            if target_path.exists():
                continue
            target_path.write_bytes(zf.read(archive_names[member_name]))


def download_wikitext2_raw() -> None:
    required_members = [WIKITEXT2_RAW_TRAIN_PATH.name, WIKITEXT2_RAW_VALID_PATH.name]
    if all((WIKITEXT2_RAW_DIR / name).exists() for name in required_members):
        return
    _download_file(WIKITEXT2_RAW_ARCHIVE_URL, WIKITEXT2_RAW_ARCHIVE_PATH, "WikiText-2 raw archive")
    _extract_zip_members(
        archive_path=WIKITEXT2_RAW_ARCHIVE_PATH,
        target_dir=WIKITEXT2_RAW_DIR,
        required_members=required_members,
        label="WikiText-2 raw",
    )


def _read_text(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path.read_text(encoding="utf-8")


def _split_text(text: str, val_frac: float) -> Tuple[str, str]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")
    if len(text) < 2:
        raise ValueError("Dataset text must contain at least 2 characters for train/val split.")

    split = int(len(text) * (1.0 - val_frac))
    split = max(1, min(split, len(text) - 1))
    return text[:split], text[split:]


def _default_sentencepiece_model_path(dataset_key: str, data_path: str | None, vocab_size: int, model_type: str) -> Path:
    if data_path:
        base_name = Path(data_path).stem
    else:
        base_name = dataset_key.replace("-", "_")
    return TOKENIZER_DIR / f"{base_name}_{model_type}_{vocab_size}.model"


def _ensure_sentencepiece_model(
    model_path: Path,
    input_path: Path,
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

    spm = _sentencepiece_module()
    model_prefix = model_path.with_suffix("")
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    LOG.info(
        "Training SentencePiece model | input=%s | model=%s | vocab=%d | type=%s",
        input_path,
        model_path,
        vocab_size,
        model_type,
    )
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
    if not model_path.exists():
        raise FileNotFoundError(f"SentencePiece training did not produce expected model file: {model_path}")
    return model_path


def _build_tokenizer(
    tokenizer_type: str,
    dataset_key: str,
    train_path: Path,
    train_text: str,
    val_text: str,
    data_path: str | None,
    sp_model: str | None,
    sp_vocab_size: int,
    sp_model_type: str,
    sp_character_coverage: float,
    sp_split_digits: bool,
    sp_byte_fallback: bool,
    allow_training_tokenizer: bool,
) -> TextTokenizer:
    tokenizer_key = tokenizer_type.strip().lower()
    if tokenizer_key == "char":
        return CharTokenizer(train_text + val_text)
    if tokenizer_key != "sentencepiece":
        raise ValueError(f"Unknown tokenizer '{tokenizer_type}'. Expected one of: char, sentencepiece.")

    model_path = Path(sp_model) if sp_model else _default_sentencepiece_model_path(
        dataset_key=dataset_key,
        data_path=data_path,
        vocab_size=sp_vocab_size,
        model_type=sp_model_type,
    )
    model_path = _ensure_sentencepiece_model(
        model_path=model_path,
        input_path=train_path,
        vocab_size=sp_vocab_size,
        model_type=sp_model_type,
        character_coverage=sp_character_coverage,
        split_digits=sp_split_digits,
        byte_fallback=sp_byte_fallback,
        allow_training=allow_training_tokenizer,
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
    dataset_key = dataset.strip().lower()
    train_text = ""
    val_text = ""
    source_desc = ""

    if dataset_key == "shakespeare":
        train_path = Path(data_path) if data_path else SHAKESPEARE_PATH
        if data_path is None:
            download_shakespeare()
        train_text = _read_text(train_path, "Shakespeare train/source")

        if val_path:
            val_text = _read_text(Path(val_path), "Shakespeare validation")
            source_desc = f"train={train_path} | val={Path(val_path)}"
        else:
            train_text, val_text = _split_text(train_text, val_frac)
            source_desc = f"train/val split from {train_path} (val_frac={val_frac:.3f})"
    elif dataset_key in {"wikitext2", "wikitext-2", "wiki", "wikitext"}:
        if data_path:
            train_path = Path(data_path)
        else:
            download_wikitext2()
            train_path = WIKITEXT2_TRAIN_PATH
        train_text = _read_text(train_path, "WikiText-2 train/source")

        if val_path:
            val_text = _read_text(Path(val_path), "WikiText-2 validation")
            source_desc = f"train={train_path} | val={Path(val_path)}"
        elif data_path is None:
            val_text = _read_text(WIKITEXT2_VALID_PATH, "WikiText-2 validation")
            source_desc = f"train={train_path} | val={WIKITEXT2_VALID_PATH}"
        else:
            train_text, val_text = _split_text(train_text, val_frac)
            source_desc = f"train/val split from {train_path} (val_frac={val_frac:.3f})"
    elif dataset_key in {"wikitext2_raw", "wikitext-2-raw", "wiki_raw", "wikitext_raw"}:
        if data_path:
            train_path = Path(data_path)
        else:
            download_wikitext2_raw()
            train_path = WIKITEXT2_RAW_TRAIN_PATH
        train_text = _read_text(train_path, "WikiText-2 raw train/source")

        if val_path:
            val_text = _read_text(Path(val_path), "WikiText-2 raw validation")
            source_desc = f"train={train_path} | val={Path(val_path)}"
        elif data_path is None:
            val_text = _read_text(WIKITEXT2_RAW_VALID_PATH, "WikiText-2 raw validation")
            source_desc = f"train={train_path} | val={WIKITEXT2_RAW_VALID_PATH}"
        else:
            train_text, val_text = _split_text(train_text, val_frac)
            source_desc = f"train/val split from {train_path} (val_frac={val_frac:.3f})"
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Expected one of: shakespeare, wikitext2, wikitext2_raw."
        )

    tokenizer = _build_tokenizer(
        tokenizer_type=tokenizer_type,
        dataset_key=dataset_key,
        train_path=train_path,
        train_text=train_text,
        val_text=val_text,
        data_path=data_path,
        sp_model=sp_model,
        sp_vocab_size=sp_vocab_size,
        sp_model_type=sp_model_type,
        sp_character_coverage=sp_character_coverage,
        sp_split_digits=sp_split_digits,
        sp_byte_fallback=sp_byte_fallback,
        allow_training_tokenizer=allow_training_tokenizer,
    )

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    if remap_by_frequency:
        train_ids, val_ids, tokenizer = _remap_by_frequency(tokenizer, train_ids, val_ids)

    train_tensor = torch.tensor(train_ids, dtype=torch.long)
    val_tensor = torch.tensor(val_ids, dtype=torch.long)

    LOG.info(
        "Dataset ready | name=%s | source=%s | tokenizer=%s | vocab=%d | train_tokens=%d | val_tokens=%d",
        dataset_key,
        source_desc,
        tokenizer.describe(),
        tokenizer.vocab_size,
        len(train_tensor),
        len(val_tensor),
    )
    return train_tensor, val_tensor, tokenizer


def load_shakespeare(val_frac: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, TextTokenizer]:
    return load_dataset(dataset="shakespeare", val_frac=val_frac)


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
