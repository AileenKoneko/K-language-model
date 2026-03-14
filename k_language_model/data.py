import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .runtime import LOG

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_PATH = DATA_DIR / "input.txt"
WIKITEXT2_DIR = DATA_DIR / "wikitext-2"
WIKITEXT2_TRAIN_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
WIKITEXT2_VALID_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
WIKITEXT2_TRAIN_PATH = WIKITEXT2_DIR / "train.txt"
WIKITEXT2_VALID_PATH = WIKITEXT2_DIR / "valid.txt"


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


def tokenize_char(text: str) -> Tuple[List[int], Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    ids = [stoi[ch] for ch in text]
    return ids, stoi, itos


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


def load_dataset(
    dataset: str = "shakespeare",
    val_frac: float = 0.1,
    data_path: str | None = None,
    val_path: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, int], Dict[int, str]]:
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
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Expected one of: shakespeare, wikitext2."
        )

    _, stoi, itos = tokenize_char(train_text + val_text)
    train_ids = torch.tensor([stoi[ch] for ch in train_text], dtype=torch.long)
    val_ids = torch.tensor([stoi[ch] for ch in val_text], dtype=torch.long)

    LOG.info(
        "Dataset ready | name=%s | source=%s | vocab=%d | train_tokens=%d | val_tokens=%d",
        dataset_key,
        source_desc,
        len(stoi),
        len(train_ids),
        len(val_ids),
    )
    return train_ids, val_ids, len(stoi), stoi, itos


def load_shakespeare(val_frac: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, int], Dict[int, str]]:
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
