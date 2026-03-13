import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .runtime import LOG

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_PATH = DATA_DIR / "input.txt"


def download_shakespeare() -> None:
    if DATA_PATH.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading Tiny Shakespeare to %s", DATA_PATH)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def tokenize_char(text: str) -> Tuple[List[int], Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    ids = [stoi[ch] for ch in text]
    return ids, stoi, itos


def load_shakespeare(val_frac: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, int], Dict[int, str]]:
    download_shakespeare()
    text = DATA_PATH.read_text(encoding="utf-8")

    ids, stoi, itos = tokenize_char(text)
    split = int(len(ids) * (1.0 - val_frac))
    train_ids = torch.tensor(ids[:split], dtype=torch.long)
    val_ids = torch.tensor(ids[split:], dtype=torch.long)

    LOG.info(
        "Dataset ready | vocab=%d | train_tokens=%d | val_tokens=%d",
        len(stoi),
        len(train_ids),
        len(val_ids),
    )
    return train_ids, val_ids, len(stoi), stoi, itos


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
