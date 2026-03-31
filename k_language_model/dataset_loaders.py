from __future__ import annotations

import re
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .runtime import LOG


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_PATH = DATA_DIR / "input.txt"
FULL_SHAKESPEARE_DIR = DATA_DIR / "full-shakespeare"
FULL_SHAKESPEARE_RAW_DIR = FULL_SHAKESPEARE_DIR / "raw"
FULL_SHAKESPEARE_PATH = FULL_SHAKESPEARE_DIR / "full-shakespeare.txt"
FULL_SHAKESPEARE_CLEAN_VERSION = 1
FULL_SHAKESPEARE_CLEAN_PATH = FULL_SHAKESPEARE_DIR / f"full-shakespeare.cleaned.v{FULL_SHAKESPEARE_CLEAN_VERSION}.txt"
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

_FOLGER_CREATED_RE = re.compile(r"^Created on .+ from FDT version", re.IGNORECASE)
_SECTION_UNDERLINE_RE = re.compile(r"^[=\-]{3,}$")


def download_shakespeare() -> None:
    _download_file(SHAKESPEARE_URL, SHAKESPEARE_PATH, "Tiny Shakespeare")


def download_wikitext2() -> None:
    _download_file(WIKITEXT2_TRAIN_URL, WIKITEXT2_TRAIN_PATH, "WikiText-2 train split")
    _download_file(WIKITEXT2_VALID_URL, WIKITEXT2_VALID_PATH, "WikiText-2 validation split")


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


@dataclass(frozen=True)
class DatasetSources:
    train_path: Path
    train_source: str | bytes
    val_source: str | bytes
    source_desc: str


class DatasetLoader(ABC):
    name = ""

    @abstractmethod
    def load(
        self,
        *,
        data_path: str | None,
        val_path: str | None,
        val_frac: float,
        source_reader,
        source_splitter,
    ) -> DatasetSources:
        raise NotImplementedError


_DATASET_LOADERS: Dict[str, type[DatasetLoader]] = {}


def register_dataset_loader(cls: type[DatasetLoader]) -> type[DatasetLoader]:
    if not cls.name:
        raise ValueError("DatasetLoader subclasses must define a non-empty name.")
    _DATASET_LOADERS[cls.name] = cls
    return cls


def build_dataset_loader(dataset: str) -> DatasetLoader:
    key = dataset.strip().lower()
    aliases = {
        "full_shakespeare": "full-shakespeare",
        "fullshakespeare": "full-shakespeare",
        "full_shakespeare_clean": "full-shakespeare-clean",
        "fullshakespeareclean": "full-shakespeare-clean",
        "wikitext-2": "wikitext2",
        "wiki": "wikitext2",
        "wikitext": "wikitext2",
        "wikitext-2-raw": "wikitext2_raw",
        "wiki_raw": "wikitext2_raw",
        "wikitext_raw": "wikitext2_raw",
    }
    resolved = aliases.get(key, key)
    try:
        return _DATASET_LOADERS[resolved]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Expected one of: shakespeare, full-shakespeare, full-shakespeare-clean, wikitext2, wikitext2_raw."
        ) from exc


@register_dataset_loader
class ShakespeareDatasetLoader(DatasetLoader):
    name = "shakespeare"

    def load(self, *, data_path, val_path, val_frac, source_reader, source_splitter) -> DatasetSources:
        train_path = Path(data_path) if data_path else SHAKESPEARE_PATH
        if data_path is None:
            download_shakespeare()
        train_source = source_reader(train_path, "Shakespeare train/source")
        if val_path:
            val_source = source_reader(Path(val_path), "Shakespeare validation")
            source_desc = f"train={train_path} | val={Path(val_path)}"
        else:
            train_source, val_source = source_splitter(train_source, val_frac)
            source_desc = f"train/val split from {train_path} (val_frac={val_frac:.3f})"
        return DatasetSources(train_path=train_path, train_source=train_source, val_source=val_source, source_desc=source_desc)


@register_dataset_loader
class FullShakespeareDatasetLoader(DatasetLoader):
    name = "full-shakespeare"

    def load(self, *, data_path, val_path, val_frac, source_reader, source_splitter) -> DatasetSources:
        if data_path:
            raw_or_merged_path = Path(data_path)
            train_path = _materialize_merged_corpus(raw_or_merged_path) if raw_or_merged_path.is_dir() else raw_or_merged_path
        else:
            train_path = _materialize_merged_corpus(FULL_SHAKESPEARE_RAW_DIR, FULL_SHAKESPEARE_PATH)
        train_source = source_reader(train_path, "Full Shakespeare train/source")
        if val_path:
            val_source = source_reader(Path(val_path), "Full Shakespeare validation")
            source_desc = f"train={train_path} | val={Path(val_path)}"
        else:
            train_source, val_source = source_splitter(train_source, val_frac)
            source_desc = f"train/val split from {train_path} (val_frac={val_frac:.3f})"
        return DatasetSources(train_path=train_path, train_source=train_source, val_source=val_source, source_desc=source_desc)


@register_dataset_loader
class FullShakespeareCleanDatasetLoader(DatasetLoader):
    name = "full-shakespeare-clean"

    def load(self, *, data_path, val_path, val_frac, source_reader, source_splitter) -> DatasetSources:
        if data_path:
            raw_or_clean_path = Path(data_path)
            train_path = _materialize_cleaned_full_shakespeare_corpus(raw_or_clean_path) if raw_or_clean_path.is_dir() else raw_or_clean_path
        else:
            train_path = _materialize_cleaned_full_shakespeare_corpus(FULL_SHAKESPEARE_RAW_DIR, FULL_SHAKESPEARE_CLEAN_PATH)
        train_source = source_reader(train_path, "Full Shakespeare clean train/source")
        if val_path:
            val_source = source_reader(Path(val_path), "Full Shakespeare clean validation")
            source_desc = f"train={train_path} | val={Path(val_path)}"
        else:
            train_source, val_source = source_splitter(train_source, val_frac)
            source_desc = f"train/val split from {train_path} (val_frac={val_frac:.3f})"
        return DatasetSources(train_path=train_path, train_source=train_source, val_source=val_source, source_desc=source_desc)


@register_dataset_loader
class WikiText2DatasetLoader(DatasetLoader):
    name = "wikitext2"

    def load(self, *, data_path, val_path, val_frac, source_reader, source_splitter) -> DatasetSources:
        if data_path:
            train_path = Path(data_path)
        else:
            download_wikitext2()
            train_path = WIKITEXT2_TRAIN_PATH
        train_source = source_reader(train_path, "WikiText-2 train/source")
        if val_path:
            val_source = source_reader(Path(val_path), "WikiText-2 validation")
            source_desc = f"train={train_path} | val={Path(val_path)}"
        elif data_path is None:
            val_source = source_reader(WIKITEXT2_VALID_PATH, "WikiText-2 validation")
            source_desc = f"train={train_path} | val={WIKITEXT2_VALID_PATH}"
        else:
            train_source, val_source = source_splitter(train_source, val_frac)
            source_desc = f"train/val split from {train_path} (val_frac={val_frac:.3f})"
        return DatasetSources(train_path=train_path, train_source=train_source, val_source=val_source, source_desc=source_desc)


@register_dataset_loader
class WikiText2RawDatasetLoader(DatasetLoader):
    name = "wikitext2_raw"

    def load(self, *, data_path, val_path, val_frac, source_reader, source_splitter) -> DatasetSources:
        if data_path:
            train_path = Path(data_path)
        else:
            download_wikitext2_raw()
            train_path = WIKITEXT2_RAW_TRAIN_PATH
        train_source = source_reader(train_path, "WikiText-2 raw train/source")
        if val_path:
            val_source = source_reader(Path(val_path), "WikiText-2 raw validation")
            source_desc = f"train={train_path} | val={Path(val_path)}"
        elif data_path is None:
            val_source = source_reader(WIKITEXT2_RAW_VALID_PATH, "WikiText-2 raw validation")
            source_desc = f"train={train_path} | val={WIKITEXT2_RAW_VALID_PATH}"
        else:
            train_source, val_source = source_splitter(train_source, val_frac)
            source_desc = f"train/val split from {train_path} (val_frac={val_frac:.3f})"
        return DatasetSources(train_path=train_path, train_source=train_source, val_source=val_source, source_desc=source_desc)


def _download_file(url: str, path: Path, label: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading %s to %s", label, path)
    urllib.request.urlretrieve(url, path)


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


def read_text(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path.read_text(encoding="utf-8")


def read_bytes(path: Path, label: str) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return path.read_bytes()


def sorted_text_files(directory: Path) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Corpus directory not found: {directory}")
    files = sorted(path for path in directory.glob("*.txt") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .txt files found in corpus directory: {directory}")
    return files


def default_merged_corpus_path(raw_dir: Path) -> Path:
    if raw_dir.resolve() == FULL_SHAKESPEARE_RAW_DIR.resolve():
        return FULL_SHAKESPEARE_PATH
    return raw_dir.parent / f"{raw_dir.name}.merged.txt"


def default_cleaned_corpus_path(raw_dir: Path) -> Path:
    if raw_dir.resolve() == FULL_SHAKESPEARE_RAW_DIR.resolve():
        return FULL_SHAKESPEARE_CLEAN_PATH
    return raw_dir.parent / f"{raw_dir.name}.cleaned.v{FULL_SHAKESPEARE_CLEAN_VERSION}.txt"


def _materialize_merged_corpus(raw_dir: Path, merged_path: Path | None = None) -> Path:
    raw_dir = raw_dir.resolve()
    merged_path = (merged_path or default_merged_corpus_path(raw_dir)).resolve()
    files = sorted_text_files(raw_dir)
    newest_source_mtime = max(path.stat().st_mtime for path in files)
    if merged_path.exists() and merged_path.stat().st_mtime >= newest_source_mtime:
        return merged_path

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = merged_path.with_suffix(merged_path.suffix + ".tmp")
    needs_separator = False
    with tmp_path.open("wb") as handle:
        for path in files:
            data = path.read_bytes()
            if needs_separator and data:
                handle.write(b"\n")
            handle.write(data)
            if data:
                needs_separator = not data.endswith((b"\n", b"\r"))
    tmp_path.replace(merged_path)
    return merged_path


def _normalize_title_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _trim_and_compact_blank_lines(lines: List[str]) -> List[str]:
    trimmed = [line.rstrip() for line in lines]
    while trimmed and not trimmed[0].strip():
        trimmed.pop(0)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    compacted: List[str] = []
    blank_run = 0
    for line in trimmed:
        if line.strip():
            blank_run = 0
            compacted.append(line)
        else:
            blank_run += 1
            if blank_run <= 2:
                compacted.append("")
    return compacted


def clean_folger_work(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    title = next((line.strip() for line in lines if line.strip()), "")

    created_idx = next((idx for idx, line in enumerate(lines) if _FOLGER_CREATED_RE.match(line.strip())), None)
    if created_idx is not None:
        lines = lines[created_idx + 1 :]
    lines = _trim_and_compact_blank_lines(lines)

    body_start = 0
    if "sonnet" in title.lower():
        sonnet_idx = next((idx for idx, line in enumerate(lines) if re.fullmatch(r"\d+", line.strip())), None)
        if sonnet_idx is not None:
            body_start = sonnet_idx
    elif any(line.strip() == "Characters in the Play" for line in lines):
        cast_idx = next(idx for idx, line in enumerate(lines) if line.strip() == "Characters in the Play")
        body_start = cast_idx + 1
        while body_start < len(lines) and lines[body_start].strip():
            body_start += 1
        while body_start < len(lines) and not lines[body_start].strip():
            body_start += 1
    elif title:
        normalized_title = _normalize_title_text(title)
        repeated_title_idx = next(
            (
                idx
                for idx, line in enumerate(lines)
                if line.strip() and _normalize_title_text(line.strip().strip('"')) == normalized_title
            ),
            None,
        )
        if repeated_title_idx is not None:
            body_start = repeated_title_idx + 1
            while body_start < len(lines) and not lines[body_start].strip():
                body_start += 1

    cleaned_lines = [line for line in lines[body_start:] if not _SECTION_UNDERLINE_RE.fullmatch(line.strip())]
    cleaned_lines = _trim_and_compact_blank_lines(cleaned_lines)
    if not cleaned_lines:
        raise ValueError("Folger text cleaning produced an empty document.")
    return "\n".join(cleaned_lines) + "\n"


def _materialize_cleaned_full_shakespeare_corpus(raw_dir: Path, cleaned_path: Path | None = None) -> Path:
    raw_dir = raw_dir.resolve()
    cleaned_path = (cleaned_path or default_cleaned_corpus_path(raw_dir)).resolve()
    files = sorted_text_files(raw_dir)
    newest_source_mtime = max(path.stat().st_mtime for path in files)
    if cleaned_path.exists() and cleaned_path.stat().st_mtime >= newest_source_mtime:
        return cleaned_path

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cleaned_path.with_suffix(cleaned_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for idx, path in enumerate(files):
            cleaned = clean_folger_work(path.read_text(encoding="utf-8"))
            if idx > 0:
                handle.write("\n")
            handle.write(cleaned)
    tmp_path.replace(cleaned_path)
    return cleaned_path
