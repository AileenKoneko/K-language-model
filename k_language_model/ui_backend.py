from __future__ import annotations

import contextlib
import io
import os
import re
import shlex
import site
import subprocess
import sys
import sysconfig
import threading
import time
import uuid
from argparse import ArgumentParser
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .infer_app import build_parser as build_infer_parser
from .train_app import build_parser as build_train_parser


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_TAIL_LINES = 400
EVENT_TAIL_LINES = 10
_LOG_LINE_RE = re.compile(r"^(?P<time>\d{2}:\d{2}:\d{2}) \| (?P<level>[A-Z]+) \| (?P<message>.*)$")
_STEP_PREFIX_RE = re.compile(r"^step=\s*(?P<step>\d+)")
_PLAIN_KEY_VALUE_RE = re.compile(r"^(?P<key>[a-zA-Z0-9_./-]+)=(?P<value>.+)$")


@dataclass(frozen=True)
class UiField:
    name: str
    label: str
    kind: str
    arg: str | None = None
    default: Any = None
    choices: tuple[str, ...] = ()
    required: bool = False
    placeholder: str = ""
    help_text: str = ""
    step: str | None = None
    min_value: str | None = None
    rows: int = 4
    true_flag: str | None = None
    false_flag: str | None = None
    always_include: bool = False
    datalist_id: str | None = None


@dataclass(frozen=True)
class UiSection:
    title: str
    fields: tuple[UiField, ...]


@dataclass(frozen=True)
class PreparedRun:
    mode: str
    name: str
    args: list[str]
    command: list[str]
    command_display: str


@dataclass
class JobRecord:
    job_id: str
    mode: str
    name: str
    args: list[str]
    command: list[str]
    command_display: str
    created_at: float
    log_path: Path
    status: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    pid: int | None = None
    cancel_requested: bool = False
    error: str | None = None
    log_line_count: int = 0
    log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=LOG_TAIL_LINES))
    recent_events: deque[str] = field(default_factory=lambda: deque(maxlen=EVENT_TAIL_LINES))
    phase: str = "queued"
    current_step: int | None = None
    total_steps: int | None = None
    summary_stats: dict[str, str] = field(default_factory=dict)
    runtime_info: dict[str, str] = field(default_factory=dict)
    warning_count: int = 0
    last_warning: str | None = None
    last_message: str | None = None
    process: subprocess.Popen[str] | None = field(default=None, repr=False)


def _extract_parser_meta(parser: ArgumentParser) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        meta[action.dest] = {
            "default": action.default,
            "choices": tuple(action.choices) if action.choices is not None else (),
        }
    return meta


_PARSER_BY_MODE = {
    "train": build_train_parser("Schedule a K-Stack training run."),
    "infer": build_infer_parser("Schedule a K-Stack inference/eval run."),
}
_PARSER_META_BY_MODE = {mode: _extract_parser_meta(parser) for mode, parser in _PARSER_BY_MODE.items()}


def _meta(mode: str, dest: str) -> dict[str, Any]:
    return _PARSER_META_BY_MODE[mode][dest]


def _option_field(
    mode: str,
    dest: str,
    label: str,
    *,
    kind: str = "text",
    required: bool = False,
    placeholder: str = "",
    help_text: str = "",
    step: str | None = None,
    min_value: str | None = None,
    rows: int = 4,
    true_flag: str | None = None,
    false_flag: str | None = None,
    always_include: bool = False,
    datalist_id: str | None = None,
) -> UiField:
    meta = _meta(mode, dest)
    return UiField(
        name=dest,
        label=label,
        kind=kind,
        arg=f"--{dest.replace('_', '-')}" if kind != "checkbox" else None,
        default=meta["default"],
        choices=meta["choices"],
        required=required,
        placeholder=placeholder,
        help_text=help_text,
        step=step,
        min_value=min_value,
        rows=rows,
        true_flag=true_flag,
        false_flag=false_flag,
        always_include=always_include,
        datalist_id=datalist_id,
    )


def _free_field(
    name: str,
    label: str,
    *,
    kind: str = "text",
    required: bool = False,
    placeholder: str = "",
    help_text: str = "",
    rows: int = 4,
) -> UiField:
    return UiField(
        name=name,
        label=label,
        kind=kind,
        required=required,
        placeholder=placeholder,
        help_text=help_text,
        rows=rows,
    )


TRAIN_FORM_SECTIONS: tuple[UiSection, ...] = (
    UiSection(
        title="Run",
        fields=(
            _free_field("job_name", "Run Name", placeholder="byte-shakespeare-sweep"),
            _option_field("train", "dataset", "Dataset", kind="select"),
            _option_field("train", "tokenizer", "Tokenizer", kind="select"),
            _option_field("train", "sp_model", "sp_model", placeholder="data/tokenizers/model.model"),
            _option_field("train", "sp_vocab_size", "sp_vocab_size", kind="number", min_value="1"),
            _option_field("train", "sp_model_type", "sp_model_type", kind="select"),
            _option_field(
                "train", "sp_character_coverage", "sp_character_coverage", kind="number", step="0.0001", min_value="0"
            ),
            _option_field("train", "sp_split_digits", "sp_split_digits", kind="checkbox", true_flag="--sp-split-digits"),
            _option_field(
                "train", "sp_byte_fallback", "sp_byte_fallback", kind="checkbox", true_flag="--sp-byte-fallback"
            ),
            _option_field(
                "train",
                "ckpt",
                "Checkpoint Path",
                placeholder="models/byte_shakespeare_v2.pt",
                help_text="Optional output checkpoint path.",
                datalist_id="checkpoint-list",
            ),
            _option_field("train", "data_path", "Train Data Path", placeholder="data/my_corpus.txt"),
            _option_field("train", "val_path", "Validation Path", placeholder="data/my_corpus_val.txt"),
            _option_field("train", "val_frac", "Val Fraction", kind="number", step="0.01", min_value="0"),
            _option_field("train", "steps", "Steps", kind="number", min_value="1"),
            _option_field("train", "batch_size", "Batch Size", kind="number", min_value="1"),
            _option_field("train", "window", "Window", kind="number", min_value="1"),
            _option_field("train", "seed", "Seed", kind="number", min_value="0"),
        ),
    ),
    UiSection(
        title="Model",
        fields=(
            _option_field("train", "d_model", "d_model", kind="number", min_value="1"),
            _option_field("train", "emb_dim", "emb_dim", kind="number", min_value="1", placeholder="defaults to d_model"),
            _option_field("train", "rank", "Rank", kind="number", min_value="1"),
            _option_field("train", "n_k2", "K2 Layers", kind="number", min_value="1"),
            _option_field("train", "k_base_rank", "k_base_rank", kind="number"),
            _option_field("train", "k_base_impl", "k_base_impl", kind="select"),
            _option_field("train", "k_base_kernel_size", "k_base_kernel_size", kind="number", min_value="1"),
            _option_field(
                "train",
                "share_k_base",
                "share_k_base",
                kind="checkbox",
                help_text="Enable one shared k_base across all K2 layers.",
                true_flag="--share-k-base",
            ),
            _option_field("train", "head_mode", "Head Mode", kind="select"),
            _option_field("train", "head_mult", "Head Mult", kind="number", min_value="1"),
            _option_field("train", "head_dropout", "head_dropout", kind="number", step="0.01", min_value="0"),
            _option_field("train", "adaptive_cutoffs", "adaptive_cutoffs", placeholder="1024,4096"),
            _option_field("train", "adaptive_div_value", "adaptive_div_value", kind="number", step="0.1", min_value="0"),
            _option_field("train", "emb_dropout", "emb_dropout", kind="number", step="0.01", min_value="0"),
            _option_field("train", "mlp_dropout", "mlp_dropout", kind="number", step="0.01", min_value="0"),
            _option_field("train", "residual_dropout", "residual_dropout", kind="number", step="0.01", min_value="0"),
        ),
    ),
    UiSection(
        title="Dynamics",
        fields=(
            _option_field("train", "decay_impl", "decay_impl", kind="select"),
            _option_field("train", "rosa_impl", "rosa_impl", kind="select"),
            _option_field("train", "rosa_layers", "rosa_layers", placeholder="all / final / 4,5,6"),
            _option_field("train", "gamma_min", "gamma_min", kind="number", step="0.0001"),
            _option_field("train", "gamma_max", "gamma_max", kind="number", step="0.0001"),
            _option_field("train", "alpha_cap", "alpha_cap", kind="number", step="0.01"),
        ),
    ),
    UiSection(
        title="Optimization",
        fields=(
            _option_field("train", "lr", "Learning Rate", kind="number", step="0.0001", min_value="0"),
            _option_field("train", "lr_floor", "lr_floor", kind="number", step="0.0001", min_value="0"),
            _option_field("train", "beta1", "beta1", kind="number", step="0.0001", min_value="0"),
            _option_field("train", "beta2", "beta2", kind="number", step="0.0001", min_value="0"),
            _option_field("train", "warmup_steps", "warmup_steps", kind="number", min_value="0"),
            _option_field("train", "weight_decay", "weight_decay", kind="number", step="0.0001", min_value="0"),
            _option_field("train", "bias_lr_mult", "bias_lr_mult", kind="number", step="0.01", min_value="0"),
            _option_field("train", "norm_lr_mult", "norm_lr_mult", kind="number", step="0.01", min_value="0"),
            _option_field("train", "emb_lr_mult", "emb_lr_mult", kind="number", step="0.01", min_value="0"),
            _option_field("train", "k_logit_lr_mult", "k_logit_lr_mult", kind="number", step="0.01", min_value="0"),
            _option_field("train", "optimizer_mode", "optimizer_mode", kind="select"),
            _option_field("train", "eval_interval", "Eval Interval", kind="number", min_value="1"),
            _option_field(
                "train",
                "fused_adamw",
                "fused_adamw",
                kind="checkbox",
                help_text="Unchecked emits --no-fused-adamw.",
                false_flag="--no-fused-adamw",
            ),
        ),
    ),
    UiSection(
        title="Runtime",
        fields=(
            _option_field("train", "compile", "compile", kind="checkbox", true_flag="--compile"),
            _option_field("train", "compile_mode", "compile_mode", kind="select"),
            _option_field("train", "diagnostics", "diagnostics", kind="checkbox", true_flag="--diagnostics"),
            _option_field("train", "deterministic", "deterministic", kind="checkbox", true_flag="--deterministic"),
            _option_field(
                "train",
                "deterministic_warn_only",
                "deterministic_warn_only",
                kind="checkbox",
                true_flag="--deterministic-warn-only",
            ),
            _option_field("train", "no_tf32", "no_tf32", kind="checkbox", true_flag="--no-tf32"),
            _option_field("train", "strict_repro", "strict_repro", kind="checkbox", true_flag="--strict-repro"),
            _option_field("train", "eval_only", "eval_only", kind="checkbox", true_flag="--eval-only"),
            _option_field("train", "run_manifest", "run_manifest", placeholder="runs/manifest.json"),
            _option_field("train", "verbose", "verbose", kind="checkbox", true_flag="--verbose"),
        ),
    ),
    UiSection(
        title="Sampling",
        fields=(
            _option_field("train", "sample", "sample after run", kind="checkbox", true_flag="--sample"),
            _option_field("train", "prompt", "Prompt", placeholder="To be, or not to be"),
            _option_field("train", "sample_tokens", "Sample Tokens", kind="number", min_value="0"),
            _option_field("train", "temperature", "Temperature", kind="number", step="0.1", min_value="0"),
            _option_field("train", "top_k", "top_k", kind="number", min_value="0"),
            _option_field("train", "top_p", "top_p", kind="number", step="0.01", min_value="0"),
            _option_field("train", "repetition_penalty", "repetition_penalty", kind="number", step="0.01", min_value="0"),
            _option_field("train", "repetition_window", "repetition_window", kind="number", min_value="0"),
            _option_field("train", "prompt_lock_tokens", "prompt_lock_tokens", kind="number", min_value="0"),
        ),
    ),
    UiSection(
        title="Advanced",
        fields=(
            _free_field(
                "extra_args",
                "Extra CLI Args",
                kind="textarea",
                rows=5,
                placeholder="--gamma-min 0.05 --gamma-max 0.9995 --strict-repro",
                help_text="Anything not exposed above can still be passed here.",
            ),
        ),
    ),
)


INFER_FORM_SECTIONS: tuple[UiSection, ...] = (
    UiSection(
        title="Run",
        fields=(
            _free_field("job_name", "Run Name", placeholder="eval-byte-shakespeare"),
            _option_field(
                "infer",
                "ckpt",
                "Checkpoint Path",
                required=True,
                placeholder="byte_shakespeare.pt",
                datalist_id="checkpoint-list",
            ),
            _option_field("infer", "dataset", "Dataset", kind="select"),
            _option_field("infer", "tokenizer", "Tokenizer", kind="select"),
            _option_field("infer", "data_path", "Train Data Path", placeholder="data/my_corpus.txt"),
            _option_field("infer", "val_path", "Validation Path", placeholder="data/my_corpus_val.txt"),
            _option_field("infer", "val_frac", "Val Fraction", kind="number", step="0.01", min_value="0"),
            _option_field("infer", "batch_size", "Batch Size", kind="number", min_value="1"),
            _option_field("infer", "window", "Window", kind="number", min_value="1"),
            _option_field("infer", "seed", "Seed", kind="number", min_value="0"),
        ),
    ),
    UiSection(
        title="Model",
        fields=(
            _option_field("infer", "d_model", "d_model", kind="number", min_value="1"),
            _option_field("infer", "emb_dim", "emb_dim", kind="number", min_value="1", placeholder="defaults to d_model"),
            _option_field("infer", "rank", "Rank", kind="number", min_value="1"),
            _option_field("infer", "n_k2", "K2 Layers", kind="number", min_value="1"),
            _option_field("infer", "k_base_rank", "k_base_rank", kind="number"),
            _option_field("infer", "k_base_impl", "k_base_impl", kind="select"),
            _option_field("infer", "k_base_kernel_size", "k_base_kernel_size", kind="number", min_value="1"),
            _option_field(
                "infer",
                "share_k_base",
                "share_k_base",
                kind="checkbox",
                help_text="Enable one shared k_base across all K2 layers.",
                true_flag="--share-k-base",
            ),
            _option_field("infer", "head_mode", "Head Mode", kind="select"),
            _option_field("infer", "head_mult", "Head Mult", kind="number", min_value="1"),
        ),
    ),
    UiSection(
        title="Dynamics",
        fields=(
            _option_field("infer", "decay_impl", "decay_impl", kind="select"),
            _option_field("infer", "rosa_impl", "rosa_impl", kind="select"),
            _option_field("infer", "rosa_layers", "rosa_layers", placeholder="all / final / 4,5,6"),
            _option_field("infer", "gamma_min", "gamma_min", kind="number", step="0.0001"),
            _option_field("infer", "gamma_max", "gamma_max", kind="number", step="0.0001"),
            _option_field("infer", "alpha_cap", "alpha_cap", kind="number", step="0.01"),
        ),
    ),
    UiSection(
        title="Eval + Sample",
        fields=(
            _option_field("infer", "compile", "compile", kind="checkbox", true_flag="--compile"),
            _option_field("infer", "compile_mode", "compile_mode", kind="select"),
            _option_field("infer", "skip_eval", "skip eval", kind="checkbox", true_flag="--skip-eval"),
            _option_field("infer", "skip_sample", "skip sample", kind="checkbox", true_flag="--skip-sample"),
            _option_field("infer", "prompt", "Prompt", placeholder="To be, or not to be"),
            _option_field("infer", "sample_tokens", "Sample Tokens", kind="number", min_value="0"),
            _option_field("infer", "temperature", "Temperature", kind="number", step="0.1", min_value="0"),
            _option_field("infer", "top_k", "top_k", kind="number", min_value="0"),
            _option_field("infer", "top_p", "top_p", kind="number", step="0.01", min_value="0"),
        ),
    ),
    UiSection(
        title="Advanced",
        fields=(
            _free_field(
                "extra_args",
                "Extra CLI Args",
                kind="textarea",
                rows=5,
                placeholder="--prompt-lock-tokens 0 --repetition-penalty 1.05",
                help_text="Anything not exposed above can still be passed here.",
            ),
        ),
    ),
)


FORM_SECTIONS_BY_MODE: dict[str, tuple[UiSection, ...]] = {
    "train": TRAIN_FORM_SECTIONS,
    "infer": INFER_FORM_SECTIONS,
}


def _timestamp_iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _bool_from_form(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _split_extra_args(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    modes = []
    for candidate in (os.name != "nt", True, False):
        if candidate not in modes:
            modes.append(candidate)
    last_error: Exception | None = None
    for posix in modes:
        try:
            return shlex.split(text, posix=posix)
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"Unable to parse extra args: {last_error}")


def _format_command(command: Iterable[str]) -> str:
    parts = list(command)
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def _candidate_python_script_dirs() -> list[str]:
    candidates: list[str] = []

    def add(path: str | Path | None) -> None:
        if not path:
            return
        path_obj = Path(path)
        if not path_obj.exists():
            return
        path_str = str(path_obj)
        if path_str not in candidates:
            candidates.append(path_str)

    add(sysconfig.get_path("scripts"))
    if os.name == "nt":
        try:
            add(sysconfig.get_path("scripts", scheme="nt_user"))
        except Exception:
            pass
        add(Path(site.USER_BASE) / f"Python{sysconfig.get_python_version().replace('.', '')}" / "Scripts")
        user_base = Path(site.USER_BASE)
        if user_base.exists():
            for path in sorted(user_base.glob("Python*/Scripts")):
                add(path)
        roaming_python = Path.home() / "AppData" / "Roaming" / "Python"
        if roaming_python.exists():
            for path in sorted(roaming_python.glob("Python*/Scripts")):
                add(path)
    return candidates


def _augment_subprocess_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    base = dict(os.environ if env is None else env)
    base["PYTHONUNBUFFERED"] = "1"
    script_candidates = _candidate_python_script_dirs()
    existing = [part for part in base.get("PATH", "").split(os.pathsep) if part]
    prepend: list[str] = []
    normalized_existing = {str(Path(part)) for part in existing if part}
    for candidate in script_candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        candidate_str = str(candidate_path)
        if not candidate_path.exists():
            continue
        if candidate_str in normalized_existing or candidate_str in prepend:
            continue
        prepend.append(candidate_str)
    if prepend:
        base["PATH"] = os.pathsep.join(prepend + existing)
    return base


def _validate_mode_args(mode: str, args: list[str]) -> None:
    parser = _PARSER_BY_MODE[mode]
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            parser.parse_args(args)
    except SystemExit as exc:
        if exc.code == 0:
            return
        message = stderr.getvalue().strip() or stdout.getvalue().strip() or "Invalid arguments."
        raise ValueError(message) from None


def build_command_from_form(mode: str, form_data: Mapping[str, Any]) -> PreparedRun:
    if mode not in FORM_SECTIONS_BY_MODE:
        raise ValueError(f"Unknown mode: {mode}")

    args: list[str] = []
    job_name = str(form_data.get("job_name", "") or "").strip()
    name = job_name or f"{mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    for section in FORM_SECTIONS_BY_MODE[mode]:
        for field in section.fields:
            if field.name in {"job_name", "extra_args"}:
                continue
            raw_value = form_data.get(field.name)
            if field.kind == "checkbox":
                checked = _bool_from_form(raw_value)
                if checked:
                    if field.true_flag is not None and field.default is not True:
                        args.append(field.true_flag)
                    elif field.true_flag is not None and field.always_include:
                        args.append(field.true_flag)
                elif field.false_flag is not None:
                    args.append(field.false_flag)
                continue

            text_value = "" if raw_value is None else str(raw_value).strip()
            if not text_value:
                continue
            default_value = "" if field.default is None else str(field.default)
            if field.always_include or text_value != default_value:
                if field.arg is None:
                    raise ValueError(f"Field {field.name} is missing a CLI flag mapping.")
                args.extend([field.arg, text_value])

    extra_args = str(form_data.get("extra_args", "") or "")
    if extra_args.strip():
        args.extend(_split_extra_args(extra_args))

    _validate_mode_args(mode, args)

    module_name = "k_language_model.train_app" if mode == "train" else "k_language_model.infer_app"
    command = [sys.executable, "-m", module_name, *args]
    return PreparedRun(
        mode=mode,
        name=name,
        args=args,
        command=command,
        command_display=_format_command(command),
    )


def list_checkpoint_paths(limit: int = 40) -> list[str]:
    candidates: dict[Path, float] = {}
    search_roots = [PROJECT_ROOT, PROJECT_ROOT / "models"]
    for root in search_roots:
        if not root.exists():
            continue
        pattern = "*.pt" if root == PROJECT_ROOT else "**/*.pt"
        for path in root.glob(pattern):
            if path.is_file():
                try:
                    candidates[path.resolve()] = path.stat().st_mtime
                except OSError:
                    continue
    ordered = sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path) for path, _ in ordered]


def _parse_structured_log_line(line: str) -> tuple[str, str]:
    match = _LOG_LINE_RE.match(line)
    if match is None:
        return "RAW", line.strip()
    return match.group("level"), match.group("message").strip()


def _parse_pipe_key_values(message: str) -> dict[str, str]:
    parts = [part.strip() for part in message.split(" | ")]
    parsed: dict[str, str] = {}
    for part in parts[1:]:
        match = _PLAIN_KEY_VALUE_RE.match(part)
        if match is not None:
            parsed[match.group("key")] = match.group("value").strip()
            continue
        if part.endswith(" ms/step"):
            parsed["ms_per_step"] = part[: -len(" ms/step")].strip()
        elif part.endswith(" tok/s"):
            parsed["tok_s"] = part[: -len(" tok/s")].strip()
    return parsed


def _update_dict_strings(target: dict[str, str], source: Mapping[str, str], allowed: set[str] | None = None) -> None:
    for key, value in source.items():
        if allowed is not None and key not in allowed:
            continue
        if value == "":
            continue
        target[key] = value


def _append_event(job: JobRecord, message: str) -> None:
    text = message.strip()
    if not text:
        return
    suppressed_prefixes = (
        "diagnostics |",
        "grad_top_params |",
        "adam_expavg_weight_top_params |",
        "adam_expavg_weight_by_layer",
        "layer_stats |",
        "eval_refinement |",
        "Command |",
        "Environment |",
    )
    if text.startswith(suppressed_prefixes):
        return
    if job.recent_events and job.recent_events[-1] == text:
        return
    job.recent_events.append(text)


def _phase_from_status(status: str, phase: str) -> str:
    if status in {"queued", "cancelled", "succeeded", "failed"}:
        return status
    return phase or "running"


def _ingest_job_output_message(job: JobRecord, level: str, message: str) -> None:
    if not message:
        return

    job.last_message = message
    if level == "WARNING" or "UserWarning:" in message:
        job.warning_count += 1
        job.last_warning = message
    if level in {"WARNING", "ERROR", "CRITICAL"} or message.startswith("[ui]"):
        _append_event(job, message)

    if message.startswith("Runtime |"):
        _update_dict_strings(
            job.runtime_info,
            _parse_pipe_key_values(message),
            allowed={"device", "amp", "seed", "strict_repro", "deterministic", "deterministic_warn_only", "tf32"},
        )
        _append_event(job, message)
        return

    if message.startswith("Training start |"):
        parsed = _parse_pipe_key_values(message)
        steps_raw = parsed.get("steps")
        if steps_raw is not None:
            try:
                job.total_steps = int(steps_raw)
            except ValueError:
                pass
        job.phase = "training"
        _update_dict_strings(
            job.summary_stats,
            parsed,
            allowed={"window", "batch", "lr", "opt_mode", "fused_adamw", "grad_scaler"},
        )
        if "device" in parsed:
            job.runtime_info["device"] = parsed["device"]
        _append_event(job, message)
        return

    if message.startswith("Compile warmup"):
        job.phase = "warming_up"
        _append_event(job, message)
        return

    if message.startswith("Model params |"):
        parsed = _parse_pipe_key_values(message)
        renamed = {
            "params_total": parsed.get("total", ""),
            "params_embedding": parsed.get("embedding", ""),
            "params_k_stack": parsed.get("k_stack", ""),
            "params_head": parsed.get("head", ""),
            "params_other": parsed.get("other", ""),
        }
        _update_dict_strings(job.summary_stats, renamed)
        _append_event(job, message)
        return

    if _STEP_PREFIX_RE.match(message):
        parsed = _parse_pipe_key_values(f"step | {message.split(' | ', 1)[1] if ' | ' in message else ''}")
        step_match = _STEP_PREFIX_RE.match(message)
        if step_match is not None:
            job.current_step = int(step_match.group("step"))
        job.phase = "training"
        _update_dict_strings(
            job.summary_stats,
            parsed,
            allowed={"train_ce", "train_ce_ema", "train_bpc", "train_bpc_ema", "val_ce", "val_bpc", "val_ppl", "best_ppl", "lr", "ms_per_step", "tok_s"},
        )
        _append_event(job, message)
        return

    if message.startswith("Eval only |") or message.startswith("Eval |"):
        job.phase = "evaluating"
        _update_dict_strings(
            job.summary_stats,
            _parse_pipe_key_values(message),
            allowed={"step", "ckpt_best_ppl", "val_ce", "val_bpc", "val_ppl", "eval_tok_s"},
        )
        _append_event(job, message)
        return

    if message.startswith("Sample speed |"):
        job.phase = "sampling"
        _update_dict_strings(
            job.summary_stats,
            _parse_pipe_key_values(message),
            allowed={"prompt_tokens", "generated_tokens", "sample_tok_s"},
        )
        _append_event(job, message)
        return

    if message.startswith("Training complete |"):
        job.phase = "completed"
        match = _PLAIN_KEY_VALUE_RE.search(message.replace("Training complete | ", ""))
        if match is not None:
            job.summary_stats["best_perplexity"] = match.group("value").strip()
        _append_event(job, message)
        return

    if message.startswith("Non-finite loss") or message.startswith("Non-finite gradient norm") or message.startswith("Plateau detected"):
        _append_event(job, message)
        return

    if message.startswith("Adaptive head |") or message.startswith("Adaptive config |") or message.startswith("Model config |"):
        _append_event(job, message)
        return

    if level == "RAW":
        if message.startswith("Traceback") or message.startswith("File "):
            _append_event(job, message)
        return

    if level in {"INFO", "WARNING", "ERROR", "CRITICAL"}:
        _append_event(job, message)


class RunScheduler:
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = PROJECT_ROOT if project_root is None else Path(project_root)
        self.log_dir = self.project_root / "runs" / "ui_jobs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._jobs: dict[str, JobRecord] = {}
        self._queue: deque[str] = deque()
        self._stop_requested = False
        self._worker = threading.Thread(target=self._worker_loop, name="k-lm-ui-worker", daemon=True)
        self._worker.start()

    def shutdown(self) -> None:
        running_processes: list[subprocess.Popen[str]] = []
        with self._condition:
            self._stop_requested = True
            for job in self._jobs.values():
                if job.status == "queued":
                    job.status = "cancelled"
                    job.phase = "cancelled"
                    job.finished_at = time.time()
                    self._append_internal_log_unlocked(job, "[ui] scheduler shutdown before start")
                elif job.status == "running" and job.process is not None:
                    job.cancel_requested = True
                    self._append_internal_log_unlocked(job, "[ui] scheduler shutdown requested")
                    running_processes.append(job.process)
            self._condition.notify_all()
        for process in running_processes:
            try:
                process.terminate()
            except OSError:
                pass
        self._worker.join(timeout=2.0)

    def submit(self, mode: str, form_data: Mapping[str, Any]) -> dict[str, Any]:
        prepared = build_command_from_form(mode, form_data)
        created_at = time.time()
        job_id = uuid.uuid4().hex[:8]
        log_path = self.log_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{job_id}_{mode}.log"
        job = JobRecord(
            job_id=job_id,
            mode=prepared.mode,
            name=prepared.name,
            args=list(prepared.args),
            command=list(prepared.command),
            command_display=prepared.command_display,
            created_at=created_at,
            log_path=log_path,
        )
        with self._condition:
            self._jobs[job_id] = job
            self._queue.append(job_id)
            self._append_internal_log_unlocked(job, f"[ui] queued {prepared.command_display}")
            self._condition.notify_all()
            return self._snapshot_job_unlocked(job)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return self._snapshot_job_unlocked(job)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [self._snapshot_job_unlocked(job) for job in jobs]

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status == "queued":
                job.status = "cancelled"
                job.phase = "cancelled"
                job.finished_at = time.time()
                self._append_internal_log_unlocked(job, "[ui] cancelled before start")
                return self._snapshot_job_unlocked(job)
            if job.status != "running":
                return self._snapshot_job_unlocked(job)
            job.cancel_requested = True
            self._append_internal_log_unlocked(job, "[ui] cancellation requested")
            process = job.process
        if process is not None:
            try:
                process.terminate()
            except OSError:
                pass
        with self._lock:
            job = self._jobs[job_id]
            return self._snapshot_job_unlocked(job)

    def _append_internal_log_unlocked(self, job: JobRecord, line: str) -> None:
        job.log_tail.append(line)
        job.log_line_count += 1
        with job.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
        _ingest_job_output_message(job, "RAW", line)

    def _append_process_output(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.log_tail.append(line)
            job.log_line_count += 1
            with job.log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
            level, message = _parse_structured_log_line(line)
            _ingest_job_output_message(job, level, message)

    def _snapshot_job_unlocked(self, job: JobRecord) -> dict[str, Any]:
        queue_position: int | None = None
        if job.status == "queued":
            live_queue = [queued_id for queued_id in self._queue if self._jobs.get(queued_id, None) is not None]
            for index, queued_id in enumerate(live_queue, 1):
                queued_job = self._jobs[queued_id]
                if queued_job.status != "queued":
                    continue
                if queued_id == job.job_id:
                    queue_position = index
                    break
        return {
            "job_id": job.job_id,
            "mode": job.mode,
            "name": job.name,
            "status": job.status,
            "created_at": _timestamp_iso(job.created_at),
            "started_at": _timestamp_iso(job.started_at),
            "finished_at": _timestamp_iso(job.finished_at),
            "returncode": job.returncode,
            "pid": job.pid,
            "cancel_requested": job.cancel_requested,
            "error": job.error,
            "queue_position": queue_position,
            "command": job.command,
            "command_display": job.command_display,
            "log_line_count": job.log_line_count,
            "log_path": str(job.log_path),
            "phase": _phase_from_status(job.status, job.phase),
            "current_step": job.current_step,
            "total_steps": job.total_steps,
            "summary_stats": dict(job.summary_stats),
            "runtime_info": dict(job.runtime_info),
            "recent_events": list(job.recent_events),
            "warning_count": job.warning_count,
            "last_warning": job.last_warning,
            "last_message": job.last_message,
        }

    def _read_process_output(self, job_id: str, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            self._append_process_output(job_id, line.rstrip("\r\n"))

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._stop_requested:
                    if self._queue:
                        candidate_id = self._queue.popleft()
                        candidate = self._jobs.get(candidate_id)
                        if candidate is not None and candidate.status == "queued":
                            job = candidate
                            break
                    self._condition.wait(timeout=0.25)
                else:
                    return

                job.status = "running"
                job.phase = "starting"
                job.started_at = time.time()
                self._append_internal_log_unlocked(job, "[ui] starting process")

            env = os.environ.copy()
            env = _augment_subprocess_env(env)
            process = subprocess.Popen(
                job.command,
                cwd=self.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with self._lock:
                job.process = process
                job.pid = process.pid

            reader = threading.Thread(
                target=self._read_process_output,
                args=(job.job_id, process),
                name=f"k-lm-ui-log-{job.job_id}",
                daemon=True,
            )
            reader.start()
            returncode = process.wait()
            reader.join(timeout=1.0)

            with self._condition:
                job.process = None
                job.pid = None
                job.returncode = returncode
                job.finished_at = time.time()
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.phase = "cancelled"
                    self._append_internal_log_unlocked(job, f"[ui] process cancelled (rc={returncode})")
                elif returncode == 0:
                    job.status = "succeeded"
                    job.phase = "succeeded"
                    self._append_internal_log_unlocked(job, "[ui] process completed successfully")
                else:
                    job.status = "failed"
                    job.phase = "failed"
                    job.error = f"Process exited with return code {returncode}."
                    self._append_internal_log_unlocked(job, f"[ui] process failed (rc={returncode})")
