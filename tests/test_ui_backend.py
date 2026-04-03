from __future__ import annotations

import os
import sys
import sysconfig
from collections import deque
from pathlib import Path

import pytest

from k_language_model.train_app import build_parser as build_train_parser
from k_language_model.ui_backend import JobRecord
from k_language_model.ui_backend import _augment_subprocess_env
from k_language_model.ui_backend import _ingest_job_output_message
from k_language_model.ui_backend import build_command_from_form
from k_language_model.ui_backend import TRAIN_FORM_SECTIONS


def test_build_train_command_from_form_emits_expected_flags() -> None:
    prepared = build_command_from_form(
        "train",
        {
            "job_name": "byte-sweep",
            "dataset": "shakespeare",
            "tokenizer": "byte",
            "ckpt": "models/char_shakespeare_v2.pt",
            "steps": "500",
            "batch_size": "16",
            "window": "256",
            "d_model": "64",
            "rank": "4",
            "n_k2": "6",
            "k_base_rank": "0",
            "k_base_impl": "fused",
            "share_k_base": "on",
            "decay_impl": "mask",
            "rosa_impl": "exact",
            "rosa_layers": "final",
            "compile": "on",
            "extra_args": "--gamma-min 0.05 --gamma-max 0.9995",
        },
    )

    assert prepared.name == "byte-sweep"
    assert prepared.command[:3] == [sys.executable, "-m", "k_language_model.train_app"]
    assert "--share-k-base" in prepared.command
    assert "--compile" in prepared.command
    assert "--no-fused-adamw" in prepared.command
    assert "--gamma-min" in prepared.command
    assert "0.9995" in prepared.command


def test_build_infer_command_requires_checkpoint() -> None:
    with pytest.raises(ValueError, match="required: --ckpt/--checkpoint"):
        build_command_from_form(
            "infer",
            {
                "dataset": "shakespeare",
                "tokenizer": "byte",
            },
        )


def test_train_form_covers_all_train_parser_options() -> None:
    parser = build_train_parser()
    parser_dests = {
        action.dest
        for action in parser._actions
        if action.option_strings and action.dest not in {"help"}
    }
    ui_dests = {
        field.name
        for section in TRAIN_FORM_SECTIONS
        for field in section.fields
    }
    missing = sorted(parser_dests - ui_dests)
    assert missing == []


def test_augment_subprocess_env_sets_python_script_paths() -> None:
    env = _augment_subprocess_env({"PATH": ""})
    assert env["PYTHONUNBUFFERED"] == "1"
    scripts_path = sysconfig.get_path("scripts")
    if scripts_path and Path(scripts_path).exists():
        assert scripts_path in env["PATH"].split(os.pathsep)


def _make_job() -> JobRecord:
    return JobRecord(
        job_id="job123",
        mode="train",
        name="demo",
        args=[],
        command=[sys.executable, "-m", "k_language_model.train_app"],
        command_display="python -m k_language_model.train_app",
        created_at=0.0,
        log_path=Path("runs/ui_jobs/demo.log"),
        recent_events=deque(maxlen=10),
        log_tail=deque(maxlen=10),
    )


def test_ingest_train_step_message_updates_progress_and_metrics() -> None:
    job = _make_job()
    job.total_steps = 25000
    _ingest_job_output_message(
        job,
        "INFO",
        "step=  500 | train_ce=2.1234 | train_ce_ema=2.0000 | val_ce=1.9000 | val_ppl=6.69 | best_ppl=6.69 | lr=1.00e-03 | 12.3 ms/step | 333 tok/s",
    )

    assert job.phase == "training"
    assert job.current_step == 500
    assert job.summary_stats["train_ce"] == "2.1234"
    assert job.summary_stats["val_ppl"] == "6.69"
    assert job.summary_stats["ms_per_step"] == "12.3"
    assert job.summary_stats["tok_s"] == "333"
    assert any("step=" in event for event in job.recent_events)


def test_ingest_diagnostics_message_is_not_added_to_recent_events() -> None:
    job = _make_job()
    _ingest_job_output_message(
        job,
        "INFO",
        "diagnostics | gnorm=1.00e+00 | gnorm_clip=1.00e+00 | clip_hit=false",
    )
    assert list(job.recent_events) == []


def test_ingest_infer_eval_message_updates_summary() -> None:
    job = _make_job()
    job.mode = "infer"
    _ingest_job_output_message(
        job,
        "INFO",
        "Eval | step=7500 | ckpt_best_ppl=22.53 | val_ce=3.1148 | val_ppl=22.53 | eval_tok_s=51234",
    )

    assert job.phase == "evaluating"
    assert job.summary_stats["step"] == "7500"
    assert job.summary_stats["val_ce"] == "3.1148"
    assert job.summary_stats["eval_tok_s"] == "51234"


def test_ingest_sample_stream_message_sets_sampling_phase() -> None:
    job = _make_job()
    job.mode = "infer"
    _ingest_job_output_message(
        job,
        "INFO",
        "Sample stream | enabled=true | interval_tokens=4 | tty=false",
    )

    assert job.phase == "sampling"
    assert job.summary_stats["interval_tokens"] == "4"
    assert job.summary_stats["tty"] == "false"
