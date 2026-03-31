from __future__ import annotations

import argparse
import html
import urllib.parse
from typing import Any

from .ui_backend import FORM_SECTIONS_BY_MODE, PROJECT_ROOT, RunScheduler, UiField, list_checkpoint_paths

_FASTAPI_IMPORT_ERROR: Exception | None = None

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    FastAPI = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]
    Request = Any  # type: ignore[assignment]
    HTMLResponse = Any  # type: ignore[assignment]
    JSONResponse = Any  # type: ignore[assignment]
    _FASTAPI_IMPORT_ERROR = exc


def _require_fastapi() -> None:
    if _FASTAPI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "FastAPI UI dependencies are not installed. Run `pip install -e .[ui]` "
            "or `pip install fastapi 'uvicorn[standard]'`."
        ) from _FASTAPI_IMPORT_ERROR


def _field_value(field: UiField) -> str:
    if field.default is None:
        return ""
    return str(field.default)


def _render_input(mode: str, field: UiField) -> str:
    field_id = f"{mode}-{field.name}"
    label = html.escape(field.label)
    help_text = f'<div class="field-help">{html.escape(field.help_text)}</div>' if field.help_text else ""
    required_attr = " required" if field.required else ""
    placeholder_attr = f' placeholder="{html.escape(field.placeholder)}"' if field.placeholder else ""
    datalist_attr = f' list="{html.escape(field.datalist_id)}"' if field.datalist_id else ""
    min_attr = f' min="{html.escape(field.min_value)}"' if field.min_value is not None else ""
    step_attr = f' step="{html.escape(field.step)}"' if field.step is not None else ""

    if field.kind == "checkbox":
        checked_attr = " checked" if bool(field.default) else ""
        return (
            f'<label class="field checkbox-field" for="{field_id}">'
            f'<input id="{field_id}" name="{html.escape(field.name)}" type="checkbox"{checked_attr}>'
            f"<span>{label}</span>"
            f"{help_text}"
            f"</label>"
        )

    if field.kind == "select":
        options = []
        selected_value = _field_value(field)
        for choice in field.choices:
            choice_str = str(choice)
            selected = " selected" if choice_str == selected_value else ""
            options.append(
                f'<option value="{html.escape(choice_str)}"{selected}>{html.escape(choice_str)}</option>'
            )
        control = (
            f'<select id="{field_id}" name="{html.escape(field.name)}"{required_attr}>'
            + "".join(options)
            + "</select>"
        )
    elif field.kind == "textarea":
        control = (
            f'<textarea id="{field_id}" name="{html.escape(field.name)}" rows="{field.rows}"'
            f"{required_attr}{placeholder_attr}>"
            f"{html.escape(_field_value(field))}</textarea>"
        )
    else:
        input_type = "number" if field.kind == "number" else "text"
        value_attr = f' value="{html.escape(_field_value(field))}"' if _field_value(field) else ""
        control = (
            f'<input id="{field_id}" name="{html.escape(field.name)}" type="{input_type}"'
            f"{value_attr}{required_attr}{placeholder_attr}{datalist_attr}{min_attr}{step_attr}>"
        )

    return (
        f'<label class="field" for="{field_id}">'
        f'<span class="field-label">{label}</span>'
        f"{control}"
        f"{help_text}"
        f"</label>"
    )


def _render_form(mode: str) -> str:
    title = "Train Queue" if mode == "train" else "Infer Queue"
    subtitle = (
        "Launch full training jobs without rewriting a giant command each time."
        if mode == "train"
        else "Queue eval or sampling runs against existing checkpoints."
    )
    sections_html = []
    for section in FORM_SECTIONS_BY_MODE[mode]:
        fields_html = "".join(_render_input(mode, field) for field in section.fields)
        sections_html.append(
            '<section class="panel-section">'
            f'<header><h3>{html.escape(section.title)}</h3></header>'
            f'<div class="field-grid">{fields_html}</div>'
            "</section>"
        )
    return (
        f'<form class="queue-form" data-mode="{mode}">'
        f'<input type="hidden" name="__mode" value="{mode}">'
        f'<div class="panel-head"><div><h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p></div>'
        f'<button type="submit" class="primary-button">Queue {html.escape(mode)}</button></div>'
        f'{"".join(sections_html)}'
        '<div class="submit-status" aria-live="polite"></div>'
        "</form>"
    )


def _render_checkpoint_datalist() -> str:
    options = [
        f'<option value="{html.escape(path)}"></option>'
        for path in list_checkpoint_paths()
    ]
    return f'<datalist id="checkpoint-list">{"".join(options)}</datalist>'


def _build_page() -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>K-LM Run Queue</title>
  <style>
    :root {{
      color-scheme: dark;
      --ink: #f5f5f5;
      --ink-soft: #a1a1aa;
      --card: rgba(22, 19, 31, 0.94);
      --card-strong: rgba(30, 26, 44, 0.96);
      --card-soft: rgba(18, 15, 26, 0.88);
      --line: rgba(167, 139, 250, 0.16);
      --line-strong: rgba(167, 139, 250, 0.24);
      --accent: #7c3aed;
      --accent-strong: #6d28d9;
      --accent-cool: #a78bfa;
      --queued: #fbbf24;
      --running: #8b5cf6;
      --success: #22c55e;
      --failed: #fb7185;
      --cancelled: #71717a;
      --shadow: 0 18px 50px rgba(0, 0, 0, 0.38);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(124, 58, 237, 0.28), transparent 30%),
        radial-gradient(circle at top right, rgba(167, 139, 250, 0.18), transparent 28%),
        linear-gradient(180deg, #0e0c14 0%, #120f1a 46%, #171320 100%);
    }}
    .shell {{
      width: min(1500px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 2.1fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .hero-card, .panel, .jobs-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .hero-card {{
      padding: 26px 28px;
    }}
    .hero-card h1 {{
      margin: 0 0 10px;
      font-family: "Palatino Linotype", "Book Antiqua", serif;
      font-size: clamp(2rem, 3vw, 3.4rem);
      line-height: 0.96;
      letter-spacing: -0.04em;
    }}
    .hero-card p {{
      margin: 0;
      max-width: 58rem;
      color: var(--ink-soft);
      line-height: 1.55;
    }}
    .meta-card {{
      padding: 22px 24px;
      display: grid;
      gap: 12px;
      align-content: start;
    }}
    .meta-card .eyebrow {{
      font-size: 0.78rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent-cool);
      font-weight: 700;
    }}
    .meta-value {{
      font-family: "Consolas", "Cascadia Mono", monospace;
      font-size: 0.95rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(360px, 0.95fr);
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      padding: 18px;
    }}
    .control-panel {{
      display: grid;
      gap: 16px;
    }}
    .tab-strip {{
      display: inline-flex;
      gap: 8px;
      padding: 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(30, 26, 44, 0.88);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }}
    .tab-button {{
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 11px 18px;
      cursor: pointer;
      font: inherit;
      font-weight: 800;
      letter-spacing: 0.04em;
      color: var(--ink-soft);
      background: transparent;
      transition: background 140ms ease, color 140ms ease, transform 140ms ease;
    }}
    .tab-button:hover {{
      transform: translateY(-1px);
    }}
    .tab-button.active {{
      color: #f5f5f5;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
      box-shadow: 0 10px 24px rgba(124, 58, 237, 0.28);
    }}
    .tab-panel {{
      display: none;
    }}
    .tab-panel.active {{
      display: block;
      animation: tabReveal 160ms ease;
    }}
    @keyframes tabReveal {{
      from {{
        opacity: 0;
        transform: translateY(4px);
      }}
      to {{
        opacity: 1;
        transform: translateY(0);
      }}
    }}
    .panel-head {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: start;
      margin-bottom: 14px;
    }}
    .panel-head h2 {{
      margin: 0 0 4px;
      font-family: "Palatino Linotype", "Book Antiqua", serif;
      font-size: 1.55rem;
    }}
    .panel-head p {{
      margin: 0;
      color: var(--ink-soft);
      line-height: 1.5;
    }}
    .panel-section {{
      padding-top: 14px;
      border-top: 1px solid var(--line);
      margin-top: 14px;
    }}
    .panel-section header h3 {{
      margin: 0 0 12px;
      font-size: 0.9rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent-cool);
    }}
    .field-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .field {{
      display: grid;
      gap: 6px;
      align-content: start;
    }}
    .field-label {{
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--ink-soft);
    }}
    .field input, .field select, .field textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: var(--card-strong);
      color: var(--ink);
      font: inherit;
    }}
    .field input::placeholder, .field textarea::placeholder {{
      color: #7f7f89;
    }}
    .field textarea {{
      resize: vertical;
      min-height: 118px;
      font-family: "Consolas", "Cascadia Mono", monospace;
      font-size: 0.92rem;
    }}
    .field-help {{
      color: var(--ink-soft);
      font-size: 0.82rem;
      line-height: 1.45;
    }}
    .checkbox-field {{
      grid-template-columns: auto 1fr;
      gap: 10px 12px;
      align-items: start;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--card-strong);
    }}
    .checkbox-field input {{
      width: 18px;
      height: 18px;
      margin-top: 2px;
    }}
    .checkbox-field span {{
      font-weight: 700;
      color: var(--ink-soft);
    }}
    .primary-button, .secondary-button {{
      appearance: none;
      border: 1px solid transparent;
      border-radius: 999px;
      padding: 12px 18px;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
      transition: transform 120ms ease, opacity 120ms ease;
    }}
    .primary-button {{
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
      color: #f5f5f5;
      box-shadow: 0 10px 24px rgba(124, 58, 237, 0.3);
    }}
    .secondary-button {{
      background: rgba(167, 139, 250, 0.08);
      border-color: var(--line);
      color: var(--ink);
    }}
    .primary-button:hover, .secondary-button:hover {{
      transform: translateY(-1px);
    }}
    .submit-status {{
      min-height: 1.2rem;
      margin-top: 14px;
      color: var(--accent-cool);
      font-weight: 700;
    }}
    .jobs-card {{
      padding: 18px;
      min-height: 70vh;
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 14px;
      position: sticky;
      top: 24px;
      max-height: calc(100vh - 48px);
    }}
    .jobs-card h2 {{
      margin: 0;
      font-family: "Palatino Linotype", "Book Antiqua", serif;
      font-size: 1.6rem;
    }}
    .jobs-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }}
    .jobs-head p {{
      margin: 4px 0 0;
      color: var(--ink-soft);
      line-height: 1.45;
    }}
    .secondary-button.is-muted {{
      background: rgba(167, 139, 250, 0.12);
      color: var(--ink-soft);
    }}
    .jobs-summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .jobs-summary .stat {{
      padding: 12px;
      border-radius: 16px;
      background: rgba(30, 26, 44, 0.84);
      border: 1px solid var(--line);
    }}
    .jobs-summary .stat strong {{
      display: block;
      font-size: 1.2rem;
      margin-bottom: 4px;
    }}
    .jobs-list {{
      display: grid;
      gap: 12px;
      align-content: start;
      overflow: auto;
      padding-right: 4px;
      min-height: 0;
    }}
    .job-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: var(--card-soft);
      display: grid;
      gap: 12px;
    }}
    .job-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      margin-bottom: 10px;
    }}
    .job-head h3 {{
      margin: 0 0 4px;
      font-size: 1rem;
    }}
    .job-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 8px 0 12px;
      color: var(--ink-soft);
      font-size: 0.84rem;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: white;
    }}
    .pill.queued {{ background: var(--queued); }}
    .pill.running {{ background: var(--running); }}
    .pill.succeeded {{ background: var(--success); }}
    .pill.failed {{ background: var(--failed); }}
    .pill.cancelled {{ background: var(--cancelled); }}
    code, pre {{
      font-family: "Consolas", "Cascadia Mono", monospace;
      font-size: 0.85rem;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(14, 12, 20, 0.78);
      border: 1px solid rgba(167, 139, 250, 0.12);
      border-radius: 14px;
      padding: 12px;
      max-height: 240px;
      overflow: auto;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .stats-chip {{
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(30, 26, 44, 0.84);
      border: 1px solid rgba(167, 139, 250, 0.12);
    }}
    .stats-chip span {{
      display: block;
      font-size: 0.74rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--ink-soft);
      margin-bottom: 4px;
    }}
    .stats-chip strong {{
      display: block;
      font-size: 1rem;
      line-height: 1.2;
    }}
    .progress-block {{
      display: grid;
      gap: 8px;
    }}
    .progress-meta {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 0.84rem;
      color: var(--ink-soft);
    }}
    .progress-bar {{
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(167, 139, 250, 0.12);
    }}
    .progress-fill {{
      height: 100%;
      width: 0%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent-cool) 0%, var(--accent) 100%);
      transition: width 180ms ease;
    }}
    .event-list {{
      display: grid;
      gap: 8px;
    }}
    .event-item {{
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(30, 26, 44, 0.72);
      color: var(--ink-soft);
      font-size: 0.84rem;
      line-height: 1.45;
    }}
    .status-line {{
      font-size: 0.92rem;
      color: var(--ink);
      font-weight: 700;
    }}
    .warning-note {{
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(251, 113, 133, 0.24);
      background: rgba(136, 19, 55, 0.2);
      color: var(--failed);
      font-size: 0.84rem;
      line-height: 1.45;
    }}
    details.command-box {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(22, 19, 31, 0.7);
      padding: 0 12px 12px;
    }}
    details.command-box summary {{
      cursor: pointer;
      padding: 12px 0;
      color: var(--ink-soft);
      font-weight: 700;
    }}
    .job-actions {{
      display: flex;
      gap: 8px;
      justify-content: flex-end;
      flex-wrap: wrap;
    }}
    .empty-state {{
      padding: 24px;
      border-radius: 18px;
      border: 1px dashed rgba(167, 139, 250, 0.24);
      background: rgba(22, 19, 31, 0.56);
      color: var(--ink-soft);
      text-align: center;
    }}
    @media (max-width: 1260px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .hero {{ grid-template-columns: 1fr; }}
      .jobs-card {{
        position: static;
        max-height: none;
      }}
    }}
    @media (max-width: 740px) {{
      .shell {{ width: min(100vw - 16px, 100%); margin-top: 8px; }}
      .field-grid, .jobs-summary, .stats-grid {{ grid-template-columns: 1fr; }}
      .panel-head, .job-head {{ flex-direction: column; }}
      .jobs-head {{ flex-direction: column; }}
      .tab-strip {{
        width: 100%;
        display: grid;
        grid-template-columns: 1fr 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-card">
        <h1>Run K-LM without babysitting the shell.</h1>
        <p>Queue train and infer jobs from one page, watch structured progress instead of raw diagnostics, and still drop down to CLI flags when the UI does not expose a knob yet.</p>
      </div>
      <aside class="hero-card meta-card">
        <div class="eyebrow">Project Root</div>
        <div class="meta-value">{html.escape(str(PROJECT_ROOT))}</div>
        <div class="eyebrow">Scheduler</div>
        <div class="meta-value">Single worker, FIFO queue, structured progress cards</div>
      </aside>
    </section>

    <div class="layout">
      <section class="panel control-panel">
        <div class="panel-head">
          <div>
            <h2>Run Settings</h2>
            <p>Switch between training and inference without losing the queued runs view on the right.</p>
          </div>
        </div>
        <div class="tab-strip" role="tablist" aria-label="Run mode">
          <button class="tab-button active" data-tab-target="train" type="button" role="tab" aria-selected="true">Train</button>
          <button class="tab-button" data-tab-target="infer" type="button" role="tab" aria-selected="false">Infer</button>
        </div>
        <div class="tab-panel active" data-tab-panel="train">{_render_form("train")}</div>
        <div class="tab-panel" data-tab-panel="infer">{_render_form("infer")}</div>
      </section>
      <section class="jobs-card">
        <div class="jobs-head">
          <div>
            <h2>Queued Runs</h2>
            <p>Live cards show progress, core metrics, and recent milestones. Full diagnostics still stream into the log file on disk.</p>
          </div>
          <button class="secondary-button" id="refresh-toggle" type="button">Pause live updates</button>
        </div>
        <div class="jobs-summary" id="jobs-summary">
          <div class="stat"><strong>0</strong><span>Total</span></div>
          <div class="stat"><strong>0</strong><span>Queued</span></div>
          <div class="stat"><strong>0</strong><span>Running</span></div>
          <div class="stat"><strong>0</strong><span>Finished</span></div>
        </div>
        <div class="jobs-list" id="jobs-list">
          <div class="empty-state">No jobs yet.</div>
        </div>
      </section>
    </div>
  </div>

  {_render_checkpoint_datalist()}

  <script>
    const jobList = document.getElementById("jobs-list");
    const summary = document.getElementById("jobs-summary");
    const refreshToggle = document.getElementById("refresh-toggle");
    const tabButtons = Array.from(document.querySelectorAll("[data-tab-target]"));
    const tabPanels = Array.from(document.querySelectorAll("[data-tab-panel]"));
    let liveRefresh = localStorage.getItem("k-lm-ui-live-refresh") !== "off";

    function setActiveTab(mode) {{
      const nextMode = ["train", "infer"].includes(mode) ? mode : "train";
      localStorage.setItem("k-lm-ui-active-tab", nextMode);
      for (const button of tabButtons) {{
        const active = button.dataset.tabTarget === nextMode;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", active ? "true" : "false");
      }}
      for (const panel of tabPanels) {{
        const active = panel.dataset.tabPanel === nextMode;
        panel.classList.toggle("active", active);
      }}
    }}

    function syncRefreshToggle() {{
      refreshToggle.textContent = liveRefresh ? "Pause live updates" : "Resume live updates";
      refreshToggle.classList.toggle("is-muted", !liveRefresh);
    }}

    function statusClass(status) {{
      return ["queued", "running", "succeeded", "failed", "cancelled"].includes(status) ? status : "queued";
    }}

    function prettyKey(key) {{
      const labels = {{
        train_ce: "Train CE",
        train_ce_ema: "Train CE EMA",
        train_bpc: "Train BPC",
        train_bpc_ema: "Train BPC EMA",
        val_ce: "Val CE",
        val_bpc: "Val BPC",
        val_ppl: "Val PPL",
        best_ppl: "Best PPL",
        best_perplexity: "Best PPL",
        lr: "LR",
        ms_per_step: "ms/step",
        tok_s: "tok/s",
        eval_tok_s: "Eval tok/s",
        sample_tok_s: "Sample tok/s",
        prompt_tokens: "Prompt toks",
        generated_tokens: "Gen toks",
        ckpt_best_ppl: "Ckpt best PPL",
        window: "Window",
        batch: "Batch",
        opt_mode: "Optimizer",
        fused_adamw: "Fused AdamW",
        grad_scaler: "Grad scaler",
        params_total: "Params",
        params_embedding: "Emb params",
        params_k_stack: "K-stack params",
        params_head: "Head params",
        params_other: "Other params",
      }};
      return labels[key] || key.replace(/_/g, " ");
    }}

    function metricEntries(job) {{
      const stats = job.summary_stats || {{}};
      const order = [
        "train_ce",
        "train_ce_ema",
        "train_bpc",
        "train_bpc_ema",
        "val_ce",
        "val_bpc",
        "val_ppl",
        "best_ppl",
        "best_perplexity",
        "lr",
        "ms_per_step",
        "tok_s",
        "eval_tok_s",
        "sample_tok_s",
        "generated_tokens",
        "prompt_tokens",
        "ckpt_best_ppl",
        "window",
        "batch",
        "opt_mode",
        "fused_adamw",
        "grad_scaler",
        "params_total",
      ];
      const seen = new Set();
      const entries = [];
      for (const key of order) {{
        if (stats[key] !== undefined && stats[key] !== null && stats[key] !== "") {{
          entries.push([key, stats[key]]);
          seen.add(key);
        }}
      }}
      for (const [key, value] of Object.entries(stats)) {{
        if (!seen.has(key) && value !== undefined && value !== null && value !== "") {{
          entries.push([key, value]);
        }}
      }}
      return entries;
    }}

    function renderStats(job) {{
      const entries = metricEntries(job);
      if (!entries.length) {{
        return '<div class="empty-state">Waiting for first structured metrics...</div>';
      }}
      return `
        <div class="stats-grid">
          ${{entries.map(([key, value]) => `
            <div class="stats-chip">
              <span>${{prettyKey(key)}}</span>
              <strong>${{value}}</strong>
            </div>
          `).join("")}}
        </div>
      `;
    }}

    function renderProgress(job) {{
      if (job.total_steps && job.current_step !== null && job.current_step !== undefined) {{
        const pct = Math.max(0, Math.min(100, (job.current_step / Math.max(job.total_steps, 1)) * 100));
        return `
          <div class="progress-block">
            <div class="progress-meta">
              <span>Progress</span>
              <span>${{job.current_step}} / ${{job.total_steps}} steps</span>
            </div>
            <div class="progress-bar"><div class="progress-fill" style="width:${{pct.toFixed(1)}}%"></div></div>
          </div>
        `;
      }}
      if (job.status === "queued") {{
        return `
          <div class="progress-block">
            <div class="progress-meta">
              <span>Queue</span>
              <span>${{job.queue_position ? `#${{job.queue_position}}` : "waiting"}}</span>
            </div>
            <div class="progress-bar"><div class="progress-fill" style="width:0%"></div></div>
          </div>
        `;
      }}
      return "";
    }}

    function renderRuntime(job) {{
      const runtime = job.runtime_info || {{}};
      const parts = [];
      for (const key of ["device", "amp", "seed", "strict_repro", "deterministic", "tf32"]) {{
        if (runtime[key] !== undefined && runtime[key] !== null && runtime[key] !== "") {{
          parts.push(`${{prettyKey(key)}}: ${{runtime[key]}}`);
        }}
      }}
      return parts.length ? `<div class="job-meta">${{parts.map((part) => `<span>${{part}}</span>`).join("")}}</div>` : "";
    }}

    function renderEvents(job) {{
      const events = (job.recent_events || []).slice(-5);
      if (!events.length) {{
        return "";
      }}
      return `
        <div class="event-list">
          ${{events.map((event) => `<div class="event-item">${{event}}</div>`).join("")}}
        </div>
      `;
    }}

    function renderWarning(job) {{
      if (!job.warning_count) {{
        return "";
      }}
      const last = job.last_warning ? `<div>${{job.last_warning}}</div>` : "";
      return `
        <div class="warning-note">
          <strong>${{job.warning_count}} warning${{job.warning_count === 1 ? "" : "s"}}</strong>
          ${{last}}
        </div>
      `;
    }}

    function summarizeJobs(jobs) {{
      const totals = {{
        total: jobs.length,
        queued: jobs.filter((job) => job.status === "queued").length,
        running: jobs.filter((job) => job.status === "running").length,
        finished: jobs.filter((job) => ["succeeded", "failed", "cancelled"].includes(job.status)).length,
      }};
      summary.innerHTML = `
        <div class="stat"><strong>${{totals.total}}</strong><span>Total</span></div>
        <div class="stat"><strong>${{totals.queued}}</strong><span>Queued</span></div>
        <div class="stat"><strong>${{totals.running}}</strong><span>Running</span></div>
        <div class="stat"><strong>${{totals.finished}}</strong><span>Finished</span></div>
      `;
    }}

    function renderJobs(jobs) {{
      summarizeJobs(jobs);
      if (!jobs.length) {{
        jobList.innerHTML = '<div class="empty-state">No jobs yet.</div>';
        return;
      }}

      jobList.innerHTML = "";
      for (const job of jobs) {{
        const article = document.createElement("article");
        article.className = "job-card";
        const queueLine = job.queue_position ? `queue #${{job.queue_position}}` : "";
        const rcLine = job.returncode === null || job.returncode === undefined ? "" : `rc=${{job.returncode}}`;
        const errorLine = job.error ? `<div class="field-help">${{job.error}}</div>` : "";
        const phaseLabel = job.phase ? job.phase.replace(/_/g, " ") : job.status;
        const cancelButton = ["queued", "running"].includes(job.status)
          ? `<button class="secondary-button" data-cancel="${{job.job_id}}">Cancel</button>`
          : "";
        article.innerHTML = `
          <div class="job-head">
            <div>
              <h3>${{job.name}}</h3>
              <div class="job-meta">
                <span class="pill ${{statusClass(job.status)}}">${{job.status}}</span>
                <span>${{job.mode}}</span>
                <span>${{phaseLabel}}</span>
                <span>${{job.created_at || ""}}</span>
                <span>${{queueLine}}</span>
                <span>${{rcLine}}</span>
              </div>
            </div>
            <div class="job-actions">
              <button class="secondary-button" data-copy="${{job.job_id}}">Copy Cmd</button>
              ${{cancelButton}}
            </div>
          </div>
          <div class="status-line">${{job.last_message || "Waiting for process output..."}}</div>
          ${{renderProgress(job)}}
          ${{renderRuntime(job)}}
          <div class="job-meta">
            <span>log: ${{job.log_path}}</span>
            <span>log lines: ${{job.log_line_count}}</span>
            <span>started: ${{job.started_at || "-"}}</span>
            <span>finished: ${{job.finished_at || "-"}}</span>
          </div>
          ${{errorLine}}
          ${{renderWarning(job)}}
          ${{renderStats(job)}}
          ${{renderEvents(job)}}
          <details class="command-box">
            <summary>Run command</summary>
            <pre>${{job.command_display}}</pre>
          </details>
        `;
        jobList.appendChild(article);
      }}

      for (const button of document.querySelectorAll("[data-copy]")) {{
        button.onclick = async () => {{
          const job = jobs.find((item) => item.job_id === button.dataset.copy);
          if (!job) return;
          await navigator.clipboard.writeText(job.command_display);
        }};
      }}
      for (const button of document.querySelectorAll("[data-cancel]")) {{
        button.onclick = async () => {{
          await fetch(`/api/jobs/${{button.dataset.cancel}}/cancel`, {{ method: "POST" }});
          await refreshJobs(true);
        }};
      }}
    }}

    async function refreshJobs(force = false) {{
      if (!force && !liveRefresh) {{
        return;
      }}
      const response = await fetch("/api/jobs");
      const jobs = await response.json();
      renderJobs(jobs);
    }}

    async function submitForm(event) {{
      event.preventDefault();
      const form = event.currentTarget;
      const formData = new FormData(form);
      const mode = form.getAttribute("data-mode") || form.dataset.mode || formData.get("__mode");
      const statusNode = form.querySelector(".submit-status");
      if (!mode) {{
        statusNode.textContent = "Missing run mode on form.";
        return;
      }}
      statusNode.textContent = "Queueing...";
      const body = new URLSearchParams(formData);
      const response = await fetch(`/api/jobs/${{mode}}`, {{
        method: "POST",
        headers: {{ "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8" }},
        body,
      }});
      const payload = await response.json();
      if (!response.ok) {{
        statusNode.textContent = payload.detail || "Request failed.";
        return;
      }}
      statusNode.textContent = `Queued ${{payload.name}} as ${{payload.job_id}}.`;
      await refreshJobs(true);
    }}

    for (const form of document.querySelectorAll(".queue-form")) {{
      form.addEventListener("submit", submitForm);
    }}
    for (const button of tabButtons) {{
      button.addEventListener("click", () => setActiveTab(button.dataset.tabTarget));
    }}
    refreshToggle.addEventListener("click", async () => {{
      liveRefresh = !liveRefresh;
      localStorage.setItem("k-lm-ui-live-refresh", liveRefresh ? "on" : "off");
      syncRefreshToggle();
      if (liveRefresh) {{
        await refreshJobs(true);
      }}
    }});

    setActiveTab(localStorage.getItem("k-lm-ui-active-tab") || "train");
    syncRefreshToggle();
    refreshJobs(true);
    setInterval(refreshJobs, 1500);
  </script>
</body>
</html>"""


def create_app():
    _require_fastapi()
    scheduler = RunScheduler()
    app = FastAPI(title="K-LM UI", version="0.1.0")

    @app.on_event("shutdown")
    def _shutdown_scheduler() -> None:
        scheduler.shutdown()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_build_page())

    @app.get("/api/jobs")
    async def jobs() -> JSONResponse:
        return JSONResponse(scheduler.list_jobs())

    @app.get("/api/jobs/{job_id}")
    async def job(job_id: str) -> JSONResponse:
        try:
            return JSONResponse(scheduler.get_job(job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc

    @app.post("/api/jobs/{mode}")
    async def submit_job(mode: str, request: Request) -> JSONResponse:
        if mode not in FORM_SECTIONS_BY_MODE:
            raise HTTPException(status_code=404, detail=f"Unknown mode: {mode}")
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = await request.json()
        elif "application/x-www-form-urlencoded" in content_type or not content_type:
            raw_body = (await request.body()).decode("utf-8")
            payload = {
                key: values[-1]
                for key, values in urllib.parse.parse_qs(raw_body, keep_blank_values=True).items()
            }
        else:
            raise HTTPException(
                status_code=415,
                detail="Unsupported content type. Use JSON or application/x-www-form-urlencoded.",
            )
        try:
            job = scheduler.submit(mode, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(job, status_code=201)

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> JSONResponse:
        try:
            return JSONResponse(scheduler.cancel(job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc

    @app.get("/api/checkpoints")
    async def checkpoints() -> JSONResponse:
        return JSONResponse(list_checkpoint_paths())

    @app.get("/healthz")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    return app


def main() -> None:
    _require_fastapi()
    parser = argparse.ArgumentParser(description="Run the local FastAPI queue UI for K-LM.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "uvicorn is not installed. Run `pip install -e .[ui]` or `pip install 'uvicorn[standard]'`."
        ) from exc

    uvicorn.run(create_app(), host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
