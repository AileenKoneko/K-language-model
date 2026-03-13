# K-Operators: K-Stack Character Language Model

This directory contains a self-contained character-level language model built around a custom K-Stack backbone.  
It supports:

- training on Tiny Shakespeare,
- deterministic evaluation (cross-entropy/perplexity),
- checkpoint save/resume/load,
- text sampling with top-k/top-p/repetition controls,
- optional `torch.compile`,
- strict reproducibility mode.

## Results

Latest deterministic evals in this repo (run on March 13, 2026):

- Flags: `--deterministic --no-tf32 --skip-sample`
- Sweep architecture flags: `--d-model 128 --window 256 --n-k2 6 --head-mode gelu --head-mult 1 --rank 32 --refine-steps 1`
- Checkpoint directory: `data/20260313_133317_26319`

| Checkpoint | Step | Val CE | Val PPL |
|---|---:|---:|---:|
| `run_1_seed_42.pt` | 3250 | 1.4838 | 4.41 |
| `run_2_seed_42.pt` | 3250 | 1.4811 | 4.40 |
| `run_3_seed_42.pt` | 2750 | 1.4876 | 4.43 |
| `run_4_seed_69.pt` | 3000 | 1.5046 | 4.50 |
| `run_5_seed_420.pt` | 3500 | 1.4878 | 4.43 |
| `run_6_seed_666.pt` | 3250 | 1.4980 | 4.47 |
| `run_7_seed_2137.pt` | 3250 | 1.4733 | 4.36 |

Aggregate over the 7-run sweep:

- Mean val CE: `1.4880` (std `0.0097`)
- Mean val PPL: `4.4286` (std `0.0426`)
- Best checkpoint: `run_7_seed_2137.pt` with `val_ce=1.4733`, `val_ppl=4.36`

Reference checkpoint (`models/char_shakespeare.pt`) deterministic eval with matching architecture:

- `val_ce=1.4773`
- `val_ppl=4.38`

Reproduce one eval:

```bash
python infer.py \
  --ckpt data/20260313_133317_26319/run_7_seed_2137.pt \
  --deterministic --no-tf32 --skip-sample \
  --d-model 128 --window 256 --n-k2 6 \
  --head-mode gelu --head-mult 1 --rank 32 --refine-steps 1
```

Note: TF32 affects CUDA kernels only. On CPU/MPS runs, `--no-tf32` has no practical effect.

## Contents

- [Results](#results)
- [Project Layout](#project-layout)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation and Inference](#evaluation-and-inference)
- [Sampling Controls](#sampling-controls)
- [Model Architecture](#model-architecture)
- [Reproducibility and Runtime Behavior](#reproducibility-and-runtime-behavior)
- [Checkpoints](#checkpoints)
- [CLI Reference](#cli-reference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Layout

```text
k_operators/
├── train.py                      # Main training entrypoint
├── infer.py                      # Main inference/eval entrypoint
├── k_lm.py                       # Backward-compatible alias to train.py
├── k_language_model/
│   ├── train_app.py              # Training CLI and orchestration
│   ├── infer_app.py              # Inference/eval CLI and orchestration
│   ├── trainer.py                # Train loop, eval loop, optimizer/scheduler
│   ├── model.py                  # K-Stack model definition
│   ├── generation.py             # Text sampling implementation
│   ├── checkpoint.py             # Save/load logic (model + optimizer)
│   ├── data.py                   # Tiny Shakespeare download + tokenization
│   └── runtime.py                # Device, AMP, logging, reproducibility
├── data/
│   └── input.txt                 # Tiny Shakespeare dataset cache
├── models/
│   └── char_shakespeare.pt       # Example trained checkpoint
└── configs/
    └── colab_seed_sweep.sh       # Optional Colab seed sweep script
```

## Quick Start

### 1) Environment

Minimum runtime dependencies (from imports):

- Python 3.10+
- PyTorch
- NumPy

Example:

```bash
cd k_operators
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- The code picks device automatically: `cuda` -> `mps` -> `cpu`.
- AMP is enabled on CUDA only.

Optional editable install (enables `k-lm-train` / `k-lm-infer` console commands):

```bash
pip install -e .
```

### 2) First training run

```bash
cd k_operators
python train.py \
  --steps 2000 \
  --eval-interval 200 \
  --ckpt models/char_shakespeare.pt \
  --sample
```

What happens:

- Tiny Shakespeare is auto-downloaded to `data/input.txt` if missing.
- Training runs and logs train CE / val CE / val PPL regularly.
- Checkpoint is saved only when validation perplexity improves.
- A text sample is printed at the end because `--sample` is enabled.

## Training

`train.py` is the main entrypoint. It handles:

- full training loop,
- optional checkpoint resume,
- periodic deterministic eval,
- optional post-training sample generation.

### Resume training from checkpoint

If `--ckpt` points to an existing file, training resumes from it:

```bash
python train.py --ckpt models/char_shakespeare.pt
```

### Evaluate only (no training)

```bash
python train.py \
  --eval-only \
  --ckpt models/char_shakespeare.pt
```

Optional: override eval-time refinement depth:

```bash
python train.py \
  --eval-only \
  --ckpt models/char_shakespeare.pt \
  --eval-refine-steps 2
```

### Useful training flags

- Model shape: `--window`, `--d-model`, `--rank`, `--n-k2`, `--head-mode`, `--head-mult`
- Refinement behavior: `--refine-steps`, `--train-refine-steps`, `--alpha-cap`, `--decay-impl`
- Optimization: `--lr`, `--lr-floor`, `--warmup-steps`, `--weight-decay`, `--optimizer-mode`
- Regularization: `--emb-dropout`, `--mlp-dropout`, `--residual-dropout`, `--head-dropout`
- CUDA perf: `--fused-adamw`, `--compile`, `--compile-mode`
- Reproducibility: `--seed`, `--deterministic`, `--strict-repro`, `--run-manifest`
- Extra logging: `--diagnostics` for verbose research/debug stats

## Evaluation and Inference

Use `infer.py` to load a checkpoint and run deterministic eval and/or sampling.

### Default inference flow (eval + sample)

```bash
python infer.py --ckpt models/char_shakespeare.pt
```

### Sample only (skip eval)

```bash
python infer.py \
  --ckpt models/char_shakespeare.pt \
  --skip-eval \
  --prompt "ROMEO:" \
  --sample-tokens 300
```

### Eval only (skip sample)

```bash
python infer.py \
  --ckpt models/char_shakespeare.pt \
  --skip-sample
```

Important: the architecture arguments (`--window`, `--d-model`, `--rank`, `--n-k2`, head/refinement flags) must match the checkpoint topology, or you will get missing/unexpected key warnings and degraded/invalid behavior.

## Sampling Controls

Both training (`--sample`) and inference support the same generation controls:

- `--prompt`: seed text
- `--sample-tokens`: number of generated chars
- `--temperature`: logits temperature (lower = more deterministic)
- `--top-k`: keep top-k logits (0 disables)
- `--top-p`: nucleus sampling threshold in `(0, 1)` (0 disables)
- `--repetition-penalty`: >1 discourages repeated tokens
- `--repetition-window`: lookback window for repetition penalty (`0` = full context)
- `--prompt-lock-chars`: keep first N prompt chars in long-context conditioning

## Model Architecture

The model is defined in `k_language_model/model.py`.

High-level structure:

1. Character embedding
2. K-Stack backbone:
   - `K1` block
   - `n_k2` x `K2` blocks
   - `K1` block
   - `K0` block
3. Final RMSNorm
4. Output head:
   - `linear` head (tied embedding weights), or
   - `gelu` MLP head

### K2 block summary

Each K2 layer mixes sequence information with two components:

- learned causal base kernel (`k_base`),
- low-rank decayed recurrent interaction (`u`, `v`, `decay_logit`, `alpha_logit`).

Two decay implementations are available:

- `mask` (default): fastest, more memory-heavy
- `block`: lower memory, uses blocked scan

### Iterative refinement

At each forward pass (unless refinement steps are set to `0`), hidden states are iteratively updated:

```text
h <- h + eta * (KStack(h) - h)
```

- `eta` is learned (`eta_logit`).
- `--train-refine-steps` controls training-time iterations.
- `--refine-steps` controls eval/inference iterations.

## Reproducibility and Runtime Behavior

Runtime behavior is managed in `k_language_model/runtime.py`.

- Device selection: CUDA first, then MPS, then CPU.
- AMP: enabled on CUDA (`bfloat16` autocast path).
- Default mode favors speed, not strict determinism.

### Determinism controls

- `--deterministic`: enables deterministic algorithms and deterministic cuDNN behavior.
- `--deterministic-warn-only`: warn instead of error for nondeterministic ops.
- `--no-tf32`: disables TF32.
- `--strict-repro`: strongest reproducibility mode; forces deterministic settings, disables TF32, disables compile, and disables fused AdamW.

### Run manifest

Use `--run-manifest path/to/manifest.json` to persist:

- command line,
- hashed config signature,
- full args,
- runtime metadata (Python, platform, torch/cuda/cudnn, device flags).

## Checkpoints

Checkpoint logic is in `k_language_model/checkpoint.py`.

Saved format:

```python
{
  "step": int,
  "best_ppl": float,
  "model": model.state_dict(),
  "optimizer": optimizer.state_dict(),
  "rng_state": {
    "torch_cpu": ...,
    "torch_cuda": ...,
    "numpy": ...,
    "python": ...,
  },
}
```

Behavior:

- Training resume (`load_checkpoint`) restores model + optimizer when available.
- Training resume restores RNG state when available, improving reproducibility after interruptions.
- Inference load (`load_model_checkpoint`) restores model only.
- Checkpoint loader normalizes some prefixes (`_orig_mod.`, `module.`) for compatibility.
- State-dict load uses `strict=False` and logs missing/unexpected keys.
- Shape mismatches (for example wrong `d-model`/`window`/`n-k2`) still raise a hard load error.

## CLI Reference

Use full help for exact defaults and descriptions:

```bash
python train.py --help
python infer.py --help
```

### `train.py` (selected high-impact args)

- Core: `--steps`, `--batch-size`, `--eval-interval`, `--ckpt`
- Model: `--window`, `--d-model`, `--rank`, `--n-k2`, `--head-mode`, `--head-mult`, `--head-dropout`
- Refinement/decay: `--refine-steps`, `--train-refine-steps`, `--alpha-cap`, `--decay-impl`
- Optimizer/schedule: `--lr`, `--lr-floor`, `--warmup-steps`, `--beta1`, `--beta2`, `--weight-decay`, `--optimizer-mode`
- Perf: `--fused-adamw` / `--no-fused-adamw`, `--compile`, `--compile-mode`
- Repro: `--seed`, `--deterministic`, `--deterministic-warn-only`, `--no-tf32`, `--strict-repro`, `--run-manifest`
- Eval-only path: `--eval-only`, `--eval-refine-steps`
- Sampling: `--sample`, `--prompt`, `--sample-tokens`, `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--repetition-window`, `--prompt-lock-chars`

### `infer.py` (selected high-impact args)

- Required: `--ckpt`
- Eval: `--batch-size`, `--skip-eval`
- Sampling: `--skip-sample`, `--prompt`, `--sample-tokens`, `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--repetition-window`, `--prompt-lock-chars`
- Architecture compatibility: `--window`, `--d-model`, `--rank`, `--n-k2`, `--head-*`, `--refine-steps`, `--train-refine-steps`, `--eval-refine-steps`, `--alpha-cap`, `--decay-impl`
- Runtime/repro: `--compile`, `--compile-mode`, `--seed`, `--deterministic`, `--strict-repro`

## Troubleshooting

### Out-of-memory (OOM)

Reduce one or more of:

- `--batch-size`
- `--window`
- `--d-model`
- `--n-k2`

If needed, switch decay backend to lower-memory mode:

```bash
--decay-impl block
```

### Checkpoint loads with many missing/unexpected keys

Most likely architecture mismatch. Ensure the checkpoint is loaded with the same:

- `window`, `d-model`, `rank`, `n-k2`
- head mode/mult/dropout
- refine/decode related flags as required for shape compatibility

### Reproducibility drift between runs

Use:

```bash
--strict-repro --seed <fixed_seed>
```

and avoid changing runtime stack (PyTorch/CUDA version, hardware, driver).

### NumPy 2 compatibility warning

If you see a warning about modules compiled for NumPy 1.x not running under NumPy 2.x, reinstall with:

```bash
pip install -r requirements.txt
```

This project pins `numpy<2` to stay compatible with common PyTorch builds.

### No checkpoint file created

A checkpoint is written only when:

- `--ckpt` is provided, and
- validation perplexity improves at an evaluation step.

## License

This project uses the MIT License. See `LICENSE`.
