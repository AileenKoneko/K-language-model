# K-Operators: K-Stack Language Model

This directory contains a self-contained language model built around a custom K-Stack backbone.  
It supports:

- training on Tiny Shakespeare or WikiText-2,
- character or SentencePiece tokenization,
- factorized token embeddings with `--emb-dim`,
- adaptive softmax output heads for larger vocabularies,
- deterministic evaluation (cross-entropy/perplexity),
- checkpoint save/resume/load,
- text sampling with top-k/top-p/repetition controls,
- optional `torch.compile`,
- strict reproducibility mode.

Archive: [Zenodo record 19004569](https://zenodo.org/records/19004569)

## Results

WikiText-2 SentencePiece checkpoint evals (run on March 15, 2026):

- Dataset/tokenizer: `--dataset wikitext2 --tokenizer sentencepiece --sp-model data/tokenizers/wikitext2_unigram_8192.model --sp-vocab-size 8192`
- Shared head/refinement flags: `--head-mode adaptive --adaptive-cutoffs 1024,4096 --adaptive-div-value 4 --emb-dim 64 --refine-steps 1`
- Eval note: reevaluated on `cuda` with `decay_impl=block` for lower eval memory. Batch size was `8` for `window=1024` and `32` otherwise.

| Step | Window | d-model | emb-dim | n-k2 | Rank | Params | Val CE | Val PPL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20000 | 512 | 128 | 64 | 8 | 32 | 3,790,353 | 3.1059 | 22.33 |
| 7500 | 512 | 256 | 64 | 6 | 32 | 5,392,077 | 3.1148 | 22.53 |
| 19750 | 512 | 128 | 64 | 6 | 32 | 3,084,237 | 3.1204 | 22.66 |
| 19500 | 512 | 128 | 64 | 6 | 32 | 3,084,237 | 3.1344 | 22.97 |
| 20000 | 1024 | 128 | 64 | 6 | 32 | 7,802,829 | 3.1452 | 23.22 |
| 17250 | 512 | 128 | 64 | 6 | 32 | 3,084,237 | 3.1458 | 23.24 |
| 11750 | 512 | 128 | 64 | 6 | 64 | 3,133,581 | 3.1563 | 23.48 |
| 5500 | 512 | 256 | 64 | 6 | 32 | 5,392,077 | 3.1601 | 23.57 |
| 14750 | 512 | 128 | 64 | 6 | 32 | 3,084,237 | 3.1622 | 23.62 |
| 6500 | 512 | 128 | 64 | 6 | 64 | 3,133,581 | 3.2136 | 24.87 |
| 7500 | 512 | 128 | 64 | 6 | 32 | 3,084,237 | 3.2190 | 25.00 |
| 5750 | 512 | 128 | 64 | 6 | 32 | 3,084,237 | 3.2565 | 25.96 |

Synthetic denoising benchmark (run on March 15, 2026):

- Script/flags: `python bench_denoise.py --backbone both --steps 500 --deterministic --no-tf32`
- Task: synthetic motif-and-copy token denoising with loss and accuracy measured only on corrupted positions.
- Takeaway: the single `K2Layer` beats the single `Conv1d` block on denoising quality at lower parameter count, but `Conv1d` remains faster.

| Mixer | Params | Val CE | Val Acc | tok/s |
|---|---:|---:|---:|---:|
| `Conv1d` | 12,528 | 2.7190 | 0.1376 | 324,310 |
| `K2` | 11,034 | 2.0492 | 0.3526 | 192,995 |


Post pre-print update (March 14th 2026):

- Preliminary results on character-level WikiText-2. Hyperparameter sweeps and ablations are pending. Before moving to sentencepiece-level, I want to ensure the model is well-understood and tuned at the char level.
- Shared flags: `--dataset wikitext2 --deterministic --no-tf32 --skip-sample --n-k2 6 --head-mode gelu --head-mult 1 --rank 32 --refine-steps 1`
- Runtime/device: `mps`

| Checkpoint                   | Step | Val BPC | Val PPL | Window | d-model | Params | Run hash |
|------------------------------|---:|---:|---:|---:|---:|---:|---|
| `models/wiki_256_256_10k.pt` | 10000 | 1.4978 | 2.82 | 256 | 256 | 3,337,421 | `9226ddba588c` |
| `models/wiki_256_256_20k.pt` | 20000 | 1.4535 | 2.74 | 256 | 256 | 3,337,421 | `9ef711bf3282` |
| `models/wiki_512_256_20k.pt` | 20000 | 1.4208 | 2.68 | 512 | 256 | 4,517,069 | `d67d459388a0` |
| `models/wiki_512_128_20k.pt` | 20000 | 1.5306 | 2.89 | 512 | 128 | 2,373,325 | `5d6d1f245439` |

Tiny Shakespeare seed-sweep deterministic evals (run on March 13, 2026):

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
│   ├── data.py                   # Dataset download/loading + char/SentencePiece tokenization
│   └── runtime.py                # Device, AMP, logging, reproducibility
├── data/
│   ├── input.txt                 # Tiny Shakespeare dataset cache
│   └── wikitext-2/               # WikiText-2 cache (auto-downloaded when used)
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
- SentencePiece

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

Synthetic denoising benchmark (single `Conv1d` block vs single `K2Layer`):

```bash
python bench_denoise.py \
  --backbone both \
  --steps 300
```

This benchmark is intentionally small and synthetic. It generates motif-and-copy token sequences, corrupts spans/tokens, and trains both models to reconstruct the original tokens only at corrupted positions.

The defaults are tuned to be easy enough to learn quickly. To stress longer-range denoising, raise `--period-max`, `--burst-count`, `--burst-max-len`, or `--seq-len`.

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

- Dataset/input: `--dataset {shakespeare,wikitext2,wikitext2_raw}`, `--data-path`, `--val-path`, `--val-frac`, `--tokenizer`
- SentencePiece: `--sp-model`, `--sp-vocab-size`, `--sp-model-type`, `--sp-character-coverage`, `--sp-split-digits`, `--sp-byte-fallback`
- Model shape: `--window`, `--d-model`, `--emb-dim`, `--rank`, `--n-k2`, `--head-mode`, `--head-mult`
- Adaptive head: `--adaptive-cutoffs`, `--adaptive-div-value`
- Refinement behavior: `--refine-steps`, `--train-refine-steps`, `--alpha-cap`, `--decay-impl`
- Optimization: `--lr`, `--lr-floor`, `--warmup-steps`, `--weight-decay`, `--optimizer-mode`
- Regularization: `--emb-dropout`, `--mlp-dropout`, `--residual-dropout`, `--head-dropout`
- CUDA perf: `--fused-adamw`, `--compile`, `--compile-mode`
- Reproducibility: `--seed`, `--deterministic`, `--strict-repro`, `--run-manifest`
- Extra logging: `--diagnostics` for verbose research/debug stats

### WikiText-2 training

Built-in WikiText-2 preset:

```bash
python train.py \
  --dataset wikitext2 \
  --steps 3500 \
  --window 256 \
  --d-model 128 \
  --n-k2 6 \
  --rank 32 \
  --head-mode gelu \
  --head-mult 1 \
  --refine-steps 1 \
  --ckpt models/wiki_text2.pt
```

Raw WikiText-2 preset (uses `wiki.train.raw` / `wiki.valid.raw` and avoids the literal `<unk>` artifacts present in the processed split):

```bash
python train.py \
  --dataset wikitext2_raw \
  --steps 3500 \
  --window 256 \
  --d-model 128 \
  --n-k2 6 \
  --rank 32 \
  --head-mode gelu \
  --head-mult 1 \
  --refine-steps 1 \
  --ckpt models/wiki_text2_raw.pt
```

SentencePiece + adaptive softmax preset:

```bash
python train.py \
  --dataset wikitext2 \
  --tokenizer sentencepiece \
  --sp-model data/tokenizers/wikitext2_unigram_8192.model \
  --sp-vocab-size 8192 \
  --window 256 \
  --d-model 256 \
  --emb-dim 64 \
  --n-k2 6 \
  --rank 32 \
  --head-mode adaptive \
  --adaptive-cutoffs 1024,4096 \
  --refine-steps 1 \
  --ckpt models/wiki_text2_sp_adaptive.pt
```

Notes:

- If `--sp-model` does not exist during training, it is trained automatically from the training split and written under `data/tokenizers/`.
- When `--head-mode adaptive` is enabled, token ids are remapped by train-set frequency so the adaptive shortlist receives the most common tokens first.

Custom text files:

```bash
python train.py \
  --dataset wikitext2 \
  --data-path /path/to/train.txt \
  --val-path /path/to/valid.txt \
  --ckpt models/wiki_text2_custom.pt
```

Colab sweep script supports dataset env vars too:

```bash
DATASET="wikitext2" \
TRAIN_DATA_PATH="/content/data/wiki_train.txt" \
VAL_DATA_PATH="/content/data/wiki_valid.txt" \
SCRIPT_PATH="train.py" \
bash configs/colab_seed_sweep.sh
```

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

Important: dataset + tokenizer context must match the checkpoint.

- Architecture arguments (`--window`, `--d-model`, `--emb-dim`, `--rank`, `--n-k2`, head/refinement flags) must match.
- Dataset/tokenizer arguments (`--dataset`, `--data-path`, `--val-path`, `--val-frac`, `--tokenizer`, SentencePiece flags) should match what was used when training the checkpoint.
- Reporting mode: WikiText-2 logs use BPC only for char-tokenized runs. SentencePiece runs report token CE and token PPL.

## Sampling Controls

Both training (`--sample`) and inference support the same generation controls:

- `--prompt`: seed text
- `--sample-tokens`: number of generated tokens
- `--temperature`: logits temperature (lower = more deterministic)
- `--top-k`: keep top-k logits (0 disables)
- `--top-p`: nucleus sampling threshold in `(0, 1)` (0 disables)
- `--repetition-penalty`: >1 discourages repeated tokens
- `--repetition-window`: lookback window for repetition penalty (`0` = full context)
- `--prompt-lock-tokens`: keep first N prompt tokens in long-context conditioning

## Model Architecture

The model is defined in `k_language_model/model.py`.

High-level structure:

1. Token embedding (`--emb-dim` can be smaller than `--d-model`)
2. K-Stack backbone:
   - `K1` block
   - `n_k2` x `K2` blocks
   - `K1` block
   - `K0` block
3. Final RMSNorm
4. Output head:
   - `linear` head (tied embedding weights, optionally through a projection when `--emb-dim != --d-model`),
   - `gelu` MLP head, or
   - `adaptive` softmax head for larger vocabularies

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
- Dataset: `--dataset`, `--data-path`, `--val-path`, `--val-frac`, `--tokenizer`
- SentencePiece: `--sp-model`, `--sp-vocab-size`, `--sp-model-type`, `--sp-character-coverage`, `--sp-split-digits`, `--sp-byte-fallback`
- Model: `--window`, `--d-model`, `--emb-dim`, `--rank`, `--n-k2`, `--head-mode`, `--head-mult`, `--head-dropout`
- Adaptive: `--adaptive-cutoffs`, `--adaptive-div-value`
- Refinement/decay: `--refine-steps`, `--train-refine-steps`, `--alpha-cap`, `--decay-impl`
- Optimizer/schedule: `--lr`, `--lr-floor`, `--warmup-steps`, `--beta1`, `--beta2`, `--weight-decay`, `--optimizer-mode`
- Perf: `--fused-adamw` / `--no-fused-adamw`, `--compile`, `--compile-mode`
- Repro: `--seed`, `--deterministic`, `--deterministic-warn-only`, `--no-tf32`, `--strict-repro`, `--run-manifest`
- Eval-only path: `--eval-only`, `--eval-refine-steps`
- Sampling: `--sample`, `--prompt`, `--sample-tokens`, `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--repetition-window`, `--prompt-lock-tokens`

### `infer.py` (selected high-impact args)

- Required: `--ckpt`
- Dataset: `--dataset`, `--data-path`, `--val-path`, `--val-frac`, `--tokenizer`
- SentencePiece: `--sp-model`, `--sp-vocab-size`, `--sp-model-type`, `--sp-character-coverage`, `--sp-split-digits`, `--sp-byte-fallback`
- Eval: `--batch-size`, `--skip-eval`
- Sampling: `--skip-sample`, `--prompt`, `--sample-tokens`, `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--repetition-window`, `--prompt-lock-tokens`
- Architecture compatibility: `--window`, `--d-model`, `--emb-dim`, `--rank`, `--n-k2`, `--head-*`, `--adaptive-*`, `--refine-steps`, `--train-refine-steps`, `--eval-refine-steps`, `--alpha-cap`, `--decay-impl`
- Runtime/repro: `--compile`, `--compile-mode`, `--seed`, `--deterministic`, `--strict-repro`

## Troubleshooting

### Out-of-memory (OOM)

Reduce one or more of:

- `--batch-size`
- `--window`
- `--d-model`
- `--emb-dim`
- `--n-k2`

If needed, switch decay backend to lower-memory mode:

```bash
--decay-impl block
```

### `torch.compile` fails with `TritonMissing`

On Windows CUDA setups, `torch.compile` may be unavailable because there is no working Triton package for the environment. The CLI now warns and falls back to eager mode instead of crashing.

If you want the old behavior explicitly, just omit `--compile`.

### Checkpoint loads with many missing/unexpected keys

Most likely architecture mismatch. Ensure the checkpoint is loaded with the same:

- `window`, `d-model`, `emb-dim`, `rank`, `n-k2`
- tokenizer settings, especially SentencePiece model path and vocab size
- head mode/mult/dropout and adaptive cutoff settings when applicable
- refine/decode related flags as required for shape compatibility
- dataset and text sources (`--dataset`, `--data-path`, `--val-path`, `--val-frac`) to keep vocab consistent

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
