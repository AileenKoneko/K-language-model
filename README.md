# K-Operators: K-Stack Language Model (V2)

This branch is the current V2 line of development for the K-Stack language model.

It is no longer trying to preserve the old V1 architecture:

- no equilibrium / refinement loop
- no V1 checkpoint compatibility
- one active model family built around the single-pass K-Stack
- modular registries for datasets, tokenizers, heads, decay backends, `k_base` backends, and ROSA backends

Historical branch material:

- [README_v1.md](README_v1.md)
- [RESULTS_V1.md](RESULTS_V1.md)

## Results

Current V2 snapshots
All runs performed on a single A100 GPU with the same training loop and hyperparameters, except for the `rosa_impl` using exact backend and no C++ formulation, and tokenizer variations. Final eval score reported from M1 Pro run with `deterministic=true, strict_repro=true`.
Shared flags: `window=2048, d_model=64, n_k2=6, rank=4, share_k_base, k_base_kernel_size=8, decay_impl=mask, batch_size=32, eval_interval=250, lr=1e-2, gamma_min=0.05, alpha_cap=1.0, steps=5000, dataset=shakespeare, seed=42`
For byte tokenizer, additionally: `emb_dim=64`

| Dataset          | Tokenizer | ROSA     | Checkpoint                                                   | Step |    Params |     Val CE |  Val PPL |        Throughput |
|------------------|-----------|----------|--------------------------------------------------------------|-----:|----------:|-----------:|---------:|------------------:|
| Tiny Shakespeare | char      | ON       | `models/tiny-shakespeare/v2/char_shakespeare_v2.pt`          | 5000 | `178,692` | **1.4028** | **4.07** |   `328,828 tok/s` |
| Tiny Shakespeare | char      | OFF      | `models/tiny-shakespeare/v2/char_shakespeare_v2_rosa_off.pt` | 5000 | `174,768` | **1.4671** | **4.34** | `2,758,557 tok/s` |
| Tiny Shakespeare | char      | ON (C++) | `models/tiny-shakespeare/v2/char_shakespeare_v2_rosa_cpp.pt` | 5000 | `178,692` | **1.3954** | **4.04** |   `749,646 tok/s` |
| Tiny Shakespeare | byte      | ON       | `models/tiny-shakespeare/v2/byte_shakespeare_v2.pt`          | 5000 | `203,140` | **1.4072** | **4.08** |   `330,524 tok/s` |
| Tiny Shakespeare | byte      | OFF      | `models/tiny-shakespeare/v2/byte_shakespeare_v2_rosa_off.pt` | 5000 | `186,692` | **1.4787** | **4.39** | `2,706,928 tok/s` |

This is the current small-model reference point for the V2 branch, not a final benchmark table.

## What changed in V2

### 1. The model is V2-only now

The active model surface is:

- [model.py](k_language_model/model.py): public composition / re-export layer
- [kstack.py](k_language_model/kstack.py): `K2Layer`, `KStack`, `KStackModel`
- [layers.py](k_language_model/layers.py): `K0Layer`, `K1Layer`, `MLP`, `RMSNorm`

The old iterative equilibrium path and its refine-step flags were removed. V2 is a direct recurrent-style K-Stack without the extra inner loop.

### 2. `k_base` is local instead of dense

V1 used a dense window-sized `k_base` matrix. V2 replaced that with a local causal kernel implemented in [kbase.py](k_language_model/kbase.py).

Current default:

- `k_base_impl=conv`
- `k_base_kernel_size=8`

Why this changed:

- it matches the learned behavior better
- it removes the `O(W^2)` parameterization for the corrective term
- it lets window length scale without making `k_base` itself grow with `W`

In practice, `k_base` is now treated as the local correction / derivative-like path, while the decay path remains the main temporal memory path.

### 3. The codebase is split by responsibility

The branch is now organized around extension points instead of one giant model file:

- [dataset_loaders.py](k_language_model/dataset_loaders.py): dataset presets and corpus materialization
- [tokenizers.py](k_language_model/tokenizers.py): tokenizer classes
- [dataset_pipeline.py](k_language_model/dataset_pipeline.py): train/val loading, split logic, tokenizer build, tensorization
- [heads.py](k_language_model/heads.py): linear / GELU / adaptive heads
- [decay.py](k_language_model/decay.py): pluggable decay implementations
- [kbase.py](k_language_model/kbase.py): pluggable `k_base` implementations
- [rosa_backends.py](k_language_model/rosa_backends.py): pluggable ROSA backends
- [model_factory.py](k_language_model/model_factory.py): model assembly from config
- [configs.py](k_language_model/configs.py): typed dataset/model config objects
- [cli_args.py](k_language_model/cli_args.py): shared CLI argument groups

Each pluggable family is keyed by a string name and registered by class. The class defines its own `name`, and the builder resolves it through the registry.

## Architecture summary

The active model is a token-level language model built around a K-Stack:

1. token embedding
2. optional embedding projection into `d_model`
3. `KStack`
4. final normalization
5. output head

Inside the stack:

- `K1` front layer
- `n_k2` copies of `K2Layer`
- `K1` tail layer
- `K0` readout layer

Inside `K2Layer`:

- `k_base` branch: local causal correction
- decay branch: rank-wise learned timescale mixing
- optional ROSA branch
- projection + residual
- MLP sub-block

## ROSA (Rapid Online Suffix Automaton)

ROSA lives in [rosa_backends.py](k_language_model/rosa_backends.py) and [rosa.py](k_language_model/rosa.py). Available backends are `off`, `exact`, `gpu_approx`, and `auto`.

The `exact` implementation is derived from the RWKV-v8 ROSA pseudocode. This repo also has an optional native helper in [rosa_ext.cpp](k_language_model/rosa_ext.cpp) for the exact batch path.

The main architectural difference is injection point:

- RWKV-v8 applies ROSA at the embedding / input side.
- This repo computes `rosa_h` once and injects it into the selected K2 mixing layers, where each receiving layer gates the contribution with its learned `rho` term.

Sources:

- [RWKV-v8 ROSA directory](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8)
- [RWKV-8 note](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-8.md)

## Data and tokenizers

Built-in dataset presets:

- `shakespeare`
- `full-shakespeare`
- `full-shakespeare-clean`
- `wikitext2`
- `wikitext2_raw`

Tokenizer options:

- `char`
- `byte`
- `sentencepiece`

Relevant V2 data changes:

- raw-directory corpora can be merged automatically
- `full-shakespeare-clean` strips Folger front matter and other boilerplate
- byte tokenization preserves raw file bytes
- SentencePiece training now uses the train split only instead of fitting on the unsplit source text

Active data entrypoint:

- [data.py](k_language_model/data.py) is intentionally thin
- most dataset/tokenizer work now lives in [dataset_pipeline.py](k_language_model/dataset_pipeline.py)

## Heads

The output head is no longer hardwired.

Available implementations in [heads.py](k_language_model/heads.py):

- `linear`
- `gelu`
- `adaptive`

The head module also owns state-dict adaptation for older V2 checkpoint key layouts.

## Decay implementations

Decay backends live in [decay.py](k_language_model/decay.py):

- `mask`: baseline dense causal-mask implementation
- `block`: lower-memory block recurrence
- `kernel`: experimental Triton/CUDA path with fallback

The active CLI still exposes the same `--decay-impl` knob, but the implementation is now selected through the registry instead of being hardcoded into the model.

## UI app

The local UI is a FastAPI queue for scheduling train and infer runs.

Relevant files:

- [ui_app.py](k_language_model/ui_app.py): web app and inline UI
- [ui_backend.py](k_language_model/ui_backend.py): queue, command building, structured progress parsing

Current UI behavior:

- one page, tabbed train / infer forms
- FIFO queue with a single worker
- structured progress cards instead of raw log spam
- command copy and cancel controls
- checkpoint-aware inputs
- optional extra CLI args when a field is not surfaced yet

Run it with:

```bash
pip install -e .[ui]
python -m k_language_model.ui_app --host 127.0.0.1 --port 8000
```

If your editable install scripts are on `PATH`, this also works:

```bash
k-lm-ui --host 127.0.0.1 --port 8000
```

## Quick start

### Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Train

`train.py` and `k_lm.py` both route to the V2 training entrypoint in [train_app.py](k_language_model/train_app.py).

Example:

```bash
python .\k_lm.py ^
  --dataset shakespeare ^
  --tokenizer byte ^
  --window 512 ^
  --d-model 96 ^
  --n-k2 6 ^
  --rank 4 ^
  --share-k-base ^
  --k-base-kernel-size 8 ^
  --decay-impl mask ^
  --rosa-impl exact ^
  --batch-size 16 ^
  --eval-interval 500 ^
  --lr 1e-2 ^
  --gamma-min 0.05 ^
  --alpha-cap 1.0 ^
  --ckpt runs\\byte_v2.pt
```

### Infer / evaluate

Inference and deterministic evaluation go through [infer_app.py](k_language_model/infer_app.py).

Example:

```bash
python .\infer.py ^
  --ckpt runs\\byte_v2.pt ^
  --dataset shakespeare ^
  --tokenizer byte ^
  --window 512 ^
  --d-model 96 ^
  --n-k2 6 ^
  --rank 4 ^
  --share-k-base ^
  --k-base-kernel-size 8 ^
  --decay-impl mask ^
  --rosa-impl exact ^
  --prompt "KING:\nExplain the nature of time in simple words."
```

## Checkpoints

Checkpoint save/load lives in [checkpoint.py](k_language_model/checkpoint.py).

Important note:

- V1 compatibility is intentionally dropped on this branch
- current checkpoints are expected to match the modular V2 architecture

The loader still handles internal V2 key migrations where practical, such as split head layouts.

## Extending the system

If you want to add a new backend or data path, the intended route is to add a class and register it.

Current extension surfaces:

- dataset loaders in [dataset_loaders.py](k_language_model/dataset_loaders.py)
- tokenizers in [tokenizers.py](k_language_model/tokenizers.py)
- heads in [heads.py](k_language_model/heads.py)
- decay backends in [decay.py](k_language_model/decay.py)
- `k_base` backends in [kbase.py](k_language_model/kbase.py)
- ROSA backends in [rosa_backends.py](k_language_model/rosa_backends.py)

The training and inference CLIs consume those modules through shared config objects and [model_factory.py](k_language_model/model_factory.py), so adding a backend later should not require another repo-wide refactor.

## Console commands

Editable install scripts from [pyproject.toml](pyproject.toml):

- `k-lm-train`
- `k-lm-infer`
- `k-lm-bench-denoise`
- `k-lm-sequence-predict`
- `k-lm-sequence-stats`
- `k-lm-ui`

## Roadmap

Current near-term work:

- Implement fused ROSA
- Add a prefix-scan decay path
- Test V2 scaling on WikiText-103

## License

This repository's original code is licensed under [MIT](LICENSE).

Some ROSA-related files include implementation material derived from the RWKV-v8 ROSA formulation and pseudocode, and are distributed with Apache-2.0 attribution:

- [k_language_model/rosa.py](k_language_model/rosa.py)
- [k_language_model/rosa_backends.py](k_language_model/rosa_backends.py)
- [k_language_model/rosa_ext.cpp](k_language_model/rosa_ext.cpp)

See:

- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)
