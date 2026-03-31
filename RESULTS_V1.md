# K-Operators V1 Results Archive

This file stores the benchmark tables that previously lived in the original README for the pre-refactor branch.

## Paper v2 ablations (March 2026)

Source: `k_operators_paper_v2.tex` (March 2026 revision).

- Recommended config from ablations: enable `K_base`, use uncapped gamma (`0.15 <= gamma <= 0.995`), disable refinement (`--refine-steps 0`).

WikiText-2 ablation (SentencePiece vocab 8192, 25K steps, `d-model=256`, `rank=32`, `window=512`):

| Configuration | K_base | Shared K_base | Params | Val PPL |
|---|:---:|:---:|---:|---:|
| Full (shared K_base, 5 seeds) | yes | yes | 4.08M | **19.99 +/- 0.09** |
| Full (per-layer K_base) | yes | no | 5.39M | 19.89 |
| No K_base (`d=256`) | no | no | 3.82M | 20.59 |
| No K_base (`d=304`, capacity control) | no | no | ~5.4M | 20.99 |
| Original (capped `gamma >= 0.85`, with refinement) | yes | no | 3.79M | 22.33 |

Tiny Shakespeare ablation (character-level vocab 65, `d-model=128`, `rank=32`, `window=256`):

| K_base | Refinement | Gamma range | Val PPL | Notes |
|:---:|:---:|:---:|---:|---|
| yes | no | uncapped (`gamma >= 0.15`) | **4.55** | Best configuration |
| yes | no | capped (`gamma >= 0.85`) | 4.67 |  |
| no | no | uncapped (`gamma >= 0.15`) | 4.78 |  |
| no | yes | uncapped (`gamma >= 0.15`) | 4.73 |  |
| yes | yes | capped (`gamma >= 0.85`) | 4.74 | Original configuration |

Component effects from the paper (positive is better):

| Component | WikiText-2 Delta PPL | Tiny Shakespeare Delta PPL |
|---|---:|---:|
| Gamma uncapping (`0.15 <= gamma`) | +2.44 | +0.12 |
| Add `K_base` (vs equal-capacity no-`K_base`) | +1.00 | +0.23 |
| Share `K_base` (vs per-layer) | -0.10 | n/a |
| Enable refinement (`eta > 0`) | -0.3 to -0.5 | -0.06 to -0.19 |

## Historical checkpoint evals (March 13-15, 2026)

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

| Checkpoint | Step | Val BPC | Val PPL | Window | d-model | Params | Run hash |
|---|---:|---:|---:|---:|---:|---:|---|
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
