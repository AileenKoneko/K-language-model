#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash configs/colab_seed_sweep_wikitext2_kbase.sh [LOG_PATH]
#
# Example:
#   bash configs/colab_seed_sweep_wikitext2_kbase.sh \
#     /content/drive/MyDrive/k_operators/logs/k_lm_seed_sweep_wikitext2_kbase.log

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-}"
SP_MODEL="${SP_MODEL:-./data/tokenizers/wikitext2_unigram_8192.model}"
CKPT_BASENAME="${CKPT_BASENAME:-papiezu_prosze_test_12}"
DEFAULT_LOG="/content/drive/MyDrive/k_operators/logs/k_lm_seed_sweep_wikitext2_kbase_$(date +%Y%m%d_%H%M%S).log"
LOG_PATH="${1:-$DEFAULT_LOG}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
CACHE_ROOT="${CACHE_ROOT:-/tmp/k_lm_compile_cache_${RUN_TAG}}"
CKPT_ROOT="${CKPT_ROOT:-/content/drive/MyDrive/k_operators/checkpoints/${RUN_TAG}}"

if [[ "$LOG_PATH" == /content/drive/* ]] && [[ ! -d /content/drive/MyDrive ]]; then
  echo "Google Drive is not mounted at /content/drive/MyDrive." >&2
  echo "In Colab, run:" >&2
  echo "from google.colab import drive; drive.mount('/content/drive')" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_PATH")"
mkdir -p "$CKPT_ROOT"

if [[ ! -f "$SP_MODEL" ]]; then
  echo "SentencePiece model not found: $SP_MODEL" >&2
  exit 1
fi

if [[ -n "$SCRIPT_PATH" ]]; then
  if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "SCRIPT_PATH is set but file does not exist: $SCRIPT_PATH" >&2
    exit 1
  fi
elif [[ -f "k_lm.py" ]]; then
  SCRIPT_PATH="k_lm.py"
elif [[ -f "train.py" ]]; then
  SCRIPT_PATH="train.py"
else
  echo "Could not find k_lm.py or train.py. Run from the k_operators directory or set SCRIPT_PATH." >&2
  exit 1
fi

BASE_ARGS=(
  --dataset wikitext2
  --tokenizer sentencepiece
  --sp-model "$SP_MODEL"
  --sp-vocab-size 8192
  --d-model 256
  --head-mode adaptive
  --adaptive-cutoffs 1024,4096
  --emb-dim 64
  --head-dropout 0.3
  --n-k2 6
  --rank 32
  --k-base-rank 0
  --share-k-base
  --window 512
  --refine-steps 0
  --optimizer-mode grouped
  --fused-adamw
  --batch-size 32
  --steps 25000
  --eval-interval 250
  --compile
  --compile-mode default
  --lr 10e-3
  --warmup-steps 3000
  --alpha-cap 0.95
  --decay-impl mask
  --lr-floor 1e-04
  --gamma-min 0.15
  --gamma-max 0.995
  --residual-dropout 0.1
  --emb-dropout 0.16
  --mlp-dropout 0.2
  --diagnostics
)

# Intentionally unique seeds (no repeated 42 runs).
SEEDS=(42 69 420 666 2137)
TOTAL_RUNS="${#SEEDS[@]}"

{
  echo "== K-LM Colab Seed Sweep (wikitext2 k-base config) =="
  echo "start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "python_bin=$PYTHON_BIN"
  echo "script=$SCRIPT_PATH"
  echo "sp_model=$SP_MODEL"
  echo "cache_root=$CACHE_ROOT"
  echo "ckpt_root=$CKPT_ROOT"
  echo "ckpt_basename=$CKPT_BASENAME"
  echo "base_args=${BASE_ARGS[*]}"
  echo "seeds=${SEEDS[*]}"
  echo "diagnostics=enabled"
  echo
} | tee "$LOG_PATH"

for idx in "${!SEEDS[@]}"; do
  run_no=$((idx + 1))
  seed="${SEEDS[$idx]}"
  run_cache_dir="${CACHE_ROOT}/run_${run_no}_seed_${seed}"
  run_inductor_cache="${run_cache_dir}/inductor"
  run_triton_cache="${run_cache_dir}/triton"
  run_cuda_cache="${run_cache_dir}/cuda"
  run_ckpt="${CKPT_ROOT}/${CKPT_BASENAME}_run_${run_no}_seed_${seed}.pt"
  mkdir -p "$run_inductor_cache" "$run_triton_cache" "$run_cuda_cache"

  {
    echo "========== RUN ${run_no}/${TOTAL_RUNS} | seed=${seed} =========="
    echo "start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "ckpt=$run_ckpt"
    echo "cmd=$PYTHON_BIN $SCRIPT_PATH ${BASE_ARGS[*]} --seed $seed --ckpt $run_ckpt"
    echo "cache_dirs: inductor=$run_inductor_cache triton=$run_triton_cache cuda=$run_cuda_cache"
  } | tee -a "$LOG_PATH"

  TORCHINDUCTOR_CACHE_DIR="$run_inductor_cache" \
  TRITON_CACHE_DIR="$run_triton_cache" \
  CUDA_CACHE_PATH="$run_cuda_cache" \
  "$PYTHON_BIN" "$SCRIPT_PATH" "${BASE_ARGS[@]}" --seed "$seed" --ckpt "$run_ckpt" 2>&1 | tee -a "$LOG_PATH"

  {
    echo "end_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo
  } | tee -a "$LOG_PATH"
done

echo "Sweep complete. Log file: $LOG_PATH" | tee -a "$LOG_PATH"
