#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash k_neurons/configs/colab_seed_sweep.sh [LOG_PATH]
#
# Example:
#   bash k_neurons/configs/colab_seed_sweep.sh \
#     /content/drive/MyDrive/k_language_model/logs/k_lm_seed_sweep.log

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-}"
DEFAULT_LOG="/content/drive/MyDrive/k_operators/logs/k_lm_seed_sweep_$(date +%Y%m%d_%H%M%S).log"
LOG_PATH="${1:-$DEFAULT_LOG}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
CACHE_ROOT="${CACHE_ROOT:-/tmp/k_lm_compile_cache_${RUN_TAG}}"

if [[ "$LOG_PATH" == /content/drive/* ]] && [[ ! -d /content/drive/MyDrive ]]; then
  echo "Google Drive is not mounted at /content/drive/MyDrive." >&2
  echo "In Colab, run:" >&2
  echo "from google.colab import drive; drive.mount('/content/drive')" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_PATH")"

if [[ -n "$SCRIPT_PATH" ]]; then
  if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "SCRIPT_PATH is set but file does not exist: $SCRIPT_PATH" >&2
    exit 1
  fi
elif [[ -f "k_lm.py" ]]; then
  SCRIPT_PATH="k_lm.py"
elif [[ -f "k_neurons/k_lm.py" ]]; then
  SCRIPT_PATH="k_neurons/k_lm.py"
else
  echo "Could not find k_lm.py. Run from repo root or /content." >&2
  exit 1
fi

BASE_ARGS=(
  --d-model 128
  --head-mode gelu
  --head-mult 1
  --n-k2 6
  --refine-steps 1
  --rank 32
  --window 256
  --optimizer-mode simple
  --batch-size 64
  --steps 4000
  --eval-interval 250
  --compile
  --lr 8e-3
  --warmup-steps 1500
)

SEEDS=(42 42 42 69 420 666 2137)
TOTAL_RUNS="${#SEEDS[@]}"

{
  echo "== K-LM Colab Seed Sweep =="
  echo "start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "python_bin=$PYTHON_BIN"
  echo "script=$SCRIPT_PATH"
  echo "cache_root=$CACHE_ROOT"
  echo "base_args=${BASE_ARGS[*]}"
  echo "seeds=${SEEDS[*]}"
  echo
} | tee "$LOG_PATH"

for idx in "${!SEEDS[@]}"; do
  run_no=$((idx + 1))
  seed="${SEEDS[$idx]}"
  run_cache_dir="${CACHE_ROOT}/run_${run_no}_seed_${seed}"
  run_inductor_cache="${run_cache_dir}/inductor"
  run_triton_cache="${run_cache_dir}/triton"
  run_cuda_cache="${run_cache_dir}/cuda"
  mkdir -p "$run_inductor_cache" "$run_triton_cache" "$run_cuda_cache"

  {
    echo "========== RUN ${run_no}/${TOTAL_RUNS} | seed=${seed} =========="
    echo "start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "cmd=$PYTHON_BIN $SCRIPT_PATH ${BASE_ARGS[*]} --seed $seed"
    echo "cache_dirs: inductor=$run_inductor_cache triton=$run_triton_cache cuda=$run_cuda_cache"
  } | tee -a "$LOG_PATH"

  TORCHINDUCTOR_CACHE_DIR="$run_inductor_cache" \
  TRITON_CACHE_DIR="$run_triton_cache" \
  CUDA_CACHE_PATH="$run_cuda_cache" \
  "$PYTHON_BIN" "$SCRIPT_PATH" "${BASE_ARGS[@]}" --seed "$seed" 2>&1 | tee -a "$LOG_PATH"

  {
    echo "end_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo
  } | tee -a "$LOG_PATH"
done

echo "Sweep complete. Log file: $LOG_PATH" | tee -a "$LOG_PATH"
