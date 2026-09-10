#!/usr/bin/env bash
# Exp3RT rating: vLLM test inference + RMSE/MAE eval (paper-comparable).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs results/current/tables

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diplom_avito

CONFIG="${EXP3RT_CONFIG:-configs/exp3rt/amazon_book_qwen.yaml}"
LOG="${EXP3RT_TEST_LOG:-logs/exp3rt_rating_test.log}"
TP="${EXP3RT_TP:-2}"
MIN_FREE_MIB="${EXP3RT_MIN_FREE_MIB:-55000}"

pick_free_gpus() {
  local n="$1"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' -v min="$MIN_FREE_MIB" '$2 + 0 >= min {print $1, $2}' \
    | sort -k2 -nr \
    | head -n "$n" \
    | awk '{print $1}' \
    | sort -n \
    | paste -sd,
}

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${EXP3RT_AUTO_GPU:-0}" != "1" ]; then
  GPUS="$CUDA_VISIBLE_DEVICES"
else
  GPUS="$(pick_free_gpus "$TP")"
  if [ -z "$GPUS" ]; then
    echo "No GPU with >= ${MIN_FREE_MIB} MiB free. Set CUDA_VISIBLE_DEVICES manually." >&2
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader >&2
    exit 1
  fi
  # Match TP to number of selected GPUs.
  TP="$(echo "$GPUS" | awk -F, '{print NF}')"
fi

export EXP3RT_TP="$TP"
export CUDA_VISIBLE_DEVICES="$GPUS"

echo "=== Exp3RT test+eval $(date -Is) TP=$TP CUDA_VISIBLE_DEVICES=$GPUS ===" | tee "$LOG"

python -u scripts/exp3rt/run_exp3rt.py --config "$CONFIG" --stage test 2>&1 | tee -a "$LOG"
python -u scripts/exp3rt/run_exp3rt.py --config "$CONFIG" --stage eval 2>&1 | tee -a "$LOG"

python scripts/snapshot_experiment_results.py
python scripts/exp3rt/build_paper_comparison.py

echo "=== done $(date -Is) ===" | tee -a "$LOG"
