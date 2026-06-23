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
GPUS="${CUDA_VISIBLE_DEVICES:-4,5}"

echo "=== Exp3RT test+eval $(date -Is) TP=$TP CUDA_VISIBLE_DEVICES=$GPUS ===" | tee "$LOG"

export CUDA_VISIBLE_DEVICES="$GPUS"
python -u scripts/exp3rt/run_exp3rt.py --config "$CONFIG" --stage test 2>&1 | tee -a "$LOG"
python -u scripts/exp3rt/run_exp3rt.py --config "$CONFIG" --stage eval 2>&1 | tee -a "$LOG"

python scripts/snapshot_experiment_results.py
python scripts/exp3rt/build_paper_comparison.py

echo "=== done $(date -Is) ===" | tee -a "$LOG"
