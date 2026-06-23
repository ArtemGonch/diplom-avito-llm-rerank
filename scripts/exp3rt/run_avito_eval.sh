#!/usr/bin/env bash
# Exp3RT-style attribute rerank on full Avito test split.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p results/current/metrics logs

CONFIG="${1:-configs/exp3rt/exp3rt_avito_full.yaml}"
OUT="results/current/metrics/exp3rt_avito_full.json"
LOG="logs/exp3rt_avito_full.log"
GPU="${AVITO_GPU:-7}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diplom_avito

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exp3RT Avito full test GPU=$GPU" | tee "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/exp3rt/exp3rt_avito_attribute_rerank.py \
  --config "$CONFIG" \
  --mode heuristic \
  --split test \
  --output "$OUT" \
  2>&1 | tee -a "$LOG"

echo "Wrote $OUT" | tee -a "$LOG"
