#!/usr/bin/env bash
# Exp3RT paper-full: 4 chained stages + vLLM test + eval.
# GPU 4-7 free while UR4Rec uses 0-3.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs checkpoints/exp3rt

CONFIG="configs/exp3rt/amazon_book_qwen_paper_full.yaml"
MASTER="logs/exp3rt_paper_full_master.log"
TRAIN_LOG="logs/exp3rt_paper_full_train.log"
TEST_LOG="logs/exp3rt_paper_full_test.log"
GPU="${EXP3RT_GPU:-4}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diplom_avito

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER"; }

log "========== Exp3RT paper-full Amazon-Books =========="
log "Config: $CONFIG | GPU: $GPU"
log "Target: beat paper RMSE<0.6508 MAE<0.4297 on test"

T0=$(date +%s)
INIT=""
for ST in preference user item rating; do
  log ">>> TRAIN stage=$ST init=${INIT:-Qwen base}"
  CMD=(python -u scripts/exp3rt/run_exp3rt.py --config "$CONFIG" --stage train --train-stage "$ST")
  if [ -n "$INIT" ]; then
    CMD+=(--init-model "$INIT")
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" 2>&1 | tee -a "$TRAIN_LOG" | tee -a "$MASTER"
  INIT="$(ls -d "$ROOT/checkpoints/exp3rt/amazon_book_qwen_paper_full/amazon-book_${ST}_r"*/merged 2>/dev/null | head -1)"
  if [ ! -d "$INIT" ]; then
    log "ERROR: missing merged after $ST: $INIT"
    exit 1
  fi
  log "  merged -> $INIT"
done

T1=$(date +%s)
log ">>> TRAIN all stages done in $(( (T1 - T0) / 3600 ))h $(( (T1 - T0) % 3600 / 60 ))m"

log ">>> TEST + EVAL (vLLM TP=2 on GPU 4,5)"
export EXP3RT_CONFIG="$CONFIG"
export EXP3RT_TEST_LOG="$TEST_LOG"
export CUDA_VISIBLE_DEVICES="${EXP3RT_TEST_GPUS:-4,5}"
bash scripts/exp3rt/run_test_eval.sh 2>&1 | tee -a "$TEST_LOG" | tee -a "$MASTER"

# point comparison script at new metrics
METRICS="$ROOT/checkpoints/exp3rt/amazon_book_qwen_paper_full"
RATING_DIR="$(find "$METRICS" -maxdepth 1 -type d -name 'amazon-book_rating_*' | head -1)"
if [ -f "$RATING_DIR/metrics.json" ]; then
  cp "$RATING_DIR/metrics.json" "$ROOT/results/current/metrics/exp3rt_paper_full_test_metrics.json"
fi

T2=$(date +%s)
log "========== FINISHED total $(( (T2 - T0) / 3600 ))h $(( (T2 - T0) % 3600 / 60 ))m =========="
log "Metrics: $RATING_DIR/metrics.json"
