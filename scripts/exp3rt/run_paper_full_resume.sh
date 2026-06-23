#!/usr/bin/env bash
# Resume Exp3RT paper-full from stage user (preference already merged).
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

INIT="$ROOT/checkpoints/exp3rt/amazon_book_qwen_paper_full/amazon-book_preference_r128_alpha16_seed425/merged"
if [ ! -d "$INIT" ]; then
  echo "Missing preference merged: $INIT" >&2
  exit 1
fi

log "========== Exp3RT paper-full RESUME (user→rating) =========="
log "Init: $INIT | GPU: $GPU"

T0=$(date +%s)
for ST in user item rating; do
  log ">>> TRAIN stage=$ST init=$INIT"
  CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/exp3rt/run_exp3rt.py \
    --config "$CONFIG" --stage train --train-stage "$ST" \
    --init-model "$INIT" \
    2>&1 | tee -a "$TRAIN_LOG" | tee -a "$MASTER"
  INIT="$(ls -d "$ROOT/checkpoints/exp3rt/amazon_book_qwen_paper_full/amazon-book_${ST}_r"*/merged 2>/dev/null | head -1)"
  log "  merged -> $INIT"
done

T1=$(date +%s)
log ">>> TRAIN resume done in $(( (T1 - T0) / 3600 ))h $(( (T1 - T0) % 3600 / 60 ))m"

log ">>> TEST + EVAL"
export EXP3RT_CONFIG="$CONFIG"
export EXP3RT_TEST_LOG="$TEST_LOG"
export CUDA_VISIBLE_DEVICES="${EXP3RT_TEST_GPUS:-4,5}"
bash scripts/exp3rt/run_test_eval.sh 2>&1 | tee -a "$TEST_LOG" | tee -a "$MASTER"

T2=$(date +%s)
log "========== FINISHED total $(( (T2 - T0) / 3600 ))h $(( (T2 - T0) % 3600 / 60 ))m =========="
