#!/usr/bin/env bash
# UR4Rec ML-1M v2: fixed retriever aug + encoder 512 + val early-stop.
# Skips knowledge regen (reuses knowledge_qwen25_full).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs checkpoints/ur4rec

CONFIG="configs/ur4rec/ur4rec_ml1m_paper_v2.yaml"
MASTER_LOG="logs/paper_v2_master.log"
TRAIN_LOG="logs/paper_v2_train.log"
TRAIN_GPU="${TRAIN_GPU:-5}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diplom_avito

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

log "========== UR4Rec paper-v2 ML-1M (fixed code) =========="
log "Config: $CONFIG | train GPU: $TRAIN_GPU"
log "Knowledge: cached at data/movielens-1m/knowledge_qwen25_full"
log ""

T0=$(date +%s)

for ST in backbone pretrain joint eval; do
  log ">>> STAGE $ST"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python -u scripts/ur4rec/run_ur4rec.py \
    --config "$CONFIG" \
    --stage "$ST" \
    --gpu-id 0 \
    2>&1 | tee -a "$TRAIN_LOG" | tee -a "$MASTER_LOG"
done

T1=$(date +%s)
log "========== FINISHED =========="
log "Total train: $(( (T1 - T0) / 3600 ))h $(( (T1 - T0) % 3600 / 60 ))m"
log "Metrics: $ROOT/checkpoints/ur4rec_ml1m_paper_v2/metrics_test.json"
