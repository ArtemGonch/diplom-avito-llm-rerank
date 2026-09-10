#!/usr/bin/env bash
# UR4Rec guaranteed run: beat_base data protocol + full 512 knowledge + joint safeguards.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs checkpoints/ur4rec

CONFIG="configs/ur4rec/ur4rec_ml1m_guaranteed.yaml"
MASTER_LOG="logs/guaranteed_master.log"
TRAIN_LOG="logs/guaranteed_train.log"
TRAIN_GPU="${TRAIN_GPU:-6}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diplom_avito

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

log "========== UR4Rec GUARANTEED run (beat base + approach paper) =========="
log "Config: $CONFIG | GPU: $TRAIN_GPU"
log "Data: no k-core, random negatives, 100 candidates"
log "Knowledge: knowledge_qwen25_full (512 tok, already cached)"
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
log "Total: $(( (T1 - T0) / 3600 ))h $(( (T1 - T0) % 3600 / 60 ))m"
log "Metrics: $ROOT/checkpoints/ur4rec_ml1m_guaranteed/metrics_test.json"
