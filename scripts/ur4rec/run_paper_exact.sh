#!/usr/bin/env bash
# UR4Rec paper-exact ML-1M: k-core + MF top-100 + paper losses/aug.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs checkpoints/ur4rec

CONFIG="configs/ur4rec/ur4rec_ml1m_paper_exact.yaml"
MASTER_LOG="logs/paper_exact_master.log"
TRAIN_LOG="logs/paper_exact_train.log"
TRAIN_GPU="${TRAIN_GPU:-6}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diplom_avito

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

log "========== UR4Rec paper-exact ML-1M =========="
log "Config: $CONFIG | GPU: $TRAIN_GPU"
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
log "Metrics: $ROOT/checkpoints/ur4rec_ml1m_paper_exact/metrics_test.json"
