#!/usr/bin/env bash
# Resume UR4Rec paper-exact from joint (backbone + pretrain already done).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs checkpoints/ur4rec

CONFIG="configs/ur4rec/ur4rec_ml1m_paper_exact.yaml"
MASTER_LOG="logs/paper_exact_master.log"
TRAIN_LOG="logs/paper_exact_train.log"
TRAIN_GPU="${TRAIN_GPU:-6}"
OUT="$ROOT/checkpoints/ur4rec_ml1m_paper_exact"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diplom_avito

for f in backbone.pt retriever_pretrain.pt; do
  if [ ! -f "$OUT/$f" ]; then
    echo "Missing checkpoint: $OUT/$f" >&2
    exit 1
  fi
done

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

log "========== UR4Rec paper-exact RESUME (joint→eval) =========="
log "Config: $CONFIG | GPU: $TRAIN_GPU"

T0=$(date +%s)

for ST in joint eval; do
  log ">>> STAGE $ST (resume)"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python -u scripts/ur4rec/run_ur4rec.py \
    --config "$CONFIG" \
    --stage "$ST" \
    --gpu-id 0 \
    2>&1 | tee -a "$TRAIN_LOG" | tee -a "$MASTER_LOG"
done

T1=$(date +%s)
log "========== RESUME FINISHED =========="
log "Total: $(( (T1 - T0) / 3600 ))h $(( (T1 - T0) % 3600 / 60 ))m"
log "Metrics: $OUT/metrics_test.json"
