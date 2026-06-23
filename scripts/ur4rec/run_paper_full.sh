#!/usr/bin/env bash
# UR4Rec paper-faithful ML-1M reproduction (100 cand, 512 tok, full split).
# Knowledge ~25-35h (4 GPUs), train ~6-10h on A100.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs checkpoints/ur4rec

CONFIG="configs/ur4rec/ur4rec_ml1m_full.yaml"
MASTER_LOG="logs/paper_full_master.log"
KNOW_LOG="logs/paper_full_knowledge.log"
TRAIN_LOG="logs/paper_full_train.log"
NUM_SHARDS="${KNOWLEDGE_NUM_SHARDS:-4}"
TRAIN_GPU="${TRAIN_GPU:-0}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diplom_avito

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

log "========== UR4Rec paper-full ML-1M =========="
log "Config: $CONFIG"
log "Knowledge shards: $NUM_SHARDS | train GPU: $TRAIN_GPU"
log ""

T0=$(date +%s)

log ">>> STAGE knowledge (${NUM_SHARDS} GPUs parallel)"
PIDS=()
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_LOG="logs/paper_full_knowledge_shard${SHARD}.log"
  log "  shard ${SHARD}/${NUM_SHARDS} GPU ${SHARD} -> ${SHARD_LOG}"
  CUDA_VISIBLE_DEVICES="${SHARD}" python -u scripts/ur4rec/run_ur4rec.py \
    --config "$CONFIG" \
    --stage knowledge \
    --knowledge-shard-id "$SHARD" \
    --knowledge-num-shards "$NUM_SHARDS" \
    --gpu-id 0 \
    >> "$SHARD_LOG" 2>&1 &
  PIDS+=("$!")
done
FAIL=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    log "ERROR: knowledge shard ${i} failed"
    FAIL=1
  fi
done
[ "$FAIL" -eq 0 ] || exit 1

log ">>> STAGE merge_knowledge"
python -u scripts/ur4rec/run_ur4rec.py \
  --config "$CONFIG" \
  --stage merge_knowledge \
  2>&1 | tee -a "$KNOW_LOG" | tee -a "$MASTER_LOG"

T1=$(date +%s)
log ">>> knowledge done in $(( (T1 - T0) / 3600 ))h $(( (T1 - T0) % 3600 / 60 ))m"

log ">>> STAGE backbone + pretrain + joint + eval"
for ST in backbone pretrain joint eval; do
  log ">>> STAGE $ST"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python -u scripts/ur4rec/run_ur4rec.py \
    --config "$CONFIG" \
    --stage "$ST" \
    --gpu-id 0 \
    2>&1 | tee -a "$TRAIN_LOG" | tee -a "$MASTER_LOG"
done

T2=$(date +%s)
log "========== FINISHED =========="
log "Total: $(( (T2 - T0) / 3600 ))h $(( (T2 - T0) % 3600 / 60 ))m"
log "Metrics: $ROOT/checkpoints/ur4rec_ml1m_full/metrics_test.json"
