#!/usr/bin/env bash
# Variant 1: full knowledge + train to beat DLCM baseline (~8-10h V100, ~2.5-3h A100 4-GPU).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs checkpoints/ur4rec

CONFIG="configs/ur4rec/ur4rec_ml1m_beat_base.yaml"
MASTER_LOG="logs/beat_baseline_master.log"
KNOW_LOG="logs/beat_baseline_knowledge.log"
TRAIN_LOG="logs/beat_baseline_train.log"
NUM_SHARDS="${KNOWLEDGE_NUM_SHARDS:-4}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diplom_avito

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

log "========== UR4Rec beat-baseline run (variant 1) =========="
log "Config: $CONFIG"
log "Knowledge shards: $NUM_SHARDS GPUs (parallel LLM)"
log "Master log: $ROOT/$MASTER_LOG"
log "Knowledge logs: $ROOT/logs/beat_baseline_knowledge_shard*.log"
log "Train log: $ROOT/$TRAIN_LOG"
log ""
log "Tail progress:"
log "  tail -f $ROOT/$MASTER_LOG"
log "  tail -f $ROOT/logs/beat_baseline_knowledge_shard0.log"
log "  tail -f $ROOT/$TRAIN_LOG"
log ""

T0=$(date +%s)

log ">>> STAGE knowledge (${NUM_SHARDS} GPUs, bs=8, ~3883 items + ~6032 users)"
PIDS=()
for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_LOG="logs/beat_baseline_knowledge_shard${SHARD}.log"
  log "  launching shard ${SHARD}/${NUM_SHARDS} on GPU ${SHARD} -> ${SHARD_LOG}"
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
    log "ERROR: knowledge shard ${i} failed (see logs/beat_baseline_knowledge_shard${i}.log)"
    FAIL=1
  fi
done
if [ "$FAIL" -ne 0 ]; then
  exit 1
fi

log ">>> STAGE merge_knowledge"
python -u scripts/ur4rec/run_ur4rec.py \
  --config "$CONFIG" \
  --stage merge_knowledge \
  2>&1 | tee -a "$KNOW_LOG" | tee -a "$MASTER_LOG"

T1=$(date +%s)
log ">>> knowledge done in $(( (T1 - T0) / 3600 ))h $(( (T1 - T0) % 3600 / 60 ))m"

log ">>> STAGE backbone + pretrain + joint + eval (expect ~1.5-2.5h on A100)"
for ST in backbone pretrain joint eval; do
  log ">>> STAGE $ST"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU:-0}" python -u scripts/ur4rec/run_ur4rec.py \
    --config "$CONFIG" \
    --stage "$ST" \
    2>&1 | tee -a "$TRAIN_LOG" | tee -a "$MASTER_LOG"
done

T2=$(date +%s)
TOTAL=$((T2 - T0))
log "========== FINISHED =========="
log "Total: $(( TOTAL / 3600 ))h $(( TOTAL % 3600 / 60 ))m"
log "Metrics: $ROOT/checkpoints/ur4rec_ml1m_beat_base/metrics_test.json"
