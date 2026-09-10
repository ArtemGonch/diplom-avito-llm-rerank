#!/usr/bin/env bash
# Corrected UR4Rec ML-1M rerun: parallel knowledge, then sequential training/eval.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/ur4rec/ur4rec_ml1m_corrected_v3.yaml"
LOG_DIR="logs/ur4rec_corrected_v3"
MASTER_LOG="$LOG_DIR/master.log"
KNOW_LOG="$LOG_DIR/merge.log"
TRAIN_LOG="$LOG_DIR/train.log"
CONDA_SH="${CONDA_SH:-/home/artem-gon/miniconda3/etc/profile.d/conda.sh}"
KNOWLEDGE_GPUS="${KNOWLEDGE_GPUS:-0,1,2,3}"

mkdir -p "$LOG_DIR"
source "$CONDA_SH"
conda activate diplom_avito

IFS=',' read -r -a GPU_IDS <<< "$KNOWLEDGE_GPUS"
NUM_SHARDS="${#GPU_IDS[@]}"
if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "KNOWLEDGE_GPUS must contain at least one GPU id" >&2
  exit 2
fi
TRAIN_GPU="${TRAIN_GPU:-${GPU_IDS[0]}}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

log "========== UR4Rec corrected-v3 ML-1M =========="
log "Config: $CONFIG"
log "Knowledge GPUs: $KNOWLEDGE_GPUS ($NUM_SHARDS shards) | train GPU: $TRAIN_GPU"

T0=$(date +%s)
PIDS=()
log ">>> STAGE knowledge"
for SHARD in "${!GPU_IDS[@]}"; do
  GPU="${GPU_IDS[$SHARD]}"
  SHARD_LOG="$LOG_DIR/knowledge_shard${SHARD}.log"
  log "  shard $SHARD/$NUM_SHARDS on physical GPU $GPU -> $SHARD_LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python -u scripts/ur4rec/run_ur4rec.py \
    --config "$CONFIG" \
    --stage knowledge \
    --knowledge-shard-id "$SHARD" \
    --knowledge-num-shards "$NUM_SHARDS" \
    --gpu-id 0 \
    >> "$SHARD_LOG" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for INDEX in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$INDEX]}"; then
    log "ERROR: knowledge shard $INDEX failed; see $LOG_DIR/knowledge_shard${INDEX}.log"
    FAIL=1
  fi
done
[[ "$FAIL" -eq 0 ]] || exit 1

log ">>> STAGE merge_knowledge"
python -u scripts/ur4rec/run_ur4rec.py \
  --config "$CONFIG" \
  --stage merge_knowledge \
  2>&1 | tee -a "$KNOW_LOG" | tee -a "$MASTER_LOG"

T1=$(date +%s)
log ">>> knowledge completed in $(( (T1 - T0) / 3600 ))h $(( (T1 - T0) % 3600 / 60 ))m"

for STAGE in backbone pretrain joint eval; do
  log ">>> STAGE $STAGE on physical GPU $TRAIN_GPU"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python -u scripts/ur4rec/run_ur4rec.py \
    --config "$CONFIG" \
    --stage "$STAGE" \
    --gpu-id 0 \
    2>&1 | tee -a "$TRAIN_LOG" | tee -a "$MASTER_LOG"
done

T2=$(date +%s)
log "========== FINISHED in $(( (T2 - T0) / 3600 ))h $(( (T2 - T0) % 3600 / 60 ))m =========="
log "Metrics: $ROOT/checkpoints/ur4rec_ml1m_corrected_v3/metrics_test.json"
