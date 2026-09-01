#!/usr/bin/env bash
# Resume corrected-v3 after completed sharded knowledge generation.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/ur4rec/ur4rec_ml1m_corrected_v3.yaml"
LOG_DIR="logs/ur4rec_corrected_v3"
MASTER_LOG="$LOG_DIR/master.log"
MERGE_LOG="$LOG_DIR/merge.log"
TRAIN_LOG="$LOG_DIR/train.log"
STATUS_FILE="$LOG_DIR/resume.status"
PID_FILE="$LOG_DIR/resume.pid"
METRICS="$ROOT/checkpoints/ur4rec_ml1m_corrected_v3/metrics_test.json"
OUT="$ROOT/checkpoints/ur4rec_ml1m_corrected_v3"
CONDA_SH="${CONDA_SH:-/home/artem-gon/miniconda3/etc/profile.d/conda.sh}"
TRAIN_GPU="${TRAIN_GPU:-2}"

mkdir -p "$LOG_DIR" "$OUT"
source "$CONDA_SH"
conda activate diplom_avito

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

set_status() {
  echo "$1" > "$STATUS_FILE"
}

CURRENT_STAGE="preflight"
on_error() {
  local exit_code="$?"
  set_status "failed stage=$CURRENT_STAGE exit=$exit_code"
  log "========== RESUME FAILED stage=$CURRENT_STAGE exit=$exit_code line=$1 =========="
  exit "$exit_code"
}
on_signal() {
  set_status "interrupted stage=$CURRENT_STAGE signal=$1"
  log "========== RESUME INTERRUPTED stage=$CURRENT_STAGE signal=$1 =========="
  exit 130
}
cleanup() {
  rm -f "$PID_FILE"
}
trap 'on_error $LINENO' ERR
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap cleanup EXIT

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(tr -dc '0-9' < "$PID_FILE")"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Resume already running with PID $OLD_PID" >&2
    exit 3
  fi
fi
echo "$$" > "$PID_FILE"

log "========== UR4Rec corrected-v3 RESUME (merge→eval) =========="
log "Config: $CONFIG | physical GPU: $TRAIN_GPU | PID: $$"
T0="$(date +%s)"

CURRENT_STAGE="merge_knowledge"
set_status "running stage=$CURRENT_STAGE pid=$$ gpu=none"
log ">>> STAGE $CURRENT_STAGE"
python -u scripts/ur4rec/run_ur4rec.py \
  --config "$CONFIG" \
  --stage "$CURRENT_STAGE" \
  2>&1 | tee -a "$MERGE_LOG" | tee -a "$MASTER_LOG"

for STAGE in backbone pretrain joint eval; do
  CURRENT_STAGE="$STAGE"
  set_status "running stage=$CURRENT_STAGE pid=$$ gpu=$TRAIN_GPU"

  if [[ "$STAGE" == "backbone" && -f "$OUT/backbone.pt" ]]; then
    log ">>> SKIP backbone: $OUT/backbone.pt already exists"
    continue
  fi
  if [[ "$STAGE" == "pretrain" && -f "$OUT/retriever_pretrain.pt" ]]; then
    log ">>> SKIP pretrain: $OUT/retriever_pretrain.pt already exists"
    continue
  fi
  if [[ "$STAGE" == "joint" && -f "$OUT/ur4rec_joint_meta.pt" ]]; then
    log ">>> SKIP joint: joint checkpoint and metadata already exist"
    continue
  fi
  if [[ "$STAGE" == "joint" && -f "$OUT/ur4rec_joint.pt" && ! -f "$OUT/ur4rec_joint_meta.pt" ]]; then
    CURRENT_STAGE="finish_joint"
    set_status "running stage=$CURRENT_STAGE pid=$$ gpu=$TRAIN_GPU"
    log ">>> STAGE finish_joint: recover metadata from saved joint checkpoint"
    CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python -u scripts/ur4rec/run_ur4rec.py \
      --config "$CONFIG" \
      --stage finish_joint \
      --gpu-id 0 \
      2>&1 | tee -a "$TRAIN_LOG" | tee -a "$MASTER_LOG"
    continue
  fi
  if [[ "$STAGE" == "eval" && -f "$METRICS" ]]; then
    log ">>> SKIP eval: $METRICS already exists"
    continue
  fi

  log ">>> STAGE $CURRENT_STAGE on physical GPU $TRAIN_GPU"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python -u scripts/ur4rec/run_ur4rec.py \
    --config "$CONFIG" \
    --stage "$CURRENT_STAGE" \
    --gpu-id 0 \
    2>&1 | tee -a "$TRAIN_LOG" | tee -a "$MASTER_LOG"
done

CURRENT_STAGE="verify_metrics"
set_status "running stage=$CURRENT_STAGE pid=$$ gpu=none"
python - "$METRICS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Missing final metrics: {path}")
metrics = json.loads(path.read_text(encoding="utf-8"))
required = {"base", "ur4rec", "ur4rec_pure", "blend_alpha"}
missing = sorted(required - set(metrics))
if missing:
    raise SystemExit(f"Final metrics missing keys: {missing}")
print(json.dumps(metrics, indent=2))
PY

T1="$(date +%s)"
set_status "done metrics=$METRICS"
log "========== RESUME FINISHED in $(( (T1 - T0) / 3600 ))h $(( (T1 - T0) % 3600 / 60 ))m =========="
log "Metrics: $METRICS"
