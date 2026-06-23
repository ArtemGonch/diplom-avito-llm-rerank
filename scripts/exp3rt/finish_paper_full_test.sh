#!/usr/bin/env bash
# Merge 4-stage LoRA chain to bf16 (if needed) and run vLLM test+eval (paper-full).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/exp3rt/amazon_book_qwen_paper_full.yaml"
MASTER="logs/exp3rt_paper_full_master.log"
TEST_LOG="logs/exp3rt_paper_full_test.log"
PAPER_ROOT="$ROOT/checkpoints/exp3rt/amazon_book_qwen_paper_full"
PREF_DIR="$PAPER_ROOT/amazon-book_preference_r128_alpha16_seed425"
USER_DIR="$PAPER_ROOT/amazon-book_user_r128_alpha16_seed425"
ITEM_DIR="$PAPER_ROOT/amazon-book_item_r128_alpha16_seed425"
RATING_DIR="$PAPER_ROOT/amazon-book_rating_r128_alpha32_seed425"
MERGED_INDEX="$RATING_DIR/merged/model.safetensors.index.json"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diplom_avito

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER"; }

if [ ! -f "$RATING_DIR/adapter_config.json" ]; then
  echo "Missing adapter: $RATING_DIR/adapter_config.json" >&2
  exit 1
fi

log "========== Exp3RT paper-full: merge (if needed) + test =========="

pkill -f "run_exp3rt.py.*paper_full.*train-stage rating" 2>/dev/null || true
sleep 2

if [ -f "$MERGED_INDEX" ] && [ "${EXP3RT_FORCE_MERGE:-0}" != "1" ]; then
  log ">>> SKIP merge (bf16 merged checkpoint exists)"
else
  MERGE_GPU="${EXP3RT_MERGE_GPU:-}"
  if [ -z "$MERGE_GPU" ]; then
    MERGE_GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F', ' '$2 + 0 >= 55000 {print $1}' | head -2 | paste -sd,)"
  fi
  log ">>> MERGE 4-stage chain to bf16 (GPUs ${MERGE_GPU:-auto})"
  CUDA_VISIBLE_DEVICES="$MERGE_GPU" python -u scripts/exp3rt/merge_rating_adapter.py \
    --adapter-dirs "$PREF_DIR" "$USER_DIR" "$ITEM_DIR" "$RATING_DIR" \
    --out-dir "$RATING_DIR/merged"
fi

log ">>> TEST + EVAL (4-stage merged model, auto GPU pick)"
export EXP3RT_CONFIG="$CONFIG"
export EXP3RT_TEST_LOG="$TEST_LOG"
export EXP3RT_AUTO_GPU=1
unset CUDA_VISIBLE_DEVICES
bash scripts/exp3rt/run_test_eval.sh 2>&1 | tee -a "$TEST_LOG" | tee -a "$MASTER"

if [ -f "$RATING_DIR/metrics.json" ]; then
  cp "$RATING_DIR/metrics.json" "$ROOT/results/current/metrics/exp3rt_paper_full_test_metrics.json"
  EXP3RT_METRICS_PATH="$RATING_DIR/metrics.json" python scripts/exp3rt/build_paper_comparison.py
  log "Metrics: $RATING_DIR/metrics.json"
fi

log "========== paper-full test FINISHED =========="
