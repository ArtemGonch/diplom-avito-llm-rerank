#!/usr/bin/env bash
# Merge 4-stage LoRA chain to bf16 and run vLLM test+eval (paper-full).
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

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diplom_avito

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER"; }

if [ ! -f "$RATING_DIR/adapter_config.json" ]; then
  echo "Missing adapter: $RATING_DIR/adapter_config.json" >&2
  exit 1
fi

log "========== Exp3RT paper-full: merge chain + test =========="

pkill -f "run_exp3rt.py.*paper_full.*train-stage rating" 2>/dev/null || true
sleep 2

MERGE_GPU="${EXP3RT_MERGE_GPU:-5,6}"
log ">>> MERGE 4-stage chain to bf16 (GPUs $MERGE_GPU)"
CUDA_VISIBLE_DEVICES="$MERGE_GPU" python -u scripts/exp3rt/merge_rating_adapter.py \
  --adapter-dirs "$PREF_DIR" "$USER_DIR" "$ITEM_DIR" "$RATING_DIR" \
  --out-dir "$RATING_DIR/merged"

log ">>> TEST + EVAL (4-stage merged model)"
export EXP3RT_CONFIG="$CONFIG"
export EXP3RT_TEST_LOG="$TEST_LOG"
export CUDA_VISIBLE_DEVICES="${EXP3RT_TEST_GPUS:-5,6}"
bash scripts/exp3rt/run_test_eval.sh 2>&1 | tee -a "$TEST_LOG" | tee -a "$MASTER"

if [ -f "$RATING_DIR/metrics.json" ]; then
  cp "$RATING_DIR/metrics.json" "$ROOT/results/current/metrics/exp3rt_paper_full_test_metrics.json"
  EXP3RT_METRICS_PATH="$RATING_DIR/metrics.json" python scripts/exp3rt/build_paper_comparison.py
  log "Metrics: $RATING_DIR/metrics.json"
fi

log "========== paper-full test FINISHED =========="
