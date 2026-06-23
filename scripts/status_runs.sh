#!/usr/bin/env bash
# Quick status of long-running experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
echo ""

echo "=== Processes ==="
ps aux | grep -E 'run_ur4rec|run_exp3rt|run_guaranteed' | grep -v grep || echo "(none)"
echo ""

echo "=== UR4Rec guaranteed ==="
if [ -f checkpoints/ur4rec_ml1m_guaranteed/metrics_test.json ]; then
  echo "DONE: checkpoints/ur4rec_ml1m_guaranteed/metrics_test.json"
else
  tail -3 logs/guaranteed_master.log 2>/dev/null || echo "no log"
fi
echo ""

echo "=== Exp3RT paper_full ==="
if [ -f checkpoints/exp3rt/amazon_book_qwen_paper_full/amazon-book_rating_r128_alpha32_seed425/metrics.json ]; then
  cat checkpoints/exp3rt/amazon_book_qwen_paper_full/amazon-book_rating_r128_alpha32_seed425/metrics.json
else
  grep -oE "[0-9]+%\|[█▉▊▋▌▍▎▏ ]+\| [0-9]+/14795" logs/exp3rt_paper_full_master.log 2>/dev/null | tail -1 || echo "rating idle or done without metrics"
fi
echo ""

echo "=== Registry ==="
grep -E "status:|stage:|metrics:" experiments/registry.yaml | head -24
