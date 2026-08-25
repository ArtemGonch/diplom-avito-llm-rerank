#!/usr/bin/env bash
# Quick status of long-running experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== GPU ==="
if gpu_status=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null); then
  printf '%s\n' "$gpu_status"
else
  echo "(GPU status unavailable in current namespace)"
fi
echo ""

echo "=== Processes ==="
ps aux | grep -E 'run_ur4rec|run_exp3rt|run_guaranteed' | grep -v grep || echo "(none visible in current namespace)"
echo ""

echo "=== UR4Rec corrected-v3 ==="
if [ -f checkpoints/ur4rec_ml1m_corrected_v3/metrics_test.json ]; then
  cat checkpoints/ur4rec_ml1m_corrected_v3/metrics_test.json
elif [ -f logs/ur4rec_corrected_v3/master.log ]; then
  tail -8 logs/ur4rec_corrected_v3/master.log
  for shard_log in logs/ur4rec_corrected_v3/knowledge_shard*.log; do
    [ -f "$shard_log" ] || continue
    progress=$(tr '\r' '\n' < "$shard_log" | grep -E 'llm-(items|users)' | tail -1 || true)
    progress=${progress%%The following generation flags*}
    [ -n "$progress" ] && printf '%s: %s\n' "$(basename "$shard_log")" "$progress"
  done
else
  echo "no log"
fi
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
