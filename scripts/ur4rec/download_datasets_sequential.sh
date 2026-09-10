#!/usr/bin/env bash
# Download missing UR4Rec datasets one by one with separate logs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diplom_avito

echo "=== status before download ==="
python scripts/download_ur4rec_datasets.py --status | tee logs/datasets_status.log

for DS in amazon-books steam; do
  LOG="logs/datasets_${DS}.log"
  echo ""
  echo "========== $(date '+%Y-%m-%d %H:%M:%S') start ${DS} ==========" | tee -a "$LOG"
  python -u scripts/download_ur4rec_datasets.py \
    --datasets "$DS" \
    --log-file "$LOG"
  echo "========== $(date '+%Y-%m-%d %H:%M:%S') done ${DS} ==========" | tee -a "$LOG"
done

echo ""
echo "=== status after download ==="
python scripts/download_ur4rec_datasets.py --status | tee -a logs/datasets_status.log
