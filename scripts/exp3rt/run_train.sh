#!/usr/bin/env bash
# Train Exp3RT stage with accelerate (optional multi-GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/exp3rt/amazon_book_qwen.yaml}"
STAGE="${2:-rating}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export NCCL_P2P_DISABLE=1

echo "Exp3RT train stage=$STAGE config=$CONFIG GPUs=$GPUS"

# Single-process fallback when peft/deepspeed not configured for multi-GPU
if [[ "${EXP3RT_SINGLE_GPU:-0}" == "1" ]]; then
  python scripts/exp3rt/run_exp3rt.py --config "$CONFIG" --stage train --train-stage "$STAGE"
  exit 0
fi

python scripts/exp3rt/run_exp3rt.py --config "$CONFIG" --stage train --train-stage "$STAGE"
