# Exp3RT reproduction (Qwen)

Paper: [Exp3RT SIGIR 2025](https://arxiv.org/abs/2408.06276)

## Layout

```
avito/
├── src/
│   ├── common/          # metrics, device_util, shared HF LLM backends
│   ├── data/            # MovieLens, Amazon, Steam, Avito loaders
│   ├── models/
│   │   ├── ur4rec/      # UR4Rec architecture (COLING 2025)
│   │   └── exp3rt/      # Exp3RT Qwen port (SIGIR 2025)
│   └── ur4rec/          # backward-compatible import shims
├── configs/
│   ├── ur4rec/
│   └── exp3rt/
├── scripts/
│   ├── ur4rec/
│   └── exp3rt/
├── checkpoints/
│   ├── ur4rec/
│   └── exp3rt/
└── data/exp3rt/         # symlink → papers/exp3rt/assets/github_repo/data
```

## Setup

```bash
conda activate diplom_avito
pip install -r requirements-exp3rt.txt
```

## Minimal reproduction (rating stage only)

Bundled JSON already contains stage-1/2 profiles inside rating examples.

```bash
# 1) Merge shards + validate paths
python scripts/exp3rt/run_exp3rt.py --config configs/exp3rt/amazon_book_qwen.yaml --stage prepare

# 2) Train stage-3 rating (~8-16 h on 8×A100)
CUDA_VISIBLE_DEVICES=4,5,6,7 EXP3RT_SINGLE_GPU=1 \
  python scripts/exp3rt/run_exp3rt.py --config configs/exp3rt/amazon_book_qwen.yaml --stage train --train-stage rating

# 3) Inference + metrics (~3-6 h)
CUDA_VISIBLE_DEVICES=4 python scripts/exp3rt/run_exp3rt.py --config configs/exp3rt/amazon_book_qwen.yaml --stage test
python scripts/exp3rt/run_exp3rt.py --config configs/exp3rt/amazon_book_qwen.yaml --stage eval
```

Or one command (rating only):

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/exp3rt/run_exp3rt.py \
  --config configs/exp3rt/amazon_book_qwen.yaml --stage all
```

## Full 3-stage pipeline

```bash
python scripts/exp3rt/run_exp3rt.py --config configs/exp3rt/amazon_book_qwen.yaml --stage all \
  --train-stages preference user item rating
```

## Time estimates (8× A100 80GB)

| Step | Hours |
|------|-------|
| preference + user + item (3× train) | ~12–24 |
| rating train | ~8–16 |
| vLLM test (11k rows) | ~3–6 |
| **Total (all stages)** | **~30–50** |
| **Rating-only (recommended first)** | **~12–24** |

## Outputs

- LoRA + merged model: `checkpoints/exp3rt/amazon_book_qwen/amazon-book_rating_r128_alpha32_seed425/merged`
- Predictions: `.../predictions_test.json`
- RMSE/MAE: `.../metrics.json`

Compare with paper Table 2 (Amazon-Books rating / rerank).
