# Exp3RT Avito adaptation — pseudo-reviews MVP

**Date:** 2026-06-16  
**Script:** `scripts/exp3rt_avito_attribute_rerank.py`  
**Config:** `configs/exp3rt_avito_smoke.yaml`

## Problem

Exp3RT expects Amazon/IMDB **review text** for three stages (preference → profile → CoT rating). Avito Auto has structured listing attributes and contact history, not reviews.

## Mapping (Exp3RT → Avito)

| Exp3RT input | Avito proxy |
|--------------|-------------|
| Item review body | `title + brand + model + price + mileage + fuel/body` via `item_pseudo_description()` |
| User review history | `users_with_history.parquet` → pseudo profile: brands, models, median contact price |
| Query context | `AvitoSERP.serp_query_text()` — category + geo (not in original Exp3RT) |
| Rerank candidates | Full SERP items (`items_with_attrs.parquet`) |
| Label | Normalized `contacts_daily` (same as UR4Rec Avito smoke) |

## Implementation

Two modes in one script:

1. **`heuristic`** (default) — attribute overlap scoring: brand history, query token overlap, popularity prior, price fit. Fast, no GPU.
2. **`llm`** — Qwen2.5-7B scores each item 1–5 on brand/price/query fit (Exp3RT stage-3 analogue, simplified). Uses `hf_chat_generator.py`.

User profile construction skips gracefully when `user_id` is missing in SERP (1360 rows) or when the user has no rows in `users_with_history.parquet` (274 users with history, all overlap with item `user_id`s).

## Results (test split, 100 SERPs)

Source: `reproduction/results/avito_exp3rt_mvp.json`

| Method | NDCG@5 | NDCG@10 | MAP@10 |
|--------|--------|---------|--------|
| Position baseline (DLCM proxy) | 0.879 | 0.888 | 0.826 |
| **Exp3RT-style heuristic** | **0.952** | **0.946** | **0.910** |

Heuristic attribute rerank beats block position by **+5.8 NDCG@10** on the smoke split.

### Comparison with UR4Rec smoke (same parquet, Qwen knowledge)

From `checkpoints/ur4rec_avito_smoke_qwen/metrics_test.json`:

| Method | NDCG@10 | MAP@10 |
|--------|---------|--------|
| DLCM base | 0.929 | 0.884 |
| UR4Rec (trained) | 0.930 | 0.886 |
| Exp3RT-style heuristic | 0.946 | 0.910 |

**Note:** Protocols differ slightly — UR4Rec subsamples 20 candidates per SERP with random negatives; Exp3RT MVP scores all items on the page (full SERP). Treat as directional, not apples-to-apples. Unified table in [results/baseline_comparison.md](results/baseline_comparison.md).

## LLM mode (Qwen2.5-7B-Instruct, 5 SERPs smoke)

Source: `reproduction/results/avito_exp3rt_llm_smoke.json`

| Method | NDCG@5 | NDCG@10 | MAP@10 |
|--------|--------|---------|--------|
| Position baseline | 0.931 | 0.910 | 0.838 |
| Exp3RT-style LLM (1–5 scores) | 0.932 | 0.897 | 0.840 |

On 5 SERPs LLM scoring ≈ position baseline (unlike heuristic on 100 SERPs). Per-item 1–5 prompts need pairwise comparison and/or fine-tuning to match paper gains.

Run full LLM eval:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/exp3rt_avito_attribute_rerank.py \
  --mode llm --max-serps 10 \
  --output papers/exp3rt/reproduction/results/avito_exp3rt_llm_smoke.json
```

## Limitations

- No fine-tuned Exp3RT LoRA on Avito (would need pseudo-review SFT data at scale).
- Heuristic uses listing-side `user_id`; many SERPs lack buyer history → empty profile falls back to query-only signals.
- LLM mode is per-item scoring, not pairwise A-vs-B comparison (backlog item in `paper_improvements_backlog.md`).

## Next steps

1. Pairwise LLM comparison prompt (items A vs B on attribute checklist).
2. Query-conditioned profiles (SERP context in stage-2 text).
3. Align candidate sampling with UR4Rec (top-100, same seed) for fair baseline table.
4. Full Exp3RT stage-3 fine-tune on synthetic Avito rationales if pseudo-data quality holds.
