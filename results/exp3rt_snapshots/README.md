# Exp3RT experiment index

## Frozen: rating-only run (already beats paper on test)

**Snapshot:** `results/exp3rt_snapshots/rating_only_qwen_2026-06-17/`

| Metric | Ours (test, n=11743) | Paper Exp3RT |
|--------|----------------------|--------------|
| RMSE (expected_rating) | **0.5906** | 0.6508 |
| MAE (expected_rating) | **0.3431** | 0.4297 |
| RMSE (max_prob_rating) | **0.6314** | 0.6508 |
| MAE (max_prob_rating) | **0.3116** | 0.4297 |

Config: `configs/exp3rt/amazon_book_qwen.yaml` — rating stage only, 1 epoch, cutoff 1024.

## In progress: paper-full sequential run

**Config:** `configs/exp3rt/amazon_book_qwen_paper_full.yaml`  
**Script:** `scripts/exp3rt/run_paper_full.sh`  
**GPU:** 4 (train), 4–5 (test)  
**Logs:** `logs/exp3rt_paper_full_master.log`, `logs/exp3rt_paper_full_train.log`

Pipeline: preference → user → item → rating (chained merged weights) → vLLM test → eval.

Paper protocol: cutoff 1200, LoRA r=128, α=16 (stages 1–2) / α=32 (rating), 3+5 epochs, full val.

**Monitor:**
```bash
tail -f logs/exp3rt_paper_full_master.log
```

**ETA (rough):** preference ~12–18h, user/item ~3–5h each, rating ~15–25h, test ~20min → **~2–3 days** total.
