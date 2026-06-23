# Exp3RT Amazon-Books smoke log

**Date:** 2026-06-16  
**Script:** `scripts/exp3rt_repo_smoke.py`  
**Repo:** `papers/exp3rt/assets/github_repo/` (clone of [jieyong99/EXP3RT](https://github.com/jieyong99/EXP3RT))

## Result: PASS

### Merge train shards

- Merged `train_0.json`, `train_1.json`, `train_2.json` → `train.json`
- **94,669** rating-bias training rows

### Bundled dataset counts (amazon-book)

| Artifact | Records |
|----------|---------|
| preference_train_0 | 31,557 |
| user_train | 8,608 |
| item_train | 8,067 |
| rating_train | 94,669 |
| rating_valid | 12,293 |
| rating_test | 11,743 |
| topk train.txt | 10,052 lines |

### Schema notes

- Stages 1–2 use Alpaca `instruction/input/output`.
- Stage 3 `rating_bias/*.json` uses keys: `user_id`, `item_id`, `user_persona`, `item_description`, `item_synopsis`, `rationale`, `score`.
- Preference sample output starts with `[Like]` bullet list (matches paper).

### Not run (blocked)

| Step | Blocker |
|------|---------|
| `data_gen/generate.sh` | Needs teacher LLM API / raw review pipeline |
| `shell/train_*.sh` | Gated `meta-llama/Meta-Llama-3-8B-Instruct` |
| `shell/test_amazon-book.sh` | Requires trained LoRA + vLLM |
| `train_amazon-book.sh` | Bug: `$rmse_patience` undefined (should be `$patience`) |

### Raw Amazon upstream

Our gzip reviews live at `data/amazon-books/reviews_Books_5.json.gz` (3.1 GB) for future custom data_gen; repo JSON is sufficient for pipeline validation.

## Next steps

1. Obtain HF access to Llama-3 or map prompts to Qwen + our `diplom_avito` env.
2. Fix `rmse_patience` typo before stage-3 train.
3. Train stage 3 only on subset for partial Table reproduction.
