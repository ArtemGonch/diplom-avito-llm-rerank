# Exp3RT — Phase 2: Deep reading notes

## Problem

LLM recommenders often use **limited input** (IDs, short history) and optimize **short outputs** (single score). Exp3RT uses **full review text** and **chain-of-thought** for rating + explanation.

## Method: 3 stages

```text
Stage 1 — Preference extraction (per review)
  Input:  raw user/item review
  Output: Like / Dislike (subjective preference snippet)

Stage 2 — Profile construction
  Input:  aggregated preferences for user or item
  Output: structured text profile (criteria-based summary)

Stage 3 — Reasoning-enhanced rating prediction
  Input:  user profile + item profile + item description
  Output: step-by-step rationale + rating digit (1–5 Amazon, 0–9 IMDB)
```

Training: **student LLM** (Llama-3-8B) fine-tuned with **QLoRA** per stage; teacher distillation from stronger LLM prompts (`data_gen/`).

Inference: merge LoRA adapters (`merge_llama.py`, `shell/merge.sh`) → **vLLM** generation → parse rating token from digit logprobs (`test.py`).

## Key design choices

1. **Review-driven** — subjective preferences extracted before aggregation (reduces noise vs one-shot summary).
2. **Separate user/item profile models** — `train_user.py`, `train_item.py` on profile JSON.
3. **Rationale before rating** — CoT in prompt (`data_gen/prompt/rationale.txt`); rating digit constrained to valid range.
4. **Top-k rerank** — candidate lists in `data/topk/amazon-book/`; rerank by predicted rating on pairs.

## Data layout (repo)

```text
data/amazon-book/
  preference_extraction/   preference_train_{0,1,2}.json → merge → train.json
  user_profile/            user_{train,valid,test}.json
  item_profile/            item_{train,valid,test}.json
  rating_bias/             train.json, valid.json, test.json (+ cold/unseen splits)
data/topk/amazon-book/     train/test candidate lists for rerank
```

Our raw Amazon 5-core gzip (`data/amazon-books/`) is **upstream** of this; repo ships **already distilled JSON** for reproduction shortcut.

## Loss / training (skim)

- `train_preference.py` — preference extraction SFT
- `train_user.py` / `train_item.py` — profile SFT
- `train.py` — rating + rationale SFT; early stopping on validation RMSE (`rmse_patience`)
- Hyperparams: see `shell/train_amazon-book.sh` — lr 2e-4, LoRA r=128, epochs=3, cutoff_len=1200

## vs UR4Rec (our baseline)

| | Exp3RT | UR4Rec |
|---|--------|--------|
| LLM role | Fine-tuned generator (3 stages) | Frozen offline generator |
| User signal | Review text → profile | Preference text → BERT embedding |
| Selection | Implicit in CoT | Explicit cross-attn retriever |
| Backbone | LLM rating / rerank score | DLCM/SASRec + e_aug |
| Avito fit | Needs pseudo-reviews | Needs knowledge JSON (in progress) |

## Reader notes (open questions)

- **Q:** How sensitive is rerank to profile length vs UR4Rec retriever compression? → Ablations in paper; we test via partial profile on Avito.
- **Q:** Can stage 3 run zero-shot with Qwen prompts only? → Our Avito MVP (`scripts/exp3rt_avito_attribute_rerank.py`).
- **Q:** `train_amazon-book.sh` `$rmse_patience` undefined — fix before full train.

## Files to read next

- `data_gen/prompt/rationale.txt` — stage-3 prompt template
- `test.py` — digit logprob parsing for Amazon 1–5
- `papers/exp3rt/reproduction/amazon_smoke.md` — our validation log
