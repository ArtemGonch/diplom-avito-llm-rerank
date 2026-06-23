# Exp3RT — talk outline (~20–25 min)

**Audience:** MIPT seminar / Avito diploma committee  
**Speaker:** Goncharov Artem, M05-513v · supervisor Roman Budylin  
**Paper:** Kim et al., SIGIR 2025 · slides 8–9 of literature review

---

## Slide 1 — Title

- **Exp3RT:** Review-driven Personalized Preference Reasoning with LLMs
- Why this paper now: code + reasoning rerank + bridge to Avito attribute comparison

---

## Slide 2 — Problem

- LLM rec often uses **too little text** and **too short outputs**
- Reviews contain **subjective preferences** — underused by ID-based rerankers
- Goal: accurate rating/rerank **and** faithful explanations

**Bullets:** limited input · no CoT · explainability gap · citation SIGIR 2025

---

## Slide 3 — Method overview (Figure / pipeline)

```text
Review → [1] Like/Dislike → [2] User/Item profile → [3] CoT + rating
```

- 3-stage **distillation** into Llama-3-8B (QLoRA)
- Separate models per stage; merge for inference

**Bullets:** preference extraction · profile construction · reasoning-enhanced rating

---

## Slide 4 — Stage 1–2: Profiles

- Extract like/dislike **per review** (not one blob per user)
- Aggregate into **criteria-based profiles** (user vs item)
- Handles **unseen** users/items if new review text exists

**Bullets:** subjective vs objective · aggregation · cold-start with text

---

## Slide 5 — Stage 3: Reasoning-enhanced rating

- Prompt: profiles + item description → **step-by-step rationale** → digit 1–5
- Used for **rating prediction** and **top-k rerank** (score candidates)
- Inference: vLLM + digit logprobs

**Bullets:** CoT · rating digit · rerank by predicted score

---

## Slide 6 — Experiments

- **Amazon-Books**, **IMDB**
- Metrics: RMSE/MAE + rerank accuracy
- Baselines: MF, LGN, BERT-AF, prompting LLMs
- Claim: strong on **unseen** entities vs ID-only models

**Bullets:** review-rich datasets · rating + rerank · unseen generalization

---

## Slide 7 — Code walkthrough

- Repo: [github.com/jieyong99/EXP3RT](https://github.com/jieyong99/EXP3RT)
- `data/amazon-book/*.json` — preprocessed stages
- `shell/train_*.sh` → `merge.sh` → `test_amazon-book.sh`
- Our clone: `avito/papers/exp3rt/assets/github_repo/`

**Bullets:** bundled JSON · 3 train scripts · vLLM test · smoke doc

---

## Slide 8 — Our reproduction status

- Smoke: merge train shards, validate JSON schema, count samples
- Blockers: Llama-3 gated; no public checkpoints; full QLoRA ~multi-GPU-days
- Next: Qwen zero-shot stage-3 **style** prompts on Avito pseudo-reviews

**Bullets:** pipeline verified · train deferred · Avito adaptation path

---

## Slide 9 — Avito transfer

- **No reviews** → pseudo-reviews from contacts + listing attrs
- LLM compares candidates on **price, brand, mileage, query**
- Script: `scripts/exp3rt_avito_attribute_rerank.py`
- Link to **C-UR4Rec**: add query-conditioned profiles + gating

**Bullets:** pseudo-reviews · attribute checklist · SERP query · backlog in `paper_improvements_backlog.md`

---

## Slide 10 — vs UR4Rec (our main baseline)

| Exp3RT | UR4Rec |
| explicit CoT reasoning | retriever over knowledge embeddings |
| needs review text | needs offline knowledge JSON |
| LLM fine-tune | frozen LLM + train retriever+DLCM |

**Bullets:** complementary · different cost profile · both rerank top-K

---

## Slide 11 — Limitations & backlog

- Latency & train cost
- Review dependency
- Small GitHub community
- Ideas: query-conditioned profiles, routing (TWn), multi-aspect (LLM4Rerank)

**Bullets:** see `docs/paper_improvements_backlog.md`

---

## Slide 12 — Takeaways

1. Exp3RT = **structured reasoning** from reviews, not just scores  
2. SIGIR 2025 + code → good deep-dive paper for semester 2  
3. Avito path = **pseudo-reviews + attribute comparison**  
4. Fits product story: offline LLM → rerank top-K, with explainability  

**Questions?**

---

## Likely audience questions

1. Why not fine-tune Exp3RT on Avito end-to-end? → no review corpus; pseudo-label cost  
2. How vs UR4Rec numbers? → see `reproduction/results/baseline_comparison.md`  
3. Latency for online? → not production-ready; Persona4Rec/TWn for online angle  
4. Explainability faithful? → Beyond Utility metrics in appendix of our review
