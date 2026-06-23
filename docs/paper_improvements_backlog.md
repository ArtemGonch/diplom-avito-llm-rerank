# Paper improvements backlog

Living document: ideas to improve methods from the literature review and our reproductions.  
Priority: items that close the **Avito gap** (search query, no reviews, latency, partial LLM coverage).

---

## Exp3RT → Exp3RT+Avito / Exp3RT+Search

- [ ] **Query-conditioned profiles** — inject SERP query (category, geo) into user profile construction, not only review history.
- [ ] **Pseudo-reviews from behavioral signals** — contacts/clicks as short text instead of Amazon review bodies.
- [ ] **Candidate-aware reasoning** — compare items A vs B on explicit attribute checklist (price, brand, mileage, query fit) before scalar rating.
- [ ] **Distillation budget** — single-stage Qwen teacher → small student instead of full 3-stage LoRA on 7B for production cost.
- [ ] **Cold-start item path** — item profile from structured attrs when no review text (Avito listings).
- [ ] **Latency gate** — run full CoT only on top-K after cheap LTR; TWn-style routing.
- [ ] **Calibration** — map LLM rating scores to contact probability, not only ranking (Beyond Utility axis).

---

## UR4Rec → C-UR4Rec (diploma contribution)

From [AGENT_HANDOFF.md §9](AGENT_HANDOFF.md):

- [ ] **Multi-aspect user memory** — K aspect vectors (price, brand, geo) instead of one concat `e_u^aggr`.
- [ ] **Candidate–context conditioned retrieval** — query = concat(h_cand, e_query, e_category).
- [ ] **Candidate item knowledge** — LLM knowledge for rerank candidates, not only history items.
- [ ] **Confidence gating** — `e_aug = gate · Retriever(...)`; gate→0 when retriever uncertain (smoke showed UR4Rec < base).
- [ ] **Partial knowledge robustness** — NDCG curves at 25% / 50% / 100% knowledge coverage.
- [ ] **Search-aware SERP objective** — rerank within query context on Avito parquet.

---

## KAR (secondary baseline)

- [ ] Compare **full KAR aug** vs **UR4Rec retriever** on same DLCM backbone and splits.
- [ ] Reuse our Qwen knowledge instead of Llama `.klg` for fair LLM comparison.
- [ ] Hybrid-expert adaptor vs cross-attention retriever — which compresses noise better?

---

## LLM4Rerank (WWW 2025)

- [ ] Import **multi-aspect Goal node** (accuracy + diversity + fairness) into Avito rerank eval.
- [ ] Replace expensive online CoT with offline aspect scores + lightweight reranker.
- [ ] Geo/brand diversity as explicit rerank constraint for auto vertical.

---

## LettinGo (KDD 2025)

- [ ] DPO on profile quality vs fixed SFT templates for Avito user profiles.
- [ ] Profile format search — avoid rigid bullet templates for heterogeneous auto intents.

---

## Persona4Rec (2026)

- [ ] Persona indexing for **offline** reasoning; dot-product online — target <1 ms rerank stage.
- [ ] Combine persona profiles with UR4Rec retriever (persona as proxy bank).

---

## Think When Needed (2026)

- [ ] Router: Think vs Non-Think before expensive LLM rerank on easy SERPs.
- [ ] Checklist probing for **model-aware difficulty** on Avito (cheap suffix prompts).

---

## Beyond Utility (evaluation)

- [ ] Report position bias, explainability faithfulness, not only NDCG@10.
- [ ] Idempotency test — same SERP, repeated LLM call → stable ranking.

---

## Cross-cutting

- [ ] Unified eval harness: ML-1M, Amazon-Books, Avito SERP — same metrics (NDCG@5/10, MAP@10).
- [ ] Cost table: GPU-h knowledge, ms/query online, $/1k SERPs.
- [ ] Shared Qwen2.5-7B-4bit backend across UR4Rec knowledge and Exp3RT-style reasoning prompts.
