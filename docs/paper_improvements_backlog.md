# Paper improvements backlog

Living document: ideas to improve methods from the literature review and our reproductions.  
Priority: items that close the **Avito gap** (search query, no reviews, latency, partial LLM coverage).

---

## Итерация 01.09–08.09.2026: Avito case browser

Исходное задание: [task_2026-09-01_artem.md](task_2026-09-01_artem.md),
отчёт: [task_2026-09-01_execution_report.md](task_2026-09-01_execution_report.md).

- [x] Собрать self-contained UI: history, SERP, Qwen/CatBoost ranks, graded
  relevance и отдельный actual TC marker.
- [x] Автоматически выбрать 8 контрастных error-analysis кейсов.
- [x] Проверить item/user LLM insights: 72 outputs, evidence guards,
  canonical batch size 1 и полный primary/retry + hardware/token provenance,
  strict schema/language/safety guards, exact seller-description spans,
  evidence-derived displayed fields и полный cache provenance contract
  (browser v5 / validation v9 / pinned-tokenizer+request+hardware+padded-token+attempt binding).
- [x] Собрать manifests на 264 featured и 4 614 all-test item IDs и overlay
  contract для картинок/описаний Ромы.
- [x] Собрать локальный portable ZIP для Google Drive; локальные изображения
  sandboxed каталогом overlay, декодером/allow-list и лимитом 20 MiB.
- [x] Загрузить в Google Drive полный пакет из семи файлов и проверить состав
  папки; выполнено 2026-09-10, рабочая ссылка хранится вне Git.
- [ ] Получить от Ромы изображения/метаданные и пересобрать UI
  с `--item-overlay`; исходный snapshot содержит только `image_count`.
- [ ] Получить team CatBoost scores и rank-time гарантию для честного
  L1/L2 personalized comparison.

---

## План на неделю 31.08–06.09.2026

Каноническая полная формулировка:
[task_2026-08-25_artem.md](task_2026-08-25_artem.md). Она объединяет блок
`2026-08-25` из командного Google-документа и добавленную задачу по изображениям.

- [x] Ранжированный обзор работ: общая 10-балльная шкала, core list и top-5.
- [x] Конечный список benchmarks и единый evaluation protocol.
- [x] Три собственные идеи: multi-aspect memory, conditioned selection, gate.
- [ ] Получить от Ромы CatBoost metric + split/candidates/config и воспроизвести.
- [ ] На одном CatBoost top-K сравнить no-history, pointwise profile, listwise
  profile и confidence-gated LLM rerank; цель — выше CatBoost по `NDCG@10`.
  Локальный L0/gate diagnostic готов, но CatBoost не превзойдён; L1/L2/L3
  ждут team artifacts и rank-time history cutoff.
- [x] Зафиксировать BM25/BLaIR top-100 для Amazon-C4 и `Recall@100`.
- [x] Скачать только candidate images, собрать coverage/manifest, закэшировать
  CLIP и сравнить text-only/image-only/late-fusion на том же top-100.

До получения CatBoost artifacts его `NDCG@10` остаётся `TBD`. Новый local
diagnostic CatBoost (`0.657788`) и L0 (`0.353148`) используют согласованный
внутренний split, но не подменяют командный baseline Ромы. Старый
pseudo-profile `0.3413` после schema/score audit признан invalid: 200/200 SERP
имеют одинаковые scores, поэтому это tie/order artifact, а не benchmark.

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

Current framing: [START_HERE.md](START_HERE.md) and [avito_preferences.md](avito_preferences.md):

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
