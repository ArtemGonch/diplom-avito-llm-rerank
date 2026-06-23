# Преференсы пользователя на Avito Auto и выбор алгоритма

## Что есть в данных (нет Amazon-отзывов)

| Сигнал | Источник | Содержание |
|--------|----------|------------|
| SERP | `items_with_attrs.parquet` | query/category, geo, объявления в выдаче |
| Контакты | `users_with_history.parquet` | user → item, brand, model, price |
| Label | `contacts_daily` | нормализованный контакт по объявлению в SERP |

**Ограничение:** `item_id` в SERP и в history **не пересекаются** — нельзя напрямую перенести Exp3RT stage-1 (Like/Dislike из текста отзыва на target item).

---

## Какой метод релевантнее для извлечения преференсов

### 1. Exp3RT-style pseudo-profiles (наиболее релевантен сейчас)

**Идея:** собрать текстовый профиль из structured signals без отзывов.

```text
User profile ← top brands/models из contacts + median price
Item text    ← title, brand, model, price, mileage, fuel
Query        ← category + geo из SERP
Score        ← heuristic или LLM 1–5 fit
```

**Плюсы:** работает offline, объяснимо, уже есть скрипт `exp3rt_avito_attribute_rerank.py`.  
**Минусы:** нет pairwise reasoning; слабый сигнал если у seller нет history.

**Реализация:** `build_user_pseudo_profile()` + `heuristic_scores()` / LLM mode.

### 2. UR4Rec-style LLM knowledge (второй по релевантности)

**Иdea:** LLM генерирует item/user knowledge JSON → BERT encoder → retriever → DLCM.

**Плюсы:** хорошо масштабируется на много полей объявления; retriever отбирает нужное под кандидата.  
**Минусы:** дорого (offline LLM на все items/users); на smoke Avito UR4Rec ≈ base без query-conditioning.

### 3. Full Exp3RT 4-stage SFT (как в paper)

**Нерелевантен без синтетики:** нужны тексты «отзывов» → preference → profile. На Avito их нет; пришлось бы генерировать pseudo-reviews LLM'ом на contacts (дорого, риск hallucination).

---

## Рекомендация для диплома

**Краткосрочно (эксперименты сейчас):** Exp3RT-style **pseudo-profiles + attribute rerank** — быстро, interpretable, beat position baseline на smoke.

**Среднесрочно (собственный вклад):** **C-UR4Rec** — UR4Rec + контекст SERP (см. ниже).

---

## +1 алгоритм: C-UR4Rec (Context-aware UR4Rec)

**Проблема UR4Rec на Avito:** один статический user embedding на весь list; **нет search query**; retriever не видит кандидата в контексте запроса «BMW X5 Москва».

**C-UR4Rec** — три добавки к UR4Rec:

1. **Multi-aspect memory** — K proxy-аспектов (цена, бренд, geo, кузов) вместо одного concat preference.
2. **Query-conditioned retrieval** — cross-attention query = `concat(h_candidate, e_serp_query, e_category)`.
3. **Confidence gating** — `e_aug = gate · Retriever(...)`; если retriever не уверен → gate→0, fallback на base DLCM.

```text
Offline:  LLM knowledge(items) + LLM summary(user contacts)
Encode:   BERT → e_item, e_user_history, e_query
Retrieve: Retriever(h_cand, e_query, Z_user) → e_aug  (per candidate!)
Rerank:   DLCM(item, user, e_aug) with gate
```

**Почему это логичный next step после Exp3RT-style на Avito:**

| | Exp3RT-style Avito | C-UR4Rec |
|--|-------------------|----------|
| User prefs | rule-based text profile | LLM + retriever |
| Query in model | только в heuristic | в retriever |
| Per-candidate | отдельный score 1–5 | e_aug per item |
| Cost online | LLM optional | BERT only |

**План ablation:** base DLCM → +UR4Rec → +query → +gate → full C-UR4Rec на Avito SERP с unified top-K sampling.

См. также `docs/AGENT_HANDOFF.md` §9.
