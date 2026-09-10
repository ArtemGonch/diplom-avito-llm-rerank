# Ранжированный обзор литературы для диплома

Проверено по первичным страницам публикаций и официальным репозиториям:
**2026-09-01**. Эта таблица — инструмент выбора идей для диплома, а не оценка
общей научной значимости работ.

## Шкала

| Код | Критерий | Баллы |
|---|---|---:|
| `F` | Близость к personalized product search/rerank: query, history, candidates | 0–3 |
| `N` | Новизна для нашей линии: memory, conditioning, routing/gate | 0–2 |
| `E` | Надёжность эксперимента: baselines, ablation, leakage/fixed candidates | 0–2 |
| `R` | Воспроизводимость: публичные data/code/checkpoints | 0–2 |
| `P` | Практичность: offline compute, latency, cache | 0–1 |

При равной сумме выше ставится работа с более прямой пользой для
`query + history + fixed candidates`, затем с меньшей стоимостью переноса.

## Итоговый рейтинг

| # | Работа | Статус | F | N | E | R | P | Σ/10 | Что берём / главное ограничение |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | [MemRerank](https://arxiv.org/abs/2603.29247) | arXiv 2026 | 3 | 2 | 2 | 1 | 1 | **9** | Сжатая preference memory, downstream reward; пока 1-in-5 benchmark и не подтверждён официальный code release |
| 2 | [A Zero Attention Model for Personalized Product Search](https://www.amazon.science/publications/a-zero-attention-model-for-personalized-product-search) | CIKM 2019 | 3 | 2 | 2 | 0 | 1 | **8** | Научное основание для «когда персонализировать»; proprietary logs/code |
| 3 | [UR4Rec](https://aclanthology.org/2025.coling-main.45/) | COLING 2025 | 2 | 2 | 2 | 1 | 1 | **8** | Candidate-aware retrieval из LLM preference/knowledge; нет официального кода и search context |
| 4 | [BLaIR / Amazon-C4](https://arxiv.org/abs/2403.03952) | arXiv 2024 | 2 | 1 | 2 | 2 | 1 | **8** | Публичные data/code/checkpoint и честный внешний retrieval test; semi-synthetic query и не Avito |
| 5 | [Think When Needed](https://arxiv.org/abs/2601.18146) | SIGIR 2026 | 2 | 2 | 2 | 1 | 1 | **8** | Pre-generation router и Pareto quality/cost; не решает построение user memory |
| 6 | [Rec-R1](https://arxiv.org/abs/2503.24289) | TMLR 2025 | 2 | 2 | 1 | 2 | 0 | **7** | Downstream reward для генератора и официальный [код](https://github.com/linjc16/Rec-R1); RL дорог и сложнее основной offline-линии |
| 7 | [LettinGo](https://arxiv.org/abs/2506.18309) | KDD 2025 | 2 | 2 | 2 | 1 | 0 | **7** | DPO по downstream-качеству профиля; обучение profile LLM тяжелее фиксированного extractor |
| 8 | [Persona4Rec](https://arxiv.org/abs/2602.21756) | arXiv 2026 | 2 | 2 | 1 | 1 | 1 | **7** | Offline multi-persona item index и дешёвый online score; review-centric и пока preprint |
| 9 | [Exp3RT](https://arxiv.org/abs/2408.06276) | SIGIR 2025 | 1 | 2 | 2 | 2 | 0 | **7** | Многоступенчатые user/item profiles и reasoning; rating/review setting, дорогая цепочка fine-tuning |
| 10 | [Hint-Augmented Re-ranking](https://www.amazon.science/publications/hint-augmented-re-ranking-efficient-product-search-using-llm-based-query-decomposition) | AACL 2025 | 2 | 1 | 2 | 1 | 1 | **7** | Offline/parallel query hints для лёгкого reranker; нет personalization и открытого production data |
| 11 | [LLM4Rerank](https://arxiv.org/abs/2406.12433) | WWW 2025 | 1 | 2 | 2 | 1 | 0 | **6** | Явные accuracy/diversity/fairness goals; online CoT плохо масштабируется |
| 12 | [Netflix GenRec](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3) | industry blog, 2026 | 1 | 2 | 2 | 0 | 1 | **6** | Verbalization, catalog head, prefill-only serving и context budget; proprietary и не peer-reviewed baseline |
| 13 | [Web-scale Semantic Product Search](https://www.amazon.science/publications/web-scale-semantic-product-search-with-large-language-models) | PAKDD 2023 | 1 | 1 | 2 | 0 | 1 | **5** | Teacher→малый bi-encoder и production latency; retrieval без user memory |
| 14 | [High-precision query-product semantic similarity](https://www.amazon.science/publications/improving-relevance-quality-in-product-search-using-high-precision-query-product-semantic-similarity) | ECNLP 2022 | 1 | 0 | 2 | 0 | 1 | **4** | Offline cross-encoder score как rank feature; закрытые данные и нет personalization |
| 15 | [Building a natural-language interface for product search](https://www.amazon.science/publications/building-natural-language-interface-for-product-search) | CIKM 2024 | 1 | 1 | 1 | 0 | 1 | **4** | Синтетические LLM queries/schema для cold-start; задача API generation, а не rerank |

## Top-5 для текста диплома

### 1. MemRerank

- Заимствуем: отделить query-independent сжатие истории от online rerank и
  оценивать память только через downstream ranking.
- Ограничение: заявленный benchmark — выбор одного товара из пяти, поэтому его
  результат нельзя напрямую сопоставлять с Avito `NDCG@10`.

### 2. Zero Attention

- Заимствуем: personalization должна быть условной — gate зависит от текущего
  query и совместимости query с purchase history.
- Ограничение: коммерческие логи и реализация не открыты; воспроизводим идею
  через прозрачный validation-selected gate, а не число статьи.

### 3. UR4Rec

- Заимствуем: bank из LLM knowledge/preferences и candidate-conditioned
  retrieval вместо передачи всей истории в prompt.
- Ограничение: original setting — recommendation rerank без явного поискового
  query; локальный corrected-v3 ещё и использует random top-100, а не
  paper-exact candidate generator.

### 4. BLaIR / Amazon-C4

- Заимствуем: официальный dense checkpoint, immutable top-100 и внешний
  product-search transfer benchmark, к которому можно присоединить изображения.
- Ограничение: запросы semi-synthetic; Automotive — автотовары, а не объявления
  о продаже автомобилей.

### 5. Think When Needed

- Заимствуем: решение о дорогом LLM-вызове принимается до generation по дешёвым
  ranking/model-aware сигналам, threshold выбирается только на validation.
- Ограничение: router экономит reasoning tokens, но сам по себе не строит
  персонализированную память и требует отдельной проверки на Avito.

## Вывод для нашей архитектуры

Литература поддерживает одну связную гипотезу: offline строить компактную
multi-aspect memory (`MemRerank`, `LettinGo`, `Persona4Rec`), выбирать её части
по текущему query/candidate (`UR4Rec`, Zero Attention), а дорогую ветку включать
только на неоднозначных SERP (`Think When Needed`, GenRec). BLaIR/Amazon-C4 даёт
публичный transfer test; Avito остаётся главным доменным тестом на контактах.

GenRec полезен как production evidence: Netflix описывает catalog-aware head,
prefill-only scoring, context compaction и online A/B test, но из-за закрытых
данных/моделей он не включается в численную benchmark-таблицу диплома.
