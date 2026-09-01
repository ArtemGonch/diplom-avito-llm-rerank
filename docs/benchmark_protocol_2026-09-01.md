# Финальные benchmarks и протокол экспериментов

Версия: **2026-09-01**. Числа в этом документе продублированы только для
чтения; источники истины — JSON, перечисленные в таблицах.

## Общие правила

1. Candidate set и test split фиксируются до выбора prompt, blend или gate.
2. Все hyperparameters выбираются только по train/dev; test считается один раз.
3. Таргет разрешён как train label и dev/test relevance, но никогда как
   rank-time feature/prompt; остальные post-exposure признаки запрещены.
4. Основная метрика — graded `NDCG@10`; дополнительно `NDCG@5`, `MAP@10` или
   `MRR@10`, retrieval `Recall@100`, latency и доля LLM-вызовов.
5. Сравниваются только результаты с одинаковыми split/candidates/relevance.
6. Для stochastic методов — три seeds; для deterministic greedy/forward LLM
   указывается детерминированный режим и отдельно измеряется runtime.

Текущий `MAP@K` бинаризует `rel>0`. В Avito contacts signal плотный, поэтому
MAP получается высоким и малоразличимым; до согласования бинарного contact
threshold это только secondary diagnostic. Основные выводы делаются по graded
NDCG.

## Avito Auto

### Целевой командный протокол

| ID | Метод | Вход online | Роль | Текущий статус |
|---|---|---|---|---|
| A0 | исходный position order | position | sanity baseline | пересчитан локально |
| A1 | командный CatBoost Ромы | только rank-time features | baseline диплома | **blocked: artifacts отсутствуют** |
| A2 | pseudo-profile heuristic | query proxy + attrs | дешёвый reference | **не реализован валидно**: прежний artifact вырожден по scores |
| L0 | no-history LLM pointwise | query proxy + attrs | вклад LLM без personalization | локальный diagnostic **done** |
| L1 | offline profile + pointwise | profile + query + attrs | вклад user memory | blocked по temporal cutoff |
| L2 | offline profile + listwise top-K | profile + query + list | inter-item reasoning | blocked по temporal cutoff/A1 candidates |
| L3 | pre-inference gated LLM+CatBoost | cheap CatBoost ambiguity → optional LLM | quality/cost method | локальный no-history control **done, negative**; personalized claim blocked |

Командный A1 и финальные L0–L3 должны использовать один список test SERP и
одни candidate rows. Relevance:

```text
rel(q, i) = max(contacts_daily(q, i), 0) /
            max_j max(contacts_daily(q, j), 0)
```

`contacts_daily`, `clicks_daily`, `has_tc_events`, `has_x_events` и
`serp_is_positive` запрещены в features/prompts. История может содержать только
события со временем `< rank_time(SERP)`.

### Что уже измерено локально

Локальный diagnostic использует deterministic SERP-disjoint `80/10/10`,
`split_seed=42`, все строки каждого SERP и CatBoostRanker
`YetiRankPairwise`, 300 GPU iterations, seeds `42/43/44`. Он нужен для проверки
evaluator/artifact contract и **не заменяет A1 Ромы**.

| Метод | Test SERP | NDCG@5 | NDCG@10 | Источник |
|---|---:|---:|---:|---|
| position order | 200 | 0.193886 | 0.302670 | `results/current/metrics/avito_local_catboost_diagnostic.json` |
| local CatBoost ensemble | 200 | 0.584950 | **0.653349** | тот же JSON |
| local CatBoost, mean ± std по seeds | 200 | 0.586872 ± 0.000715 | **0.657644 ± 0.002592** | тот же JSON |
| Qwen2.5-7B L0 pointwise, no history | 200 | 0.252425 | 0.353667 | `results/current/metrics/avito_local_llm_diagnostic.json` |
| local gated CatBoost+L0 | 200 | 0.571184 | 0.638136 | тот же JSON |
| invalid Exp3RT-style tie diagnostic | 200, другой split | 0.239336 | 0.341060 | `results/current/metrics/exp3rt_avito_full_leakage_free.json`, `valid_for_claims=false` |

У CatBoost per-seed metric и score-ensemble metric закономерно различаются:
первая строка стабильности усредняет три значения метрики, вторая сначала
усредняет score каждого кандидата и только затем считает NDCG.

Exp3RT-style число в последней строке **не является результатом модели**:
schema audit показал, что history `brand`/`model_name` — несовместимые с Auto
category-like значения, price в history отсутствует, а после исправления
семантики 200/200 SERP имеют одинаковые candidate scores. NDCG определяется
tie/order policy. Artifact сохранён только как отрицательный audit; A2 остаётся
пустой ячейкой benchmark matrix.

L0 — детерминированный next-token score `E[label]`, `label∈{0,…,4}`, по
Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Prompt содержит только allow-list
query/category/location proxies и item attributes; position, contacts, clicks
и history исключены, ties разрешаются по candidate id. L0 лучше position на
`+0.050997` NDCG@10, но ниже local CatBoost на `−0.299682`.
Полный прогон 8 884 dev+test candidates занял `318.3 s` на A100, включая
загрузку модели и scoring.

Gate выбирал на dev `alpha=0.4` и CatBoost top-two margin threshold `0.170985`:
dev NDCG@10 вырос `0.653381 → 0.654566`. На test результат ухудшился
`0.653349 → 0.638136`, хотя simulated LLM-call fraction составила `46%`.
Следовательно, этот простой margin gate **отклонён**: dev gain не перенёсся на
test. Для I3 нужны более устойчивые pre-generation features и nested/несколько
validation folds; test threshold повторно не подбирается.

Для финального A1 от Ромы нужны:

Текущий history extract содержит 2 028 contacts для 274 users и пересекается
только с 295/2 000 SERP. На local seed-42 split history покрывает 34/200 dev и
30/200 test SERP. Поля `brand`/`model_name` заполнены лишь на 49.46%/13.51%,
имеют category-like семантику и нулевое vocabulary overlap с автомобильными
listing attrs; price/year/mileage/gearbox/fuel history недоступны. Кроме
временного cutoff финальный protocol должен заранее определить schema mapping
и cold-start/coverage policy.

- test SERP ids и полный candidate list;
- candidate score или checkpoint + точный feature config;
- split/config revision и seeds;
- точная relevance formula;
- timestamp/cutoff для формирования user history.

### Минимальный executable artifact contract

Файл scores — Parquet с одной строкой на candidate:

| Поле | Тип | Ограничение |
|---|---|---|
| `split` | string | `dev` или `test` |
| `serp_x` | string | group id |
| `item_id` | int64 | уникален внутри SERP |
| `catboost_score` | float64 | только rank-time inference |

Evaluator делает exact one-to-one join по `split, serp_x, item_id`; отсутствие,
дубликат или лишний candidate — ошибка, а не silently dropped row.

## Amazon-C4 User Purchase History / Automotive

Данные: 41 962 Automotive items; 444/72/76 train/dev/test queries. Reference
поля `ori_review` и `ori_rating` не входят в retrieval query. Положительный item
не вставляется в candidates искусственно.

### Retrieval

| Method | Candidate set | Dev Recall@100 | Test Recall@100 | Test MRR@10 | Test NDCG@10 |
|---|---|---:|---:|---:|---:|
| BM25 | собственный immutable top-100 | 0.180556 | 0.157895 | 0.021303 | 0.028509 |
| `hyp1231/blair-roberta-base` | собственный immutable top-100 | **0.583333** | **0.592105** | **0.155310** | **0.196720** |

Для downstream rerank зафиксирован BLaIR top-100. Источник:
`results/current/metrics/amazon_c4_automotive_retrieval.json`.

### Изображения и multimodal control

Из union фиксированных candidates найдено 21 558 уникальных items. Валидно
скачано 21 545 изображений: общая coverage `99.9397%`, dev `99.9110%`, test
`99.9674%`. Missing/failed items остаются в candidate set.

`MM2` использует `w·z(BLaIR) + (1-w)·z(CLIP-image)`, где `w=0.4` выбран по dev
`NDCG@10`; test не участвовал в подборе.

| Method на том же BLaIR top-100 | Dev NDCG@10 | Test NDCG@10 | Test MRR@10 | Test Hit@10 |
|---|---:|---:|---:|---:|
| MM0 text-only BLaIR | 0.166218 | 0.196720 | 0.155310 | 0.328947 |
| MM1 CLIP image-only | 0.070657 | 0.079851 | 0.067544 | 0.118421 |
| MM2 late fusion | **0.182471** | **0.226064** | **0.195734** | **0.328947** |

MM2 улучшил test `NDCG@10` на `+0.029344` absolute (`+14.92%` relative) и
`MRR@10` на `+0.040424` (`+26.03%` relative), не изменив `Recall@100` и
candidate set. Источник:
`results/current/metrics/amazon_c4_automotive_multimodal.json`.

Следующий честный ряд на том же top-100: cross-encoder → recency/category → raw
history → static profile → corrected UR4Rec → C-UR4Rec. Каждый новый reranker
должен сохранять BLaIR `Recall@100=0.592105`, потому что retrieval candidates не
меняются.

## MovieLens-1M: corrected UR4Rec control

Corrected-v3 завершён на temporal-per-user split и random top-100. Это
correctness-controlled run, но не paper-exact candidate generation.

| Method | NDCG@5 | NDCG@10 | MAP@10 |
|---|---:|---:|---:|
| local DLCM-style backbone | **0.175297** | **0.214796** | **0.162291** |
| pure UR4Rec, α=1 | 0.143121 | 0.183334 | 0.134540 |

UR4Rec хуже backbone на `0.031462` NDCG@10 (`−14.65%`). Это negative result,
который мотивирует conditioned selection и confidence gate; он не даёт права
менять test split или включать base fallback под названием UR4Rec.

## Три собственные идеи и их DoD

### I1. Multi-aspect preference memory

- **Гипотеза:** один общий profile смешивает устойчивые и контекстные вкусы;
  раздельные аспекты улучшают выбор кандидата.
- **Реализация:** offline JSON memory по `brand/model`, `price`, `body/specs`,
  `geo`, `recency`; у каждого аспекта текст, embedding, support count и cutoff.
- **Ablation:** no history → raw last-H → single profile → K aspects; одинаковые
  candidates/ranker.
- **Успех:** прирост dev/test NDCG@10 при неизменном leakage audit; отдельно
  slices по history length и sparsity.

### I2. Query–candidate-conditioned selection

- **Гипотеза:** для конкретного объявления полезна малая часть user memory;
  static concatenation добавляет шум.
- **Реализация:** retrieval query `concat(e_query, e_candidate, e_category)`;
  top-m аспектов/событий с mask и fallback на zero memory.
- **Ablation:** static profile → query-only selection → query+candidate
  selection; `m ∈ {1,2,4,all}` выбирается по dev.
- **Успех:** NDCG@10 выше single-profile, особенно на длинной истории; attention
  audit подтверждает несколько memory keys, а не collapsed one-token memory.

### I3. Calibrated confidence/cost gate

- **Гипотеза:** LLM помогает прежде всего на неоднозначных SERP, поэтому
  selective invocation сохраняет качество при меньшей стоимости.
- **Реализация:** до LLM использовать CatBoost margin/entropy, history coverage и
  model-aware checklist; threshold выбирать на validation Pareto frontier.
- **Ablation:** CatBoost-only → always-LLM → ungated blend → pre-inference gate;
  дополнительно 25/50/100% LLM budget.
- **Успех:** `NDCG@10(L3) > NDCG@10(A1)` либо Pareto-dominance: равное качество
  при существенно меньшей доле LLM calls/tokens.

Все три идеи — последовательные блоки C-UR4Rec: `structured memory → relevant
memory selection → cost-aware invocation`, а не три независимых метода.
