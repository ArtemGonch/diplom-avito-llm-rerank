# Отчёт о выполнении задания 2026-08-25

Дата выполнения: **2026-09-01**.

## Сводка

| Пункт Google-документа / наша добавленная задача | Что сделано | Итог |
|---|---|---|
| Ранжировать литературу, проверить Netflix GenRec и Amazon | Проверены первичные страницы, введена единая шкала 10 баллов, оценены 15 работ, разобран top-5 | **Готово** |
| Зафиксировать финальные benchmarks | Определены IDs, split/candidate/relevance contract, leakage rules, metrics и DoD для Avito, Amazon-C4 и ML-1M | **Готово** |
| Сформулировать три собственные идеи | Multi-aspect memory, query–candidate selection, confidence/cost gate описаны как hypothesis → implementation → ablation → success | **Готово** |
| Запустить варианты LLM-ranking на Avito и превзойти CatBoost | Обучен local CatBoost control, запущен честный no-history Qwen L0, проверен dev-selected pre-inference gate | **Частично: CatBoost не превзойдён; personalized часть externally blocked** |
| Узнать CatBoost Ромы | В репозитории проверены config/split/scores/metric; их нет, сформирован exact artifact contract | **Ожидается от Ромы** |
| Amazon-C4 images | Зафиксирован BLaIR top-100, скачаны candidate-only images, построен CLIP control и late fusion | **Готово** |

## 1. Литература

Создан [ранжированный обзор](literature_ranked_review_2026-09-01.md):

- шкала `fit 0–3 + novelty 0–2 + experiment 0–2 + reproducibility 0–2 + practicality 0–1`;
- 15 научных/индустриальных работ, включая весь core list;
- top-5: MemRerank, Zero Attention, UR4Rec, BLaIR, Think When Needed;
- для каждой top-5 зафиксировано одно конкретное заимствование и одно
  ограничение;
- Netflix GenRec явно отделён как industry blog с proprietary evidence, а не
  peer-reviewed/public baseline;
- отдельно включены Amazon Zero Attention, Hint-Augmented reranking, web-scale
  semantic search, cross-encoder relevance и natural-language interface.

Главный вывод обзора: дипломная линия должна быть единой цепочкой
`compact memory → query/candidate selection → pre-inference cost gate`, а не
набором несвязанных LLM prompts.

## 2. Финальный benchmark protocol

Создан [единый протокол](benchmark_protocol_2026-09-01.md). Он фиксирует:

- immutable candidate sets;
- выбор параметров только на dev и однократный test;
- graded `NDCG@10` по контактам как главную Avito metric;
- запрет `contacts_daily`, `clicks_daily`, event flags и target-derived полей в
  rank-time features/prompts;
- exact one-to-one score artifact join без silently missing candidates;
- отдельные роли retrieval `Recall@100` и rerank `NDCG@10/MRR@10`;
- cost, LLM call fraction и stability как обязательные дополнительные axes.

Также исправлена старая ошибка в документации paper target UR4Rec: Appendix
Table 6 сообщает DLCM `NDCG@10 0.315 → 0.661`; значения `0.359 → 0.678`
относятся к `NDCG@20`.

## 3. Три собственные идеи

Для каждой идеи определены реализация и falsifiable ablation:

1. **Multi-aspect memory:** `brand/model`, `price`, `body/specs`, `geo`,
   `recency` против raw history и single profile.
2. **Query–candidate-conditioned selection:** static profile против query-only
   и query+candidate top-m memory retrieval.
3. **Confidence/cost gate:** CatBoost-only, always-LLM, ungated blend и
   pre-inference gate на budget levels 25/50/100%.

## 4. Avito: реализованные controls

### Local CatBoost

Добавлен `scripts/avito/run_local_catboost.py`:

- SERP-disjoint deterministic `80/10/10`, seed 42;
- все candidates каждого SERP без resampling;
- CatBoostRanker/YetiRankPairwise, 300 iterations, seeds 42/43/44;
- target/post-exposure columns исключены и защищены regression test;
- dev/test per-candidate score artifact сохраняется для exact join.

| Test metric | Position | CatBoost ensemble | Mean ± std по CatBoost seeds |
|---|---:|---:|---:|
| NDCG@5 | 0.193886 | 0.584950 | 0.586872 ± 0.000715 |
| NDCG@10 | 0.302670 | **0.653349** | **0.657644 ± 0.002592** |

Это local diagnostic, а не командный CatBoost Ромы.

### Qwen L0 и gate

Добавлен `scripts/avito/run_llm_pointwise_diagnostic.py`:

- Qwen2.5-7B-Instruct revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, deterministic next-token
  expectation по labels `0…4`;
- prompt использует только query category/location proxies и rank-time item
  attributes; history и position отсутствуют;
- 8 884 dev+test candidates обработаны за 318.3 s на A100, включая загрузку
  модели и scoring;
- cache позволяет продолжать и повторно считать evaluator без загрузки LLM;
- контрольный cached rerun повторно объединил все 8 884 candidate scores и
  воспроизвёл метрики без расхождений;
- простой gate решает до LLM по CatBoost top-two margin, alpha/threshold
  выбираются на dev.

| Method | Dev NDCG@10 | Test NDCG@10 | Test LLM-call fraction |
|---|---:|---:|---:|
| local CatBoost | 0.653381 | **0.653349** | 0% |
| Qwen L0, no history | 0.360456 | 0.353667 | 100% |
| gated CatBoost + L0 | **0.654566** | 0.638136 | 46% simulated |

L0 выше position на `+0.050997` absolute (`+16.85%` relative), но ниже
CatBoost на `0.299682` absolute. Gate дал небольшой dev gain `+0.001185`, но на
test потерял `0.015212`; поэтому gate **отклонён**, а threshold не
переподбирался на test. Это полезный negative result: одного category/location
proxy и pointwise semantic score недостаточно, а простой margin gate
переобучается на 200 dev SERP.

### Что объективно заблокировано

L1/L2/personalized L3 нельзя честно запускать на текущем extract:

- у SERP нет rank-time timestamp/cutoff;
- поэтому нельзя доказать, что contact history была доступна до target;
- history-user overlap есть лишь для 295/2 000 SERP (34 dev и 30 test SERP на
  local split), поэтому нужен заранее фиксированный coverage/cold-start policy;
- дополнительный schema audit показал, что экспортированные history
  `brand`/`model_name` содержат sparse category-like значения, не совпадающие
  с автомобильными listing brand/model; price и остальные авто-attrs истории
  отсутствуют;
- отсутствуют CatBoost split/candidates/scores/config Ромы;
- нельзя сделать утверждение `L3 > team CatBoost` на едином protocol.

Нужны от Ромы: test SERP ids, полный candidate score artifact, config revision,
seeds, relevance formula и rank-time cutoff. До этого `A1=TBD` — осознанная
граница валидности, а не пропущенный локальный запуск.

## 5. Amazon-C4 Automotive

### Retrieval

Добавлены loader/BM25/evaluator и candidate builder. Положительный product не
вставляется в top-100, reference review/rating не читаются как query.

| Method | Dev Recall@100 | Test Recall@100 | Test NDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.180556 | 0.157895 | 0.028509 |
| official BLaIR-base | **0.583333** | **0.592105** | **0.196720** |

BLaIR даёт в 3.75 раза больший test Recall@100, поэтому его top-100
зафиксирован для downstream controls.

### Images и CLIP

Добавлен resumable downloader с retries, MIME/decode validation, atomic JPEG
normalization и manifest. Он скачивает только union фиксированных candidates.

| Split | Unique candidates | Valid images | Coverage |
|---|---:|---:|---:|
| train | 18 780 | 18 770 | 99.9468% |
| dev | 5 615 | 5 610 | 99.9110% |
| test | 6 136 | 6 134 | 99.9674% |
| union | 21 558 | 21 545 | 99.9397% |

На неизменном BLaIR top-100:

| Method | Dev NDCG@10 | Test NDCG@10 | Test MRR@10 |
|---|---:|---:|---:|
| text-only | 0.166218 | 0.196720 | 0.155310 |
| image-only CLIP | 0.070657 | 0.079851 | 0.067544 |
| late fusion, dev-selected `text_weight=0.4` | **0.182471** | **0.226064** | **0.195734** |

Late fusion улучшила test NDCG@10 на `+14.92%` и MRR@10 на `+26.03%`;
Hit@10 и Recall@100 не изменились. Missing images не удаляют candidates.

## 6. Дополнительно приведена в порядок текущая документация

- UR4Rec corrected-v3 переведён `running → done` в registry/manifest/docs;
- сохранён result snapshot: base `0.214796`, UR4Rec `0.183334` NDCG@10;
- negative result отделён от paper-exact claims;
- START_HERE, TEAM_PROJECT, reproduction docs и result README обновлены;
- небольшие aggregated JSON сохранены в Git-области, а datasets, images,
  embeddings, model weights и per-candidate scores остаются игнорируемыми.

## 7. Проверки

```text
python -m unittest -q tests/test_correctness_guards.py tests/test_amazon_c4_pipeline.py
Ran 24 tests — OK
```

Проверки покрывают в том числе запрет leakage-полей в CatBoost и LLM prompt,
immutable candidate logic, detection вырожденных all-tie rankings, image
selection/fallback и ranking metrics.

## Основные артефакты

| Назначение | Путь |
|---|---|
| Литература | `docs/literature_ranked_review_2026-09-01.md` |
| Benchmark/ideas | `docs/benchmark_protocol_2026-09-01.md` |
| UR4Rec metrics | `results/current/metrics/ur4rec_ml1m_corrected_v3.json` |
| Avito CatBoost | `results/current/metrics/avito_local_catboost_diagnostic.json` |
| Avito L0/gate | `results/current/metrics/avito_local_llm_diagnostic.json` |
| Amazon retrieval | `results/current/metrics/amazon_c4_automotive_retrieval.json` |
| Amazon image coverage | `results/current/metrics/amazon_c4_automotive_image_coverage.json` |
| Amazon multimodal | `results/current/metrics/amazon_c4_automotive_multimodal.json` |
