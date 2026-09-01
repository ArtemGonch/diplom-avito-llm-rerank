# Задание Артёма на неделю 31.08–06.09.2026

Источник: блок `2026-08-25` в
[командном Google-документе](https://docs.google.com/document/d/1RIM-1r5kKr_rC7uFQa-m6iaht65K7R0-plInyvwA0XI/edit),
дополненный решением от 2026-08-25 скачивать изображения Amazon только после
фиксации candidate set.

Живой export Google-документа повторно проверен **2026-09-01**: последний
датированный блок — `2026-08-25`, и все его пункты включены ниже; более нового
задания в документе нет.

## Статус выполнения на 2026-09-01

| Пункт | Статус | Артефакт / результат |
|---|---|---|
| 1. Ранжированный обзор | **готово** | [единая шкала, 15 работ и top-5](literature_ranked_review_2026-09-01.md) |
| 2. Финальные benchmarks | **готово** | [immutable candidates, splits, metrics и artifact contract](benchmark_protocol_2026-09-01.md) |
| 3. Три собственные идеи | **готово** | hypothesis → implementation → ablation → success criterion в benchmark protocol |
| 4. Avito LLM vs CatBoost | **частично, external block** | local CatBoost `0.653349`, L0 `0.353667`, gate `0.638136`; A1 Ромы и leakage-free L1/L2/personalized L3 невозможны без перечисленных artifacts/cutoff |
| 5. Amazon images | **готово** | 21 545/21 558 images; text `0.196720` → late fusion `0.226064` NDCG@10 |

Подробный отчёт с командами, проверками и выводами:
[task_2026-08-25_execution_report.md](task_2026-08-25_execution_report.md).

## Коротко: что должно быть готово к следующей встрече

1. Ранжированный обзор релевантных научных и индустриальных работ.
2. Конечный список benchmark-методов и единый протокол сравнения.
3. Три собственные идеи для LLM-ранжирования с проверяемыми ablation.
4. Несколько leakage-free вариантов LLM-reranking на Avito, сравненных с одним
   и тем же CatBoost по `NDCG@10` на таргете контактов.
5. Воспроизводимый план/пайплайн изображений Amazon-C4 Automotive и
   text-only/image-only/text+image control на фиксированных candidates.

## 1. Ранжированный обзор работ

Сделать не просто перечень статей, а таблицу, в которой каждая работа получает
оценку по общей шкале:

| Критерий | Баллы | Что проверяем |
|---|---:|---|
| Близость к product search + personalization | 0–3 | query, user history, candidate rerank |
| Новизна для нашего метода | 0–2 | memory, query conditioning, gate/adaptive compute |
| Надёжность эксперимента | 0–2 | fixed candidates, baselines, ablation, leakage control |
| Воспроизводимость | 0–2 | публичные data/code/checkpoints |
| Практичность | 0–1 | offline/online cost, latency, кэширование |

Итог — сумма из 10 и короткий вывод «что берём / почему не берём». В core list
включить UR4Rec, Exp3RT, LLM4Rerank, BLaIR/Amazon-C4, MemRerank, Rec-R1,
LettinGo, Persona4Rec, Think When Needed и
[Netflix GenRec](https://netflixtechblog.com/genrec-towards-llm-native-recommendation-at-netflix-f20be6f643e3).
GenRec помечать как индустриальный материал, а не peer-reviewed baseline.
Отдельно найти и оценить Amazon-работы по product search/recommendation.

**Артефакт:** ранжированная таблица, top-5 для текста диплома и по одному
конкретному заимствованию/ограничению для каждой работы top-5.

## 2. Конечный список benchmarks

### Avito: главный целевой протокол

| ID | Метод | Статус / роль |
|---|---|---|
| A0 | исходный position order | sanity baseline; локально `0.3126`, но его надо пересчитать на CatBoost split |
| A1 | командный CatBoost ranker | главный baseline, который требуется превзойти; метрика и artifacts ожидаются от Ромы |
| A2 | pseudo-profile heuristic | пока отсутствует: прежний `0.3413` после schema/score audit признан all-tie artifact |
| L0 | no-history LLM pointwise | query proxy + candidate attrs, вклад LLM без personalization |
| L1 | offline LLM profile + pointwise rerank | вклад сжатой user history |
| L2 | offline LLM profile + listwise top-K rerank | вклад listwise сравнения candidates |
| L3 | confidence-gated blend LLM + CatBoost | предлагаемый cost-aware вариант с fallback |

Для `A1`, `L0–L3` обязателен один и тот же CatBoost candidate set и test split.
`contacts_daily` используется только как graded relevance при evaluation;
`contacts_daily`, `clicks_daily` и любые post-exposure производные запрещены в
rank-time features/prompts. Основная метрика — `NDCG@10`; дополнительно
`NDCG@5`, `MAP@10`, latency, tokens/1k SERP и стабильность трёх повторов.

### Amazon-C4 Automotive: внешний переносимый протокол

Зафиксированный состав находится в
[задании 26.06](task_2026-06-26_artem.md#зафиксированные-baselines-и-ablation):
BM25 и BLaIR-base для отдельных top-100; затем retrieval score, cross-encoder,
recency/category, raw history, static profile, corrected UR4Rec и C-UR4Rec
ablation. Retrieval отчитывает `Recall@100`, rerank — `NDCG@10/MRR@10`.

## 3. Три собственные идеи

1. **Multi-aspect preference memory.** Offline LLM строит не один общий профиль,
   а аспекты `brand/model`, `price`, `body/specs`, `geo` и `recency`. Ablation:
   один profile против K аспектов.
2. **Query–candidate conditioned selection.** Для каждого кандидата выбираются
   только релевантные текущему query proxy аспекты истории. Ablation: static
   profile против query-only и query+candidate conditioning.
3. **Calibrated confidence/cost gate.** LLM-сигнал применяется только при
   достаточной уверенности и потенциальной перестановке top-K; иначе остаётся
   CatBoost. Ablation: always-LLM против gate и CatBoost-only; вместе с качеством
   считать долю LLM-вызовов.

Это три проверяемые части общей линии C-UR4Rec, а не три несвязанных названия.

## 4. Avito: эксперименты с целью превзойти CatBoost

Порядок работы:

- получить от Ромы CatBoost `NDCG@10`, config/seed, test SERP ids, candidate
  scores и точное определение relevance;
- воспроизвести CatBoost metric локально до запуска LLM-вариантов;
- зафиксировать один top-K и один prompt/model/generation config для `L0–L3`;
- построить leakage-free offline profiles только из доступной до target истории;
- прогнать pointwise, listwise и gated варианты;
- выбрать prompt, blend и threshold только по validation, test открыть один раз;
- сохранить metrics JSON, per-SERP scores, cost/stability table и ошибки по
  history length / SERP size / brand-model sparsity.

Цель формулируется как `NDCG@10(L3) > NDCG@10(CatBoost)` на одном протоколе.
Если CatBoost не превзойдён, результат всё равно считается содержательным при
наличии честной ablation и анализа стоимости/ошибок; число нельзя улучшать
сменой split или candidate set.

## 5. Amazon-C4 Automotive: изображения

- материализовать BM25/BLaIR top-100 до скачивания изображений;
- соединить candidate `parent_asin` с `Amazon Reviews'23 Automotive.images`;
- выбрать один `variant=MAIN` URL: `large`, fallback `hi_res`, затем `thumb`;
- сделать resumable downloader с retries, decode/content-type validation и
  manifest `ok/missing/failed`;
- не скачивать весь каталог: только уникальные candidates;
- посчитать image coverage отдельно для train/dev/test;
- закэшировать `openai/clip-vit-base-patch32` embeddings;
- сравнить `MM0 text-only`, `MM1 image-only`, `MM2 text+image late fusion` на
  неизменяемом top-100; fusion weight выбирать только по dev.

## Зависимость от Ромы

По Google-документу Рома должен сообщить текущий CatBoost `NDCG@10`. Для
воспроизводимого сравнения дополнительно нужны test SERP ids, candidate scores,
model/config revision и формула relevance. Пока этих artifacts нет в локальном
репозитории, значение CatBoost отмечается `TBD`, а claim «побили CatBoost»
невозможен.

## Definition of done

- обзор имеет единую шкалу и обоснованный top-5;
- benchmark-таблица фиксирует data split, candidates, features, metric и cost;
- три идеи описаны как hypothesis → implementation → ablation;
- CatBoost и LLM-варианты пересчитаны одним evaluator на одном test set;
- результаты сохранены как JSON + компактная сводная таблица;
- image downloader идемпотентен, missing images не меняют candidate set;
- все claims отделяют доказанный результат от design/legacy.
