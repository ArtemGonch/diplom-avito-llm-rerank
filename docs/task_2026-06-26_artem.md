# Задание 26.06: датасет, типология rerank-методов и новая статья

Дата аудита: 2026-08-25. Исполнитель: Артём. Актуализация после code review: corrected-v3 запущен от `knowledge`; финальных UR4Rec метрик ещё нет.

## Короткий вывод

1. Наиболее близкий публичный стенд для основной гипотезы диплома — **Amazon-C4 User Purchase History**, дополненный текстами и метаданными товаров из **Amazon Reviews'23**. Он одновременно даёт длинный товарный запрос, историю пользователя и target product. Для доменного среза следует использовать `Automotive` в истории/метаданных; это автотовары, а не объявления о продаже автомобилей, поэтому переносимость надо проверять отдельно.
2. **ESCI Shopping Queries** нужен как дополнительный неперсонализированный контроль качества query–item relevance: у него реальные запросы, выдачи до 40 товаров и graded ESCI labels, но нет пользовательской истории.
3. Новая статья для обзора — **MemRerank (2026)**. Она напрямую закрывает разрыв диплома: сжимает длинную историю в query-independent preference memory, затем применяет память только на rerank-стадии.
4. Главная линия собственного метода остаётся **C-UR4Rec**: candidate-aware memory retrieval + query context + confidence gate. После аудита это уже не просто улучшение UR4Rec: исправленная memory должна оставаться последовательностью токенов, иначе cross-attention вырождается.
5. Старый Avito-результат `0.9417` нельзя использовать: score видел target `contacts_daily` и связанный post-exposure сигнал `clicks_daily`, а NDCG делал все ненулевые labels бинарными. После устранения утечки graded NDCG@10 равен **0.3413** против **0.3126** у position baseline, дельта **+0.0288**.

## Что именно было в задании

Из документа с задачами:

- классифицировать датасеты по модальностям, типу запроса, важности персонализации и знаниям LLM;
- выбрать датасет, похожий на Avito Auto, и найти лучшие статьи на нём;
- сделать обзор идей ранжирования с типологией и эволюцией;
- добавить ещё одну статью;
- попробовать `/teach`.

Задача «дать ассистенту датасет» была назначена Роме, поэтому она не считается задачей Артёма. Локальные Avito parquet при этом полностью проверены.

## Ментальная модель диплома

Рабочая гипотеза диплома: LLM не заменяет retrieval/LTR на всём каталоге. Он создаёт или сжимает семантический сигнал offline, а дешёвая модель использует этот сигнал при rerank фиксированного top-K.

```text
catalogue + user history + query
             │
             ├─ cheap retrieval / LTR ──> fixed candidate set
             │                              │
offline LLM ─┴─> profiles / knowledge ─────┤
                                            ▼
                              candidate-aware reranker
                                            │
                                            ▼
                              ranking + cost + stability
```

Эта схема объединяет презентацию и код:

| Метод | Роль LLM | Что переносится в Avito | Главный риск |
|---|---|---|---|
| LettinGo | исследует профили и DPO-выбирает полезные downstream | адаптивный user profile | нет task-level preference pairs |
| UR4Rec | offline user/item knowledge | компактный trainable retriever | корректность memory, стоимость генерации |
| Exp3RT | извлекает preference из reviews, строит profiles, reasoning rating | pseudo-profile из истории контактов | в Avito нет reviews; SFT transfer слабый |
| LLM4Rerank | zero-shot multi-aspect list rerank по Goal | accuracy/diversity/fairness nodes | несколько дорогих online вызовов |
| Persona4Rec | offline item personas | быстрый user–persona score | нужны содержательные reviews |
| Think When Needed | router Think/Non-Think | вызывать reasoning только на сложных SERP | нужен честный difficulty target |
| MemRerank | offline preference memory из purchase history | наиболее близкая к нашей задаче память | preprint; evaluation пока узкая |

## Классификация датасетов

### Критерии

Для Avito-подобной задачи недостаточно совпадения домена. Нужны пять независимых измерений:

1. **Item representation:** изображения, свободный текст, структурированные атрибуты.
2. **Query:** фильтры, короткий текст, длинный/диалоговый intent.
3. **Personalization:** стабильный user id, временная история, длина и тип feedback.
4. **Ranking supervision:** impression-level candidate set и graded/implicit labels.
5. **LLM prior knowledge:** насколько модель знает предметы и их атрибуты без приватного каталога.

### Сравнение

| Датасет | Items | Query / candidate set | История | Labels | Знание LLM | Близость к Avito Auto |
|---|---|---|---|---|---|---|
| **Avito local** | title, short description, авто-attrs, `image_count`; самих изображений нет | только category + location; исходных фильтров/текста нет | 274 users, median 2 contacts | contacts/clicks после показа | высокое для марок/моделей, низкое для конкретных объявлений | целевой домен, но неполный query log |
| **Amazon-C4 + User Purchase History + Reviews'23** | длинный text/metadata; по parent ASIN можно присоединить images/details | 21,223 длинных semi-synthetic product queries; candidate pool строится retrieval | temporal purchases/reviews, within- и cross-category | один positive product; negatives из retrieval | высокое для массовых товаров | **лучшее структурное совпадение** |
| **ESCI** | title, description, bullets, brand, color | реальные query и до 40 candidates | нет | Exact/Substitute/Complement/Irrelevant | высокое | отличный query-relevance control, без personalization |
| Coveo SIGIR e-commerce | анонимизированные item fields и text/image vectors | реальные search sessions и impressions | session/browse history | click/non-click | низкое из-за анонимизации | хорошо для search bias, слабо для LLM knowledge |
| KuaiSAR | short-video side information | реальные search + recommendation actions | 25,877 users | positive/negative actions | домен не товарный | полезен для cross-scenario, но не Auto |
| Amazon Books | metadata + reviews | recommendation, не search query | длинная rating/review history | rating | высокое | годится для Exp3RT/UR4Rec, не для search |
| Steam | game metadata + reviews | recommendation | длинная history | rating/review | высокое | ещё дальше от Auto |
| MovieLens-1M | title/genre | recommendation | ratings | rating/implicit target | очень высокое | только отладочный benchmark |

Источники: [Amazon-C4](https://huggingface.co/datasets/McAuley-Lab/Amazon-C4), [purchase-history release](https://huggingface.co/datasets/zhiyuanpeng/amazon-c4-user-purchase-history), [Amazon Reviews'23](https://amazon-reviews-2023.github.io/), [ESCI](https://github.com/amazon-science/esci-data), [Coveo](https://github.com/coveooss/SIGIR-ecom-data-challenge), [KuaiSAR](https://kuaisar.github.io/).

### Почему выбран Amazon-C4 User Purchase History

Он единственный из рассмотренных публичных вариантов одновременно предоставляет:

- сложный товарный intent;
- `user_id` и историю до target purchase без temporal leakage;
- target parent ASIN, соединяемый с товарными metadata;
- split train/dev/test;
- небольшой готовый release (около 155 MB), пригодный для первого опыта.

Ограничения выбора:

- Amazon-C4 query синтезирован из положительного review, а не введён живым пользователем;
- это товары и автокомпоненты, не уникальные объявления подержанных машин;
- в release companion с o3-mini rewrite описан только для Electronics и Beauty; для других категорий надо использовать original Amazon-C4 query;
- один positive не равен логам полного SERP; candidate pool надо зафиксировать до rerank;
- картинки доступны через metadata URLs, но их использование потребует отдельного multimodal baseline.

Поэтому корректная формулировка в дипломе: **ближайший публичный proxy по структуре personalization + product search**, а не «аналог Avito Auto».

## Предлагаемый эксперимент на выбранном датасете

### Протокол

1. Взять опубликованные train/dev/test и не менять temporal cutoff.
2. Получить один и тот же top-100 для всех rerankers: BM25 и один dense retriever.
3. Считать Recall@100 отдельно: reranker не может исправить отсутствие positive в candidate set.
4. На фиксированном top-100 сравнить NDCG@10/MRR@10:
   - retrieval score без rerank;
   - неперсонализированный cross-encoder;
   - raw-history concat;
   - recency/category heuristic;
   - compact profile/memory;
   - query-conditioned C-UR4Rec.
5. Разбить test по длине истории, within/cross-category history и target category.
6. Для LLM-методов добавить latency, input/output tokens и повторяемость ranking при 3–5 запусках.

### Минимальные ablation

| ID | User memory | Query | Candidate | Gate | Зачем |
|---|---:|---:|---:|---:|---|
| B0 | — | ✓ | ✓ | — | неперсонализированный baseline |
| B1 | raw history | ✓ | ✓ | — | проверка, помогает ли история вообще |
| B2 | static profile | ✓ | ✓ | — | цена сжатия history |
| B3 | token memory | — | ✓ | — | чистый UR4Rec-style retrieval |
| B4 | token memory | ✓ | ✓ | — | вклад query conditioning |
| B5 | token memory | ✓ | ✓ | ✓ | полный C-UR4Rec |

## Типология и эволюция идей ранжирования

### Три ортогональные оси

**По стадии:** retrieval → rank/LTR → rerank → generation/explanation.

**По обучающему объекту:** pointwise item score → pairwise preference → listwise permutation/metric → policy/reward.

**По роли LLM:** data augmentation → semantic encoder → profile/memory builder → direct scorer → listwise agent → router/teacher.

Не следует смешивать эти оси. Например, Exp3RT — generative pointwise rating на rerank-стадии, а LLM4Rerank — listwise prompt-agent на той же стадии.

### Эволюция

| Этап | Главная идея | Представители | Что осталось нерешённым |
|---|---|---|---|
| 1. Classic LTR | hand-crafted features и point/pair/list losses | BM25, LambdaMART | слабая семантика и cold-start |
| 2. Neural list/sequential | моделировать зависимости candidates/history | GRU4Rec, DLCM, PRM, SetRank, SASRec | ID-only знания и перенос домена |
| 3. Pretrained semantics | кодировать query/item text | BERT cross-encoder, BLaIR | длинная user history и стоимость |
| 4. Direct LLM rerank | prompt по user + top-K | RankGPT-подобные методы | latency, order bias, hallucinations |
| 5. Offline knowledge/profile | вынести LLM из online path | KAR, UR4Rec, Exp3RT, LettinGo | как сжать сигнал без потери utility |
| 6. Multi-objective orchestration | управлять accuracy/diversity/fairness текстовым Goal | LLM4Rerank | нестабильность и число вызовов |
| 7. Offline index/memory | материализовать personas/preferences заранее | Persona4Rec, MemRerank | update policy и query mismatch |
| 8. Adaptive compute/RL | тратить reasoning только там, где он полезен | Think When Needed, Rec-R1, MemRerank-RL | надёжная награда и calibration |

Практический вывод: собственный вклад разумно формулировать не как «ещё один LLM-ranker», а как **query-conditioned preference memory для дешёвого rerank с fallback на base model**.

## Новая статья: MemRerank

[MemRerank: Preference Memory for Personalized Product Reranking](https://arxiv.org/abs/2603.29247) — preprint 2026 года.

### Идея

Raw purchase history длинна, шумна и часто не соответствует текущему query. MemRerank сначала строит компактную query-independent memory, затем проверяет её utility через downstream 1-in-5 LLM reranking. Memory extractor обучается RL-наградой от reranker, а не только совпадением с эталонным текстовым summary.

### Почему статья важнее ещё одного direct reranker

| Вопрос | UR4Rec | Exp3RT | MemRerank |
|---|---|---|---|
| Исходный сигнал | generated user/item knowledge | reviews | temporal purchase history |
| Сжатие | trainable cross-attention proxies | SFT profiles | compact explicit memory |
| Query dependence memory | нет | нет | memory нет, reranker видит query |
| Обучающий feedback | contrastive + matching + RecLoss | teacher distillation | downstream rerank reward |
| Близость к shopping search | средняя | низкая | высокая |

Авторы сообщают до **+10.61 абсолютных пунктов** в 1-in-5 accuracy относительно memory baselines. В основной end-to-end таблице выигрыш по MRR@10 заметно меньше, поэтому статью нельзя пересказывать как универсальный большой рост. Дополнительные ограничения: preprint; основная оценка сосредоточена на Electronics/Beauty; candidate recall ограничивает потолок; в data/pipeline используются proprietary teacher/rewrite models.

### Что взять в диплом

- query-independent memory можно кэшировать и обновлять независимо от запросов;
- качество profile надо учить/оценивать downstream-метрикой, как также делает LettinGo;
- memory необходимо тестировать против raw-history и no-memory, а не только против слабого шаблона;
- один и тот же candidate set обязателен для честного сравнения.

Связанные работы на том же стенде: [BLaIR / Amazon-C4](https://arxiv.org/abs/2403.03952), [Rec-R1](https://arxiv.org/abs/2503.24289).

## Аудит локального Avito датасета

### Факты

| Проверка | Результат |
|---|---:|
| item rows / SERP | 44,736 / 2,000 |
| размер SERP mean / max | 22.368 / 435 |
| history rows / users | 2,028 / 274 |
| history length median / mean / max | 2 / 7.40 / 178 |
| пустой brand / model в history | 50.5% / 86.5% |
| item-id overlap items↔history | 0 |
| user-id overlap items↔history | 274 |
| SERP с 0 / 1 / >1 `user_id` | 68 / 1,932 / 0 |
| строки с `contacts_daily > 0` | 91.34% |
| строки с `clicks_daily > 0` | 99.45% |
| corr(clicks, contacts) | 0.6204 |
| `serp_is_positive` | True для 100% строк |

`user_id` постоянен внутри SERP и все 274 history users пересекаются с items. Это сильное свидетельство, что поле относится к пользователю поиска, а не к продавцу. Без исходного schema contract это остаётся обоснованной интерпретацией, не доказанным названием поля.

Файл содержит `image_count`, но не сами изображения. Query представлен category и location id; значений фильтров и текстового запроса нет. Поэтому по этому срезу нельзя честно проверить assistant-like длинные запросы или полноценный query-conditioned Auto rerank.

### Исправленный Avito-прогон

Артефакт: `results/current/metrics/exp3rt_avito_full_leakage_free.json`.

| Model | NDCG@1 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|
| leakage-free pseudo-profile heuristic | 0.1399 | 0.2394 | **0.3413** |
| position baseline | 0.1217 | 0.2092 | **0.3126** |
| delta | +0.0182 | +0.0302 | **+0.0288** |

Ограничения: random SERP split, маленькая test-выборка 200 SERP, сильный position/exposure bias, почти все items имеют положительный implicit signal, нет propensity correction. Это offline correlation, не доказательство роста contacts online.

## Сопоставление прошлых задач с кодом

| Задача | Артефакт | Фактический статус после аудита |
|---|---|---|
| Общий формат проекта | `README.md`, `docs/TEAM_PROJECT.md`, `experiments/registry.yaml` | сделано; часть статусов в handoff устарела |
| Два прогона | UR4Rec ML-1M, Exp3RT Amazon checkpoints/results | Exp3RT test готов; UR4Rec corrected-v3 запущен, прежние веса legacy |
| Выбрать метод preferences для Auto | `docs/avito_preferences.md`, Avito heuristic | направление верно, старый результат содержал leakage |
| +1 алгоритм | `docs/llm4rerank_vs_ur4rec_exp3rt.md`, C-UR4Rec design | концепт сделан, LLM4Rerank не воспроизведён |
| Calibration | `docs/paper_improvements_backlog.md` | только backlog, реализации нет |
| Ограничение числа LLM requests | offline generation, cached knowledge | частично; строгой cost table нет |
| Ограниченный контекст | profiles/proxies | token memory и proxies исправлены; эффект проверит corrected-v3 |
| References | docs и presentation | сделано, здесь добавлены primary links |
| Idempotency | backlog | тестов повторяемости пока нет |
| Классификация датасетов | этот документ | сделано |
| Типология/evolution | этот документ | сделано |
| +1 статья | MemRerank section | сделано |
| `/teach` | `teach/` | установлен и создан первый lesson/reference |

## Аудит кода и воспроизводимости

### Исправлено

1. `.gitignore` исключал не только большие `data/`/`models/`, но и `src/data/`, `src/models/`, `src/ur4rec/data/`. Основной код теперь явно unignore.
2. HF batch generation при left padding срезал output по неполному `attention_mask.sum`, из-за чего в knowledge попадали хвосты prompt. Исправлен срез по общей padded input width. Простая prefix-проверка нашла остаточный `assistant`/prompt в 1,391 из 3,883 item и 504 из 6,037 user записей ML-1M full, а также в 60/300 item и 25/600 user записей Avito smoke. Существующие Qwen knowledge cache необходимо сгенерировать заново.
3. UR4Rec Eq. 5 теперь передаёт user/item knowledge как несколько memory tokens. Старый flatten → one token делал cross-attention математически вырожденным.
4. Figure 3(c) mask раньше изменял пустой slice `mask[i:p, i:p]`; теперь distinct item tokens не видят друг друга, сохраняя self-attention.
5. Self-attention projection weights UR4Rec теперь действительно инициализируются из BERT вместе с FFN/norm.
6. Общий NDCG сохраняет graded labels; на бинарных датасетах поведение не изменилось.
7. Avito heuristic больше не читает `contacts_daily` или `clicks_daily`; добавлены correctness tests.
8. Knowledge cache v2 проверяет generator/model/tokens/shards и флаг полноты, поддерживает resumable shard checkpoints и явный merge.
9. ML-1M corrected-v3 использует temporal-per-user targets; user embeddings validation/test больше не остаются необученными из-за user-wise split.
10. Static user profile строится по train history и не включает validation/test targets.
11. Frozen BERT работает в `eval()`, повторяющиеся тексты кодируются батчами и кэшируются; для offline smoke добавлен deterministic hashing encoder.
12. Contrastive negatives исключают известные positives; конфигурация retriever едина между pretrain/joint/eval.

### Открытые блокеры перед сильным экспериментальным claim

1. UR4Rec corrected-v3 запущен 2026-08-25 от `knowledge`; до успешных `merge → backbone → pretrain → joint → eval` и появления test JSON все прежние weights/metrics остаются legacy.
2. Corrected-v3 исправляет split через temporal-per-user targets, но использует random top-100. Для paper-like claim нужен отдельный temporal MF/BPR candidate protocol; названия прежних `paper_exact` configs сильнее фактической гарантии.
3. Avito UR4Rec создаёт internal user по SERP и не использует `users_with_history` как реальную последовательность поведения.
4. Full Avito query отсутствует, поэтому C-UR4Rec нельзя корректно оценить на текущем extract.
5. Exp3RT generic `--stage all` не гарантирует chaining всех adapters так, как shell paper-full pipeline; merge после train должен явно брать best checkpoint.
6. `AGENT_HANDOFF.md` содержит полезную хронологию, но остаётся архивом; текущий вход — `START_HERE.md`, registry и узкие reproduction-документы.

## Что можно утверждать на защите уже сейчас

- Архитектурная позиция: LLM полезнее как offline profile/memory/knowledge builder и top-K reranker, чем как full-catalog ranker.
- Exp3RT reproduction на Amazon имеет отдельные rating-only и 4-stage artifacts; paper-full expected RMSE `0.5624`, MAE `0.3496` на 11,743 examples.
- Avito показывает наличие user-history join по `user_id`, но история короткая и атрибуты сильно разрежены.
- Leakage-free pseudo-profile heuristic улучшает graded NDCG@10 относительно position baseline на `0.0288`; это предварительный offline результат с явными ограничениями.
- Amazon-C4 User Purchase History — выбранный внешний proxy для следующего персонализированного product-search эксперимента.

Нельзя утверждать до завершения corrected-v3: «UR4Rec воспроизведён paper-exact» или «UR4Rec превосходит backbone». Также нельзя утверждать: «Avito NDCG@10 = 0.9417», «C-UR4Rec доказан экспериментально».

## Основные источники

- [UR4Rec, COLING 2025](https://aclanthology.org/2025.coling-main.45/)
- [Exp3RT](https://arxiv.org/abs/2408.06276)
- [LLM4Rerank](https://arxiv.org/abs/2406.12433)
- [LettinGo](https://arxiv.org/abs/2506.18309)
- [Persona4Rec](https://arxiv.org/abs/2602.21756)
- [Think When Needed](https://arxiv.org/abs/2601.18146)
- [Beyond Utility](https://arxiv.org/abs/2411.00331)
- [MemRerank](https://arxiv.org/abs/2603.29247)
- [Amazon-C4 / BLaIR](https://arxiv.org/abs/2403.03952)
- [ESCI](https://github.com/amazon-science/esci-data)
