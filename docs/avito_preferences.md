# Avito Auto: данные, признаки и границы персонализации

Проверено по двум Parquet snapshot и текущим metric artifacts:
**2026-09-01**.

## Короткий вывод

Avito snapshot хорошо подходит для content/rank-time reranking по атрибутам
объявления, но пока недостаточен для доказанного personalized reranking:

- объявления подробно описаны и почти полностью заполнены;
- вместо текста запроса доступны только category/location proxies;
- история содержит 2 028 контактов 274 пользователей, но её экспортированные
  `brand`/`model_name` фактически являются значениями товарной таксономии и не
  совпадают со словарём автомобильных марок/моделей;
- времени показа SERP нет, поэтому нельзя доказать, что контакт был известен до
  целевой выдачи;
- текущие безопасные controls — local CatBoost и no-history Qwen L0;
- UR4Rec Avito остаётся legacy smoke, C-UR4Rec — proposed design.

## `items_with_attrs.parquet`: выдача и объявления

Размер snapshot: 44 736 candidate rows, 2 000 SERP и 41 592 уникальных
`item_id`. Одно объявление может встретиться в нескольких SERP.

### Автомобильные и текстовые признаки

| Группа | Поля | Непустая заполненность |
|---|---|---:|
| Текст | `title`, `description_short` | 100.00%, 97.85% |
| Идентичность авто | `brand`, `model_name` | 99.64%, 97.52% |
| Цена | `price`, `price_percentile`, `price_diff_from_median`, `imv` | 100% для `price`, 81.65% для `imv` |
| Эксплуатация | `year_text`, `mileage_km` | 99.98%, 96.83% |
| Спецификация | `gearbox_text`, `fuel_text`, `body_type`, `drive_text`, `doors_text` | 99.98% |
| Регистрация | `rf_reg_text` | 99.98% |
| Качество карточки | `image_count`, `title_len`, `desc_len`, `seconds_age` | 100% |
| Продавец | `seller_is_private`, `seller_rating` | 100%, но `seller_rating` константно 0 |

Словарь содержит 116 марок, 1 079 моделей, 12 типов кузова, 6 типов топлива,
5 вариантов коробки и 4 типа привода. В snapshot есть только `image_count`:
файлов изображений и URL для Avito нет.

### Контекст выдачи

| Поле | Смысл | Ограничение |
|---|---|---|
| `serp_x` | идентификатор выдачи | основной group id |
| `item_id` | кандидат | уникален внутри SERP |
| `block`, `block_pos` | блок и исходная позиция | `block_pos` допустим только для модели, которой исходный rank доступен online |
| `query_infm_logical_category` | category proxy | только 2 значения; это не полный запрос |
| `query_loc` | location id | 443 значения; нет человекочитаемого текста/фильтров |
| `user_id` | пользователь выдачи | 1 932 SERP имеют ненулевой id |
| `user_dist` | расстояние | константно 0 и неинформативно |

### Target и post-exposure поля

`contacts_daily` используется как graded train/evaluation target. Следующие
поля запрещены как rank-time features/prompts:

- `contacts_daily`;
- `clicks_daily`;
- `has_tc_events`;
- `has_x_events`;
- `serp_is_positive` (к тому же константно `True`).

Их использование в score создаёт target/post-exposure leakage.

## `users_with_history.parquet`: что реально есть в истории

История содержит 2 028 contact rows, 274 пользователя и 1 855 уникальных
исторических `item_id`; диапазон `contact_date` — 2026-02-24…2026-05-24.

| Экспортированное поле | Заполненность | Фактическая интерпретация в snapshot |
|---|---:|---|
| `contact_date` | 100% | дата контакта |
| `brand` | 49.46% | category-like level-1 values, например виды товаров/техники, не автомобильные марки |
| `model_name` | 13.51% | category-like level-2 values, не автомобильные модели |
| `year_raw` | 0% | сигнала нет |
| `mileage_km` | 0% | сигнала нет |
| `gearbox_text` | 0% | сигнала нет |
| `fuel_text` | 0% | сигнала нет |

Поля цены в history нет. Точное пересечение vocabulary равно нулю и для
`history.brand ↔ listing.brand`, и для
`history.model_name ↔ listing.model_name`. Исторические `item_id` также не
пересекаются с 41 592 кандидатами выдачи.

По `user_id` история покрывает 295 из 2 000 SERP; на local seed-42 split это
34/200 dev и 30/200 test SERP. Даже для этих SERP временная валидность не
доказана: у выдачи отсутствует `rank_time`, с которым можно сравнить
`contact_date`.

## Какие эксперименты можно интерпретировать

Все значения ниже — graded `NDCG@10` на local seed-42 split, если не указано
иное.

| Метод | Результат | Статус |
|---|---:|---|
| position order | 0.302670 | sanity baseline |
| local CatBoost ensemble | **0.653349** | текущий сильнейший local diagnostic; не CatBoost Ромы |
| Qwen2.5-7B L0, без history/position | 0.353667 | валидный no-history diagnostic |
| CatBoost + dev-selected L0 gate | 0.638136 | negative result; хуже CatBoost, gate отклонён |
| старый UR4Rec Avito smoke | 0.929924 | legacy; протокол несопоставим, rerun required |

### Почему прежний Exp3RT-style `0.3413` больше не benchmark

После отдельного schema/score audit выяснилось:

- history-поля были ошибочно названы автомобильными brand/model и price,
  хотя price отсутствует, а taxonomy values не совпадают с кандидатами;
- после устранения этой ложной семантики 200/200 test SERP получают одинаковые
  candidate scores;
- вычисленный NDCG определяется порядком разрешения ничьих, а не работой
  персонализированной модели.

Artifact `results/current/metrics/exp3rt_avito_full_leakage_free.json` сохраняется
для аудита с `valid_for_claims=false`; использовать его число в сравнительной
таблице или на защите нельзя. Более ранний legacy `0.9417` дополнительно
содержал прямой target/post-exposure leakage.

## Что можно строить сейчас

1. **Content-only/L0:** title, short description, brand/model, price, year,
   mileage, gearbox, fuel, body, drive, doors, registration, listing age и
   image count.
2. **LTR control:** те же rank-time признаки плюс исходная позиция, если она
   является доступным online feature; текущий control — CatBoostRanker.
3. **Cold-start analysis:** отдельно сравнивать SERP без history и 295 SERP с
   совпавшим user id, не выдавая последнее за temporal-safe personalization.

## Что необходимо для personalized C-UR4Rec

- `rank_time` для каждого SERP или гарантированный history cutoff;
- автомобильная история с корректными brand/model/price/attrs либо таблица
  соответствия history item metadata;
- зафиксированная policy для 85.25% SERP без совпавшей истории;
- CatBoost Ромы: test SERP ids, полный candidate score artifact, config/seeds и
  формула relevance;
- общий immutable candidate set для A1 и L0–L3.

После получения этих данных порядок ablation остаётся таким:
`no history → raw history → single profile → multi-aspect memory →
query/candidate-conditioned selection → calibrated cost gate`.

См. также [benchmark protocol](benchmark_protocol_2026-09-01.md),
[актуальную точку входа](START_HERE.md) и
[backlog улучшений](paper_improvements_backlog.md).
