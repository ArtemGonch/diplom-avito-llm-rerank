# Avito Case Browser: сборка, запуск и чтение интерфейса

Обновлено: **2026-09-10**. Это практическая инструкция к локальному
self-contained интерфейсу разбора выдач Avito. Полные числа и статус нужно
сверять с приватным локальным файлом
`results/current/metrics/avito_case_browser.json`; агрегированный публичный
итог находится в `results/current/manifest.json`.

Для агента тот же workflow оформлен как repo-skill `$avito-case-browser` в
`.agents/skills/avito-case-browser/`.

## Что показывает интерфейс

Одна страница объединяет для каждого `serp_x`:

- экспортированный список кандидатов и его block-aware порядок;
- локальные порядки CatBoost, Qwen L0 и отклонённого gate;
- доступную экспортированную историю контактов пользователя;
- graded benchmark relevance из `contacts_daily`;
- отдельный post-exposure marker `has_tc_events`;
- проверенные Qwen-сводки по featured объявлениям и историям.

Это offline diagnostic, а не production-интерфейс и не causal/off-policy
оценка. Qwen L0 не использует историю или позицию; история отображается для
исследования следующей итерации.

## Быстрый запуск готовой сборки

Из корня репозитория:

```bash
conda activate diplom_avito
test -s reports/avito_case_browser/index.html
python -m http.server 8765 --bind 127.0.0.1 \
  --directory reports/avito_case_browser
```

Открыть: `http://localhost:8765/`.

Процесс с сервером должен оставаться работающим. При удалённом VS Code порт
нужно пробросить на локальный `8765`; repo-skill `$avito-case-browser`
содержит проверенную конфигурацию автоматического проброса. По умолчанию
сервер слушает только loopback: интерфейс содержит псевдонимизированную
пользовательскую историю, поэтому делать публичный анонимный tunnel нельзя без
явного разрешения.

## Как открыть копию с Google Drive

Пакет загружен **2026-09-10** в папку владельца
`Диплом — Avito Case Browser — 2026-09-10`. Рабочий URL и Drive-ID намеренно
не хранятся в Git; актуальную ссылку следует брать из командного handoff.
Google Drive используется как хранилище файлов, а не как web-хостинг: HTML
нужно скачать, а не пытаться запускать внутри Drive Preview.

Самый простой вариант для просмотра:

1. Скачать `avito_case_browser.html`.
2. Открыть скачанный файл в современном браузере. Интернет и backend не нужны.
3. Если браузер или корпоративная политика блокирует локальный HTML, положить
   файл в отдельную папку как `index.html` и запустить из неё
   `python -m http.server 8765 --bind 127.0.0.1`.

Вариант с полным пакетом:

1. Скачать `avito_case_browser_share_bundle.zip` и распаковать его.
2. Открыть `index.html` из распакованной папки или запустить в ней тот же
   локальный HTTP-сервер.
3. Не переименовывать и не отделять будущую папку `assets/` от `index.html`,
   если после получения изображений она появится в новой сборке.

В корне проверенной Drive-папки находятся семь handoff-файлов и отдельный
каталог `private_avito_data` с 16 исходными/построчными артефактами, удалёнными
из GitHub:

| Файл | Для чего нужен |
|---|---|
| `avito_case_browser.html` | готовый автономный интерфейс для просмотра |
| `avito_case_browser_share_bundle.zip` | полный переносимый пакет с HTML, метриками и CSV |
| `metrics.json` | подробный audit текущей сборки |
| `item_request_featured.csv` | 264 приоритетных item ID для картинок/метаданных |
| `item_request_all_test.csv` | 4 614 item ID полного test split |
| `item_overlay_template.csv` | шаблон ответа с title/description/image |
| `ИНСТРУКЦИЯ_avito_case_browser.md` | эта инструкция для получателя пакета |

В `private_avito_data` раздельно хранятся `raw`, `benchmark`,
`case_browser_internal`, `knowledge` и `eda`. Каталог наследует действующее
право «все, у кого есть ссылка, могут редактировать» от родительской папки;
поэтому Drive-ссылку нельзя публиковать или коммитить.

Наличие файла в папке не доказывает, что ссылка доступна конкретному человеку:
перед отправкой нужно отдельно проверить права общей папки под нужным
аккаунтом. Отдельные дочерние файлы без необходимости не расшаривать — они
должны наследовать доступ папки. Для текущей папки владелец **2026-09-10 явно
подтвердил** режим «все, у кого есть ссылка, могут редактировать», чтобы Рома
мог не только скачать пакет, но и загрузить overlay. Ссылку поэтому следует
передавать только Роме: любой её обладатель сможет заменить или удалить файлы.
Агент не должен самостоятельно сужать эти права без нового запроса владельца.

### Что отправить Роме и что получить обратно

Для первого прохода отправить `item_request_featured.csv` и
`item_overlay_template.csv`; полный `item_request_all_test.csv` нужен только
после согласования покрытия. В ответ ожидается заполненный overlay с ключом
`item_id` и доступными `title`, `description_short`, `image_path` либо
`image_url`. Локальные картинки нужно вернуть вместе с CSV, сохранив указанные
относительные пути.

После получения данных скачать overlay и картинки в одну локальную папку,
пересобрать browser с `--item-overlay`, прогнать проверки из конца этой
инструкции и заменить на Drive существующие HTML, ZIP, metrics и инструкцию
**на месте**, без создания файлов-дубликатов. Текущий статус: пакет передан,
но изображения и дополнительные описания Ромы ещё не получены.

## Полная сборка

### Требуемые входы

| Вход | Путь по умолчанию | Назначение |
|---|---|---|
| Кандидаты | `items_with_attrs.parquet` | объявления, SERP, атрибуты и evaluation signals |
| История | `users_with_history.parquet` | экспортированные контакты по `user_id` |
| Scores | `data/avito/benchmark/local_llm_pointwise_eval_scores.parquet` | CatBoost, Qwen L0 и gate на одном split |
| Insight cache | `data/avito/case_browser/insights.jsonl` | структурированные ответы и provenance |
| Generation contract | `data/avito/case_browser/generation_contract.json` | pinned runtime/tokenizer/code contract |
| Шаблон | `apps/avito_case_browser/template.html` | HTML/CSS/JS интерфейса |

Оба parquet snapshot, scores, LLM cache, row-level metrics, изображения,
generated HTML и ZIP хранятся только на рабочей VM и в командной папке Drive;
в Git они не входят.

### Пересборка с уже валидным cache

```bash
python scripts/avito/build_case_browser.py
```

Команда проверяет точный join score-артефактов, relevance contract, cache и
generation provenance, затем обновляет:

- `reports/avito_case_browser/index.html` — автономный HTML;
- `reports/avito_case_browser/metrics.json` — метрики этой локальной сборки;
- `results/current/metrics/avito_case_browser.json` — приватный локальный snapshot с идентификаторами;
- `data/avito/case_browser/insight_requests.jsonl` — запросы к LLM;
- CSV manifest для featured и полного test split;
- `avito_case_browser_share_bundle.zip`, только если статус `done`.

Если cache отсутствует, частичный или не соответствует текущему контракту,
статус будет `needs_insights`, а старый share ZIP будет удалён, чтобы его нельзя
было ошибочно выдать за актуальный.

### Генерация недостающих инсайтов

Первый build формирует leakage-free requests. Затем на свободной GPU:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=0 \
  python scripts/avito/generate_case_insights.py \
  --device cuda:0 --batch-size 1

python scripts/avito/build_case_browser.py
```

Номер `2` — пример физической GPU; перед запуском его нужно заменить на реально
свободный. Внутри процесса выбранная карта видна как `cuda:0`. Для
канонического cache обязательны batch size 1, pinned revision и неизменённые
validation/provenance guards. По умолчанию генератор использует только локально
закэшированную модель.

### Overlay с изображениями и дополнительным текстом

Builder создаёт
`data/avito/case_browser/item_overlay_template.csv`. Поддерживаемые поля:

| Поле | Смысл |
|---|---|
| `item_id` | обязательный ключ объявления |
| `title` | замена/добавление заголовка |
| `description_short` | замена/добавление описания |
| `image_path` | локальный JPEG/PNG/WebP/GIF рядом с overlay |
| `image_url` | HTTP(S)-ссылка, если локального файла нет |

Сборка:

```bash
python scripts/avito/build_case_browser.py \
  --item-overlay /absolute/path/to/item_overlay.csv
```

Локальные изображения по умолчанию копируются в переносимый `assets/`.
Проверки путей, symlink, размера, формата, числа кадров и пикселей отключать не
следует.

## Как пользоваться экраном

### Поиск и фильтры слева

Поиск работает по `serp_x`, `user_id`, `item_id`, заголовку, brand/model и
доступным query proxies.

| Элемент | Что означает |
|---|---|
| `Все кейсы` | все SERP выбранного test split |
| `★ Отобранные` | восемь автоматически выбранных контрастных кейсов |
| `С историей` | `user_id` соединён хотя бы с одной строкой history |
| `С TC-event` | в выдаче есть хотя бы один `has_tc_events=1` |
| `Qwen лучше` | Qwen выше CatBoost по per-SERP graded NDCG@10 |
| `Targets ≠` | направление graded-сравнения расходится с порядком первого TC |
| `Δ Qwen − CB` | `NDCG@10(Qwen L0) − NDCG@10(CatBoost)` для одной выдачи |

Featured — не выбор удачных примеров вручную. Они детерминированно покрывают
максимальный выигрыш/провал Qwen, варианты с history, самую длинную history,
выигрыш/провал gate и расхождение target-слоёв.

### Верхние карточки метрик

Все значения относятся к выбранному SERP:

| Карточка | Интерпретация |
|---|---|
| `Local CatBoost · NDCG@10` | качество локального leakage-guarded baseline по graded target |
| `Qwen L0 · NDCG@10` | качество pointwise Qwen без history/position и дельта к CatBoost |
| `Gate · NDCG@10` | диагностический результат отклонённого blend |
| `Первый TC: Qwen / CB` | позиции первого observed TC; это не graded target |
| `История` | число экспортированных контактов, доступных для показа в UI |

`contacts_daily / max(contacts_daily внутри SERP)` задаёт graded relevance для
NDCG@10. Равные scores разрешаются по `item_id ascending`.

`has_tc_events` — другой слой: sparse событие после показа. Красный TC-бейдж и
позиция первого TC используются только для error analysis. Нельзя выбирать
между target-слоями постфактум по тому, какой выгоднее модели.

### История пользователя

Лента показывает экспортированные contact events и даты. `Период экспорта` не
доказывает, что событие было до текущего SERP: в выдаче нет `rank_time`.
History `brand` и `model_name` фактически являются sparse raw category-like
taxonomy, а не маркой и моделью автомобиля.

`Metadata coverage` — доля history rows, для которых overlay дал хотя бы
заголовок, описание или изображение. `Evidence-сводка истории` допускает лишь
наблюдаемые counts/dates/categories и осторожные гипотезы с низкой
уверенностью.

### Переключатели ранжирования

Они сортируют один и тот же набор кандидатов:

| Режим | Порядок |
|---|---|
| `Выдача` | экспортированный block-aware source order |
| `CatBoost` | локальный CatBoost ensemble; это не отсутствующий командный baseline Ромы |
| `Qwen L0` | Qwen2.5-7B pointwise score без history и position |
| `Gate` | dev-selected blend, отклонённый из-за результата ниже CatBoost на test |

`Top-5`, `Top-10` и `Все` меняют только число отображаемых карточек.

### Карточка объявления и «Подробнее»

- `#N` — rank в выбранном режиме;
- `benchmark rel.` — нормализованная graded relevance;
- `Qwen / CB rank` — быстрый просмотр двух порядков;
- красный `TC event` — отдельное наблюдаемое post-exposure событие;
- окно «Подробнее» — описание, ranks/scores всех методов и LLM-разбор.

Для item-insight обычный статус означает, что LLM output прошёл
schema/language/safety/exact-span guards. `guarded fallback` означает, что
исходный свободный ответ был отклонён, а видимые поля безопасно и
детерминированно восстановлены из evidence. `confidence low` не является
калиброванной вероятностью.

## Ограничения интерпретации

- Нельзя утверждать, что показанная history предшествует выбранной выдаче.
- Нельзя говорить, что Qwen L0 использовал историю: он её не использовал.
- Нельзя трактовать `has_tc_events` как тот же target, что `contacts_daily`.
- Нельзя выдавать local CatBoost за командный/production baseline.
- Нельзя делать выводы о покупке, доходе, профессии, возрасте, поле или иных
  чувствительных свойствах пользователя.
- В исходном snapshot есть `image_count`, но нет самих изображений или URL.

## Проверка после изменений

```bash
python -m unittest -q tests/test_avito_case_browser.py
python scripts/check_project_docs.py
```

Подробный исследовательский результат и текущие числа находятся в
[`task_2026-09-01_execution_report.md`](task_2026-09-01_execution_report.md).
