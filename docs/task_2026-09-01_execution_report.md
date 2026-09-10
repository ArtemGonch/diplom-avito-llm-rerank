# Отчёт по заданию 2026-09-01

Выполнено: **2026-09-08**, статус передачи обновлён **2026-09-10**. Код,
интерфейс, выбор кейсов, LLM-анализ, локальный share bundle и загрузка пакета
на Google Drive готовы. Ожидаются только исходные изображения и дополнительные
описания от Ромы.

## 1. Что построено

Создан воспроизводимый offline case browser для test split Avito:

- слева — поиск и фильтры по 200 SERP, включая 8 featured cases;
- история контактов пользователя и evidence-bounded аналитическая сводка;
- карточки 4 458 кандидатов с title, description и структурными атрибутами;
- переключение block-aware исходной выдачи, local CatBoost, Qwen L0 и
  отклонённого gated blend;
- позиции, scores, graded benchmark relevance и отдельный бейдж фактического
  `has_tc_events`;
- подробная карточка объявления и структурированный Qwen-анализ;
- все данные встроены в один HTML, внешняя БД или backend не нужны.

Основные файлы:

| Артефакт | Назначение | Git |
|---|---|---|
| `apps/avito_case_browser/template.html` | UI-шаблон | да |
| `src/avito_case_browser/core.py` | join, метрики, кейсы, prompts, rendering | да |
| `src/avito_case_browser/insights.py` | единый evidence/provenance contract | да |
| `scripts/avito/build_case_browser.py` | воспроизводимая сборка | да |
| `scripts/avito/generate_case_insights.py` | pinned Qwen generation и guards | да |
| `results/current/metrics/avito_case_browser.json` | приватный локальный итог с идентификаторами | нет, VM/Drive |
| `reports/avito_case_browser/index.html` | локальный self-contained browser | нет, generated |
| `data/avito/case_browser/avito_case_browser_share_bundle.zip` | пакет для Drive | нет, generated |

## 2. Контракт и аудит данных

Score artifacts CatBoost и Qwen соединены по точному ключу
`(split, serp_x, item_id)`: для test совпали все 4 458 строк, потерь и лишних
ключей нет. Relevance в обоих artifacts совпадает с
`contacts_daily / max(contacts_daily внутри SERP)` с максимальным численным
расхождением `0.0`: все три evaluator теперь нормализуют relevance в
`float64`.

Важно разделены два сигнала:

- `contacts_daily` — dense graded relevance фиксированного benchmark;
- `has_tc_events` — sparse post-exposure TC-marker конкретного показа; он
  отображается только для диагностики и никогда не попадает в признаки или
  LLM prompt.

Корреляция этих полей на уровне test-строк лишь `0.0963`, поэтому смешивать их
в одной метрике нельзя. История найдена у 30/200 SERP: 223 контакта. В выдаче
нет `rank_time`, следовательно нельзя утверждать, что вся экспортированная
история была известна до показа.

## 3. Сводные результаты

Основная метрика — graded NDCG@10 с явным переносимым tie-break
`item_id ascending`:

| Порядок | NDCG@10 |
|---|---:|
| Исходный block-aware порядок | 0.428257 |
| Local CatBoost ensemble | **0.657788** |
| Qwen2.5-7B L0, без history/position | 0.353148 |
| Dev-selected gated blend | 0.644844 |

CatBoost diagnostic, Qwen diagnostic и browser используют один явный порядок
`score descending → item_id ascending` и один `float64` relevance-контракт;
в current artifacts все три NDCG@10 совпадают при повторном пересчёте точно
(`max abs difference = 0.0`). Для
CatBoost exact evaluation воспроизводится по зафиксированным моделям и score
artifact: combined split SHA-256 —
`4669f4da20b6005bc79e4e8b929b8010b4d624d46590d1ab8318362e4e3e583b`,
score parquet SHA-256 —
`2723f3ad8e44f9644084aeeb83ae8e66b98d7cd5610ebfa28c31c2af9c9c235b`.
Три `.cbm` имеют отдельные hashes/metadata; новый GPU fit не заявляется
бит-в-бит детерминированным. Режим `--reuse-existing-models` до вычислений
требует полный набор regular `.cbm` + sidecar для всех seeds и падает при
любом пропуске. Новый fit не может затронуть существующие model/score/metrics
без явного `--force-retrain`; штатный путь — три новых output path.

Прежнее число position baseline `0.302670` признано невалидным: формула
`-block_pos` поднимала 967 supplemental candidates с sentinel `-1` выше
основного блока с позициями `0+`. В интерфейсе и текущем сравнении этот метод
удалён; для исходной выдачи используется явный порядок `block priority → valid
block_pos → item_id`. В CatBoost JSON оставлена только причина исключения
этого метода, но само число больше не публикуется как текущая метрика.

На 142 SERP хотя бы с одним `has_tc_events=1` получен отдельный описательный
диагностический срез:

| Порядок | Hit@10 | MRR@10 | Binary NDCG@10 |
|---|---:|---:|---:|
| Исходная выдача | **0.781690** | **0.351260** | **0.433401** |
| Local CatBoost | 0.711268 | 0.235904 | 0.325679 |
| Qwen L0 | 0.570423 | 0.190065 | 0.261393 |
| Gated blend | 0.697183 | 0.229429 | 0.320657 |

Это только error analysis post-exposure marker, а не замена фиксированного
graded benchmark и не causal/off-policy оценка.

## 4. Найденные кейсы

Выбрано восемь непересекающихся типов кейсов:

| Тип | Наблюдение |
|---|---|
| Максимальный выигрыш Qwen | `+0.914` NDCG@10 против CatBoost; TC поднят с позиции 14 на 1 |
| Максимальный провал Qwen | `−0.964`; при этом TC у Qwen на 6, а у CatBoost на 15 — цели расходятся |
| Лучший Qwen при history | `+0.226`, история из 1 контакта; текущий L0 её фактически не использует |
| Худший Qwen при history | `−0.945`, история из 8 контактов; TC: CatBoost 1, Qwen 15 |
| Самая длинная history | 76 контактов, но taxonomy sparse и temporal cutoff неизвестен |
| Максимальный выигрыш gate | `+0.304` локально |
| Максимальный вред gate | `−0.340` локально |
| Несогласие targets | Qwen сильно хуже по graded target (`−0.933`), но выше CatBoost по первому TC (4 против 6) |

Машинно проверяемый обезличенный срез из приватного current JSON (этот блок
валидирует `scripts/check_project_docs.py`; исходные `serp_x` не публикуются):

<!-- BEGIN AVITO FEATURED CASES -->
| reason_id | history | delta_qwen | delta_gate | CatBoost TC rank | Qwen TC rank |
|---|---:|---:|---:|---:|---:|
| target_disagreement | 1 | -0.933363 | -0.291949 | 6 | 4 |
| largest_gate_failure | 0 | -0.785932 | -0.340220 | 23 | 23 |
| largest_qwen_failure | 0 | -0.964384 | +0.000000 | 15 | 6 |
| largest_qwen_gain | 0 | +0.913913 | +0.000000 | 14 | 1 |
| history_qwen_failure | 8 | -0.945093 | +0.000000 | 1 | 15 |
| largest_gate_gain | 0 | -0.377818 | +0.304237 | — | — |
| history_qwen_gain | 1 | +0.226145 | +0.000000 | 6 | 12 |
| longest_history | 76 | -0.080928 | +0.000000 | — | — |
<!-- END AVITO FEATURED CASES -->

Вывод: Qwen иногда резко улучшает отдельные выдачи, но L0 в среднем намного
хуже CatBoost. Простая gate-эвристика нестабильна. Различия между
`contacts_daily` и единичным фактическим TC-event нужно изучать отдельно, не
выбирая удобную метрику постфактум.

## 5. Что получилось узнать от LLM

Для featured cases создано 72 запроса: 68 уникальных объявлений и 4 истории
пользователя. Использован `Qwen/Qwen2.5-7B-Instruct` на pinned revision
`a09a35458c702b33eeacc393d103063234e8bc28`.

- Для объявления LLM выбирает только потенциальные дословные фрагменты
  продавца. В UI попадают лишь фрагменты, которые автоматическая проверка
  находит в `seller_description` без перефразирования.
- Про пользователя наблюдения ограничены точным числом контактов, периодом и
  raw taxonomy counts; гипотезы детерминированно строятся только из реально
  встречавшихся raw categories и содержат модальность «может».
- Запрещены выводы о покупке, просмотре/клике, профессии, доходе, возрасте,
  поле и других чувствительных свойствах.
- Post-exposure labels не входят в prompt. Все показываемые поля item insight,
  кроме exact-span `seller_claims`, и вообще все поля user insight полностью
  пересобираются детерминированно из evidence; `confidence` фиксируется как
  `low`, а неподтверждаемая `suitable_for` остаётся пустой.

После primary pass и одного retry 44/72 (`61.1%`) ответов прошли строгую
schema/language/safety/evidence проверку. Для 28/72 (`38.9%`) применён
детерминированный guarded fallback, сохраняющий только подтверждённые поля.
Это самостоятельный результат: свободный LLM-текст нельзя безопасно показывать
как факт без структурной валидации и fallback. Builder повторно проверяет
`request_id`, kind/entity, prompt hash/version, model/revision, canonical batch
size 1, SHA-256 полного ordered request manifest, physical/logical GPU mapping
и driver, версии Torch/CUDA/Transformers и deterministic flags. Contract v5
дополнительно фиксирует SHA-256 `tokenizer.json`, `tokenizer_config.json`,
`vocab.json`, `merges.txt` и chat template. При validation pinned tokenizer
загружается локально, для каждой primary/retry попытки заново строятся template,
`input_ids` и `attention_mask`, после чего реальные hashes/shapes сравниваются
с cache. Также проверяются batch membership, token budget, raw LLM text и
итоговый JSON; primary никогда не затирается retry. Текущий контракт —
browser v5, prompt v3, validation v9, signature
`2028fe94f1117523fd3fa11243bd133b84ac1324bd108de74258a3ede2c78118`.
Cache JSONL SHA-256 — `ed0fde8d3b661fab2cc0229daded5a7967503201dff38200358cd8f80ceb0de9`,
standalone contract SHA-256 — `571771ef5e5e9f90aac96ebdb74d00718a43ac591857e0493f7797022041c5db`.
Не соответствующий контракту, частичный или устаревший cache в HTML не встраивается; сборка
получает `needs_insights` и не создаёт финальный ZIP, пока валидны не все 72
запроса.

Практический вывод для user profile отрицательный, но полезный: свободный
LLM-пересказ истории нельзя считать подтверждённым профилем. В текущем UI все
user-поля, включая ограничения и confidence, строятся проверяемо из
counts/dates/raw categories; исходный LLM-текст остаётся только в audit cache.

## 6. Передача айтемов и изображения

Подготовлены:

- `item_request_featured.csv` — 264 уникальных candidate/history item IDs;
- `item_request_all_test.csv` — 4 614 уникальных item IDs полного test split;
- `item_overlay_template.csv` — контракт `item_id`, title, description,
  `image_path`/`image_url` для ответа Ромы;
- `avito_case_browser_share_bundle.zip` — браузер, метрики и приоритетный
  manifest, размер около 1.6 MB.

В текущем Avito snapshot есть только `image_count`; файлов и URL нет. После
получения overlay интерфейс пересобирается командой с `--item-overlay`;
локальные файлы атомарно копируются в переносимый `assets/` (отключение —
`--no-copy-overlay-images`). Локальный путь обязан оставаться внутри каталога
overlay даже после разрешения symlink/`..`; symlink destination/parent
отклоняется. Файл ограничен 20 MiB, одним кадром не более 40 MP, всей
анимацией — не более 32 кадров и 40 MP суммарно; все кадры должны полностью
декодироваться как JPEG/PNG/WebP/GIF. ZIP повторно проверяет только assets, на
которые ссылается текущая сборка, и не подхватывает устаревшие файлы.

**2026-09-10 передача через Google Drive выполнена.** В папку владельца
`Диплом — Avito Case Browser — 2026-09-10` загружены и затем повторно
проверены семь файлов: standalone HTML, share ZIP, `metrics.json`, featured и
all-test manifests, overlay template и инструкция. Drive-ссылка и file ID не
коммитятся в публичный репозиторий; их нужно брать из командного handoff.

Drive — это хранилище, а не static hosting: получателю следует скачать HTML
или ZIP и открыть локальный `index.html`. Права доступа папки необходимо
проверять отдельно; факт успешной загрузки не является проверкой доступа
третьего лица. Владелец **2026-09-10 явно подтвердил**, что у текущей папки
нужно сохранить режим «все, у кого есть ссылка, могут редактировать», и
передавать ссылку только Роме. Агент не должен самостоятельно менять этот
режим. После ответа Ромы существующие файлы в папке следует обновить на месте,
чтобы не появлялись неоднозначные копии.

## 7. Воспроизведение

Полная инструкция, включая входные артефакты, overlay, автоматический проброс
порта и значение элементов экрана: [Avito Case Browser](avito_case_browser.md).

```bash
conda activate diplom_avito

# Сформировать leakage-free prompts и browser без/с уже валидным cache
python scripts/avito/build_case_browser.py --generated-at 2026-09-08

# Сгенерировать недостающие инсайты; повторный запуск использует cache
CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=0 \
  python scripts/avito/generate_case_insights.py --device cuda:0 --batch-size 1

# Встроить проверенные инсайты и обновить bundle
python scripts/avito/build_case_browser.py --generated-at 2026-09-08

# Открыть локально
python -m http.server 8765 --bind 127.0.0.1 \
  --directory reports/avito_case_browser
```

Затем открыть `http://localhost:8765/`. Численная истина и полный список
ограничений находятся в приватном локальном
`results/current/metrics/avito_case_browser.json`; этот файл не коммитится.
