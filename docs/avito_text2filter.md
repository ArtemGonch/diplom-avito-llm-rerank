# UsedCars. Разбор запроса и сравнение моделей

Локальный парсер возвращает `filters` и `unresolved`. Поля `start` и `end`
задают полуоткрытый диапазон индексов символов исходной строки. По нему можно
восстановить `evidence`. Словарный фильтр хранит пару `attr_id` и `value_id`.
Числовой хранит `min`, `max` и при необходимости признак строгой границы.
Это внутренний формат эксперимента. Для интеграции в Avito потребуется контракт
поискового API и таблица допустимых зависимостей атрибутов.

## Команды

```bash
python scripts/avito/text2filter.py --archive "$QUERIES_ZIP" parse 'автомат до 1,5 млн без ДТП'
python scripts/avito/text2filter.py --archive "$QUERIES_ZIP" serve --port 8782
python scripts/avito/text2filter.py --archive "$QUERIES_ZIP" packet --out reports/text2filter_annotation
```

Демо доступно по `http://127.0.0.1:8782/`. Входы, ответы моделей, веса и ручные
оценки остаются в игнорируемых каталогах. Примеры в интерфейсе синтетические.

Для небольшой локальной выборки можно запустить Qwen3-4B-4bit через MLX на
Apple Silicon. Модель получает до 20 вариантов, подтверждённых словарём или
числовым разбором с фрагментом исходного запроса. Вариант с коллизией ID
остаётся нерешённым. Выбор модели проверяется по списку вариантов. Возврат
парсера при ошибке модели имеет отдельный статус. При оценке такие запросы
входят в общую выборку с результатом парсера, число замен выводится отдельно.

```bash
python scripts/avito/text2filter.py --archive "$QUERIES_ZIP" teacher \
  --manifest reports/text2filter_annotation/sampling.jsonl --limit 50 \
  --qwen "$QWEN_WEIGHTS" --out reports/qwen_pilot.jsonl
```

Перед использованием псевдометок проверяющий сохраняет принятые строки со
статусом `reviewed`. Студент классифицирует пары «запрос, предложенный фильтр»
общим для обоих вариантов линкером. `--base` указывает на локальную копию
`cointegrated/rubert-tiny2`. Для контроля применяются одинаковые seed,
архитектура, порог 0,5 и число шагов. Для второго обучения добавляется
`--pseudo` с проверенными модельными строками. Обе версии требуют полных
80 исходных эталонных примеров из зафиксированного train.

```bash
python scripts/avito/text2filter.py --archive "$QUERIES_ZIP" train \
  --train reports/text2filter_annotation/train.csv \
  --manifest reports/text2filter_annotation/sampling.jsonl \
  --base "$RUBERT_WEIGHTS" --out checkpoints/text2filter_human
python scripts/avito/text2filter.py --archive "$QUERIES_ZIP" predict \
  --input reports/text2filter_annotation/test.csv \
  --student checkpoints/text2filter_human --out reports/text2filter_test_predictions.jsonl
python scripts/avito/text2filter.py score --gold reports/text2filter_annotation/test.csv \
  --predictions reports/text2filter_test_predictions.jsonl \
  --manifest reports/text2filter_annotation/sampling.jsonl
```

`predict` без модели запускает правила. `--qwen` даёт оценку прямого выбора
через Qwen на том же списке кандидатов. Оценка закрытого теста использует
полностью заполненную независимую эталонную разметку. Для макрооценки семейства взвешены
по обратной вероятности отбора, для оценки трафика вес дополнительно
умножен на частоту семейства. Представитель семейства отбирается с
вероятностью, пропорциональной его частоте. Задержку замеряем после
прогрева, без загрузки весов и индекса.

При оценке двух вариантов BERT используйте ту же разметку и одинаковые ID
запросов. Команда `compare` считает попарную разницу доли точных разборов
между вариантами и интервалы по 2 000 повторных выборок внутри страт.
Десять самых частых семейств входят в каждый повтор целиком. Интервалы
отражают неопределённость отбора при фиксированной разметке.

```bash
python scripts/avito/text2filter.py compare --gold "$TEST_REFERENCE" \
  --control "$BERT_CONTROL_PREDICTIONS" --augmented "$BERT_AUGMENTED_PREDICTIONS" \
  --manifest reports/text2filter_annotation/sampling.jsonl
```

Альтернативный источник эталона — слепая разметка от независимых экземпляров
GPT-6.2 Astra xhigh. В этом режиме измеряется совпадение с её разметкой на
реальных запросах. Версию модели-разметчика, объём повторной разметки и
согласие разметчиков фиксируют вместе с метриками.

Эксперимент от 25 ноября использует 80 эталонных train, 40 dev и 120 test.
На 30 тестовых запросах повторно сравнили наборы фильтров, совпали 27 из 30.
Для трёх расхождений использована отдельная слепая разметка Astra. Для одной
половины проверки взяли первую завершённую резервную разметку до подсчёта
метрик моделей. Из 80
проверенных обучающих ответов Qwen 34 совпали по набору фильтров с независимой
разметкой и вошли во второй BERT. Оба BERT дали одинаковые разборы на всех
120 запросах. [Агрегированные метрики](../results/current/metrics/avito_text2filter_astra_20261125.json).

## Пересчёт BX на нормализованном пилоте

Из полученного ZIP извлеките `candidates.parquet` в приватный каталог и
передайте исходный Parquet для строгой сверки. Пилот целиком относится к
test. Для шардов с явной колонкой split предусмотрен `--candidates`.
Для файлов без неё задайте по одному `--candidate-split` на файл.

```bash
python scripts/avito/evaluate_bx_baseline.py \
  --candidate-split test "$NORMALIZED_CANDIDATES" --original "$ORIGINAL_PILOT"
```
