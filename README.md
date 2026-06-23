# diplom-avito-llm-rerank

Общий репозиторий команды: **UR4Rec + Exp3RT + Avito Auto** (LLM rerank / rating).

| Документ | Описание |
|----------|----------|
| [docs/TEAM_PROJECT.md](docs/TEAM_PROJECT.md) | формат проекта, команды, статус прогонов |
| [experiments/registry.yaml](experiments/registry.yaml) | реестр экспериментов (single source of truth) |
| [docs/avito_preferences.md](docs/avito_preferences.md) | преференсы на Авто, C-UR4Rec |

```bash
conda env create -f environment.yml   # или: conda activate diplom_avito
conda activate diplom_avito
pip install -r requirements.txt -r requirements-exp3rt.txt

bash scripts/status_runs.sh          # статус GPU и прогонов
```

**Данные и чекпоинты не в git** — см. `.gitignore`. Метрики: `results/current/metrics/`, `checkpoints/*/metrics*.json` (локально).

---

# Avito Auto — данные и EDA

## Файлы данных

| Файл | Строк | Суть |
|------|-------|------|
| `items_with_attrs.parquet` | 44 736 | **Показы в поисковой выдаче (SERP)** — одна строка = объявление в конкретной выдаче `serp_x` |
| `users_with_history.parquet` | 2 028 | **История контактов** — пользователь связался с объявлением (`contact_date`) |

### `items_with_attrs` — ключевые поля

- **Идентификаторы:** `serp_x` (сессия выдачи, ~2000 уникальных), `item_id`, `user_id` (продавец)
- **Позиция:** `block` (`items` / `extra-reg` / `extra-candidates-extra_regions`), `block_pos`
- **Запрос:** `query_infm_logical_category` (UsedCars / NewCars), `query_loc`
- **Объявление:** `price`, `title`, `description_short`, `brand`, `model_name`, `mileage_km`, …
- **Сигналы:** `clicks_daily`, `contacts_daily`, `serp_is_positive`, `price_percentile`, …

### `users_with_history` — ключевые поля

- `user_id`, `item_id`, `contact_date`
- Атрибуты авто: `brand`, `model_name` (часто пустые: `year_raw`, `mileage_km` — 100% NaN)

**Важно:** пересечение `item_id` между двумя файлами **пустое** — это разные срезы (показы SERP vs контакты), не одна таблица фактов.

---

## Conda-окружение `diplom_avito`

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate diplom_avito
```

Создание с нуля:

```bash
conda env create -f environment.yml
conda activate diplom_avito
python -m ipykernel install --user --name diplom_avito --display-name "diplom_avito"
```

---

## Как посмотреть данные

### 1. HTML-отчёт (быстро)

```bash
conda activate diplom_avito
cd /home/artem-gon/MIPT/DIPLOM/avito
python scripts/generate_eda_report.py
```

Открыть в браузере: `reports/eda_report.html`

### 2. Jupyter (интерактивно, таблицы + графики)

```bash
conda activate diplom_avito
cd /home/artem-gon/MIPT/DIPLOM/avito
jupyter notebook notebooks/01_eda_avito_data.ipynb
```

### 3. В Cursor / VS Code

Открыть `notebooks/01_eda_avito_data.ipynb`, kernel: **diplom_avito**.

### 4. Только pandas в REPL

```python
import pandas as pd
items = pd.read_parquet("items_with_attrs.parquet")
users = pd.read_parquet("users_with_history.parquet")
items.head(20)
items.sample(5)[["serp_x", "title", "price", "brand", "block_pos"]]
```

---

## UR4Rec: LLM-профили (§3.1)

По умолчанию **`use_template_generator: false`** — офлайн **Qwen2.5-7B-Instruct** или **DeepSeek-R1-Distill-Qwen-7B** (4-bit на V100). Llama2 в статье, но на HF нужен отдельный approve.

```bash
pip install -r requirements.txt

# Qwen (рекомендуется):
CUDA_VISIBLE_DEVICES=0 python scripts/run_ur4rec.py \
  --config configs/ur4rec_smoke_qwen.yaml --stage knowledge

# DeepSeek R1-Distill:
CUDA_VISIBLE_DEVICES=0 python scripts/run_ur4rec.py \
  --config configs/ur4rec_smoke_deepseek.yaml --stage knowledge
# затем --stage all
```

Подробнее: [docs/UR4Rec_code_and_reproduction.md](docs/UR4Rec_code_and_reproduction.md).

---

## UR4Rec: метрики на Avito parquet

Одна выдача = `serp_x`, метки — **`contacts_daily`** (или `clicks_daily` в конфиге).  
`serp_is_positive` везде `True`, для ранжирования не используется.  
`users_with_history` подключается к текстам сессий; **`item_id` между файлами не пересекаются** — история контактов не даёт label на объявления в SERP.

```bash
conda activate diplom_avito
cd /home/artem-gon/MIPT/DIPLOM/avito

CUDA_VISIBLE_DEVICES=0 python scripts/run_ur4rec.py \
  --config configs/ur4rec_avito_smoke.yaml \
  --stage all \
  2>&1 | tee logs/ur4rec_avito_smoke.log
```

Результат: `checkpoints/ur4rec_avito_smoke/metrics_test.json` (NDCG@K, MAP@K: base vs UR4Rec).

---

## Варианты визуализации (что имеет смысл дальше)

| Визуализация | Данные | Зачем |
|--------------|--------|-------|
| Распределение `price` (log) | items | Понять диапазон «Авто» |
| `block` / `block_pos` | items | Как устроена выдача, позиционный bias |
| Топ `brand` / `model_name` | items | Домен каталога |
| Размер SERP по `serp_x` | items | Длина candidate list для rerank |
| `clicks_daily` vs `contacts_daily` | items | Proxy релевантности / label |
| Контакты по датам | users | Временной охват истории |
| Длина истории на `user_id` | users | Sparsity для LettinGo/Persona4Rec |
| Join `user_id` (без `item_id`) | оба | Связать историю с показами одного пользователя |

Для диплома: **items** → LTR/rerank baseline; **users** → построение профиля / cold-warm анализ.
