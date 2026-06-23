# Полная история проекта диплома: LLM в ранжировании (Avito)

> **Назначение файла:** передать контекст новому агенту / новой сессии Cursor на другой машине.  
> **Как использовать:** `@docs/AGENT_HANDOFF.md` в первом сообщении + конкретная задача.  
> **Репозиторий:** `/home/artem-gon/MIPT/DIPLOM/avito`  
> **Автор:** Гончаров Артём, М05-513в · **Научный руководитель:** Роман Будылин  
> **Семестр:** заканчиваем **2-й семестр** (обзор литературы + EDA + baseline replication в процессе)

---

## Оглавление

1. [Суть проекта](#1-суть-проекта)
2. [Критерии Avito/MIPT и что уже закрыто](#2-критерии-avitomipt-и-что-уже-закрыто)
3. [Хронология работы с агентом](#3-хронология-работы-с-агентом)
4. [Обзор литературы: прочитанные статьи](#4-обзор-литературы-прочитанные-статьи)
5. [Презентация: как делали и финальная структура](#5-презентация-как-делали-и-финальная-структура)
6. [Данные Avito (parquet)](#6-данные-avito-parquet)
7. [Воспроизведение UR4Rec: код и архитектура](#7-воспроизведение-ur4rec-код-и-архитектура)
8. [Эксперименты, метрики, конфиги](#8-эксперименты-метрики-конфиги)
9. [Идея собственного вклада: C-UR4Rec](#9-идея-собственного-вклада-c-urrec)
10. [Инфраструктура: V100, A100, SSH, перенос](#10-инфраструктура-v100-a100-ssh-перенос)
11. [Структура репозитория и команды](#11-структура-репозитория-и-команды)
12. [Что сделано / что дальше](#12-что-сделано--что-дальше)
13. [Ссылки](#13-ссылки)

---

## 1. Суть проекта

### Тема (рабочая формулировка)

**LLM в ранжировании выдачи объявлений** — cost-aware применение больших языковых моделей в многостадийном пайплайне маркетплейса (retrieval → rank → rerank), с фокусом на вертикаль **«Авто»** Avito.

### Product decision (что решает продукт)

Команда вертикали «Авто» выбирает **архитектуру ранжирования top-N выдачи**:

| Вариант | Описание |
|---------|----------|
| **A** | Только классический ranker (LightGBM / DCN / DLCM) |
| **B** | LLM **offline** для обогащения признаков + ranker |
| **C** | LLM **online rerank** top-K после ranker |
| **D** | Hybrid: routing (когда LLM окупается, когда нет) |

Диплом должен дать **сравнение подходов** с метриками качества, latency, GPU-часов и ограничениями продакшена.

### Research questions (черновик)

1. Где LLM даёт измеримый прирост NDCG/MAP на rerank, а где достаточно LTR?
2. Как **сжать/отобрать** LLM-сигнал (retriever, routing, gating), не тратя запросы на каждый item?
3. Как оценивать LLM-ranker **не только NDCG** (bias, fairness, calibration)?
4. Как перенести academic rerank (MovieLens) на **search-aware SERP Avito**?

### Выбранный baseline для воспроизведения

**UR4Rec** (COLING 2025) — retriever поверх LLM-knowledge для rerank.  
Причины: прямо про «не тащить весь LLM-текст в reranker», есть численные таблицы, близко к Avito (rerank top-N после ranker).  
Официального кода **нет** → реализовали сами по §3 paper.

---

## 2. Критерии Avito/MIPT и что уже закрыто

### Требования к диплому (кратко)

- Problem statement + цена ошибки + ограничения (latency, бюджет, данные)
- Обзор **30–50** работ, trade-offs, practical gap для Avito
- **Воспроизведённый baseline** (seeds, splits, README)
- **2–3 альтернативы** + ablations
- Собственный вклад (алгоритм / инженерное улучшение)
- Limitations, воспроизводимость, аналитика экспериментов

### Образ результата 2-го семестра (цель 8+)

| Критерий | Статус |
|----------|--------|
| Тема + problem statement | ✅ сформулировано |
| Обзор литературы | ✅ ~7–10 ключевых статей разобраны глубоко + обзор Hou et al. |
| Презентация обзора | ✅ Google Slides, текст выступления ~45 мин |
| EDA данных | ✅ Avito parquet + HTML-отчёт + notebook |
| Baseline replication | 🔄 UR4Rec smoke пройден; beat baseline на ML-1M **в процессе** |
| План экспериментов на диплом | ✅ C-UR4Rec + ablations (черновик) |
| Аннотация / текст ВКР | ⏳ черновики обсуждались, не финализированы |

### Презентация Google Slides

https://docs.google.com/presentation/d/11wAZZad4MuDdk8JIZy_AySQ7jVjIxX82lJl0EyGJRiU/edit

Формат слайдов: **картинка из статьи + 4 буллета** (смысл + короткое раскрытие) + **ссылка на paper внизу**.  
Референс по стилю — презентация по другой теме (музыка/диффузия), от старых слайдов остатки **удалены**.

---

## 3. Хронология работы с агентом

| Этап | Что делали |
|------|------------|
| **Старт** | Разбор критериев Avito/MIPT, план диплома на 4 семестра, тема LLM в ranking/recs, product scenario = поиск/rank «Авто» |
| **Обзор 2026** | Подбор survey-статей, план под максимальный балл |
| **Survey Hou et al.** | Полный разбор *A Survey on Generative Recommendation* (Data / Model / Task) |
| **5 статей от научрука** | LLM4Rerank, Exp3RT, Beyond Utility, Reason4Rec, UR4Rec — критический разбор каждой |
| **Презентация v1** | 3 статьи (UR4Rec, Beyond Utility, LLM4Rerank) + SOTA TWn |
| **Презентация v2** | Расширение: LettinGo, Exp3RT, Persona4Rec; Beyond Utility → appendix; текст выступления 45 мин |
| **EDA Avito** | Разбор `items_with_attrs.parquet`, `users_with_history.parquet`, conda `diplom_avito`, notebook + HTML |
| **UR4Rec code hunt** | Официального кода нет; план через KAR + свой retriever → `docs/UR4Rec_code_and_reproduction.md` |
| **Реализация UR4Rec** | Полный pipeline `scripts/run_ur4rec.py`, smoke на ML-1M и Avito |
| **LLM** | Qwen2.5-7B-Instruct (4bit) вместо gated Llama-2; промпты из §3.1 paper |
| **Датасеты paper** | ML-1M на сервере; Amazon/Steam — скачивание с Mac + rsync; фикс URL Amazon (mcauleylab.ucsd.edu) |
| **Метрики smoke** | UR4Rec хуже base на ML-1M smoke — объяснено; оценка GPU-часов для paper-level |
| **Beat baseline run** | `ur4rec_ml1m_beat_base.yaml` + `run_beat_baseline.sh`, ~8–10 GPU-h |
| **Миграция V100→A100** | Перенос только `avito/` (~1.7G), не весь DIPLOM (81G — старый диплом про diffusion) |

---

## 4. Обзор литературы: прочитанные статьи

### 4.1. Рамка: Generative Recommendation Survey

**Hou et al., *A Survey on Generative Recommendation: Data, Model, and Tasks***  
[arXiv:2510.27157](https://arxiv.org/abs/2510.27157) · ~200+ работ

**Идея:** три уровня, где может стоять LLM:

```text
Data   — offline обогащение (профили, тексты, knowledge graphs)
Model  — LLM как reranker / augment / retriever
Task   — генерация списка, объяснений, диалог
```

**Figure 2** — главная картинка для слайда 3 презентации.  
Для Avito «Авто»: LLM чаще всего **Model (rerank)** или **Data (offline features)**, не full generative rank по всему каталогу.

---

### 4.2. LettinGo (KDD 2025)

**Wang et al., *LettinGo: Explore User Profile Generation for Recommendation System***  
[arXiv:2506.18309](https://arxiv.org/abs/2506.18309)

- **Уровень:** Data (offline)
- **Идея:** LLM сжимает длинную history → **текстовый user profile** → downstream rec model
- **3 стадии:** profile generation → integration → recommendation
- **Метрики:** Acc/F1 на MovieLens-10M, Amazon Books, Yelp
- **Для Avito:** профиль из `users_with_history` (бренды, контакты), но **нет search query**

---

### 4.3. UR4Rec (COLING 2025) — **основной baseline диплома**

**Zhang et al., *Enhancing Reranking for Recommendation with LLMs through User Preference Retrieval***  
[ACL Anthology](https://aclanthology.org/2025.coling-main.45/) · PDF: [2025.coling-main.45.pdf](https://aclanthology.org/2025.coling-main.45.pdf)

**Проблема KAR:** весь LLM user preference слишком длинный → шум в reranker.

**Решение:**

```text
Offline: Llama2-Chat → user preference su + item knowledge si (по history items)
         Frozen BERT → eu, eik, e_u^aggr = concat embeddings

Pretrain retriever:
  K learnable proxy embeddings P
  Self-attn + cross-attn: candidate h_t = query, knowledge e_u^aggr = keys/values
  Loss: L_CL (InfoNCE) + α·L_CF (preference-item matching)

Joint: e_aug = Agg(Retriever([P; h_t], e_u^aggr))
       concat e_aug → DLCM / SASRec / PRM / SetRank
```

**Как retriever «понимает», что брать:** не правилами, а **обучением** — embedding кандидата задаёт query, cross-attention выделяет релевантные части aggregated knowledge.

**Table 2 (paper):** ML-1M base NDCG@10 ~0.315 → UR4Rec ~**0.661**; Amazon, Steam аналогично.

**Маски (Fig. 3c):** isolation masking в cross-attention — candidate не «утекает» в knowledge side.

---

### 4.4. Exp3RT (SIGIR 2025)

**Kim et al., *Review-driven Personalized Preference Reasoning with LLMs for Recommendation***  
[arXiv:2408.06276](https://arxiv.org/abs/2408.06276) · код: [EXP3RT](https://github.com/jieyong99/EXP3RT)

- **3 шага:** Like/Dislike из reviews → user/item profiles → CoT reasoning → rating → rerank
- **Distillation** в smaller model
- **Риск:** нужны **reviews** (на Avito SERP их мало в том же виде)
- **Место в линейке:** reasoning из **текстов**, не из ID-only history

---

### 4.5. LLM4Rerank (WWW 2025)

**Gao et al., *LLM4Rerank: LLM-based Auto-Reranking Framework for Recommendations***  
[arXiv:2406.12433](https://arxiv.org/abs/2406.12433)

- **Zero-shot** multi-aspect rerank: accuracy + diversity + fairness
- **Function graph** узлов + **Goal** (natural language) → reorder list
- **~14 s/sample** — дорого для online
- **Для Avito:** multi-aspect (релевантность + разнообразие брендов + fairness geo) — концептуально близко

---

### 4.6. Beyond Utility (бенчмарк)

**Jiang et al., *Beyond Utility: Evaluating LLM-based Recommendation Systems***  
[arXiv:2411.00331](https://arxiv.org/abs/2411.00331)

- **Не метод**, а **6 измерений** оценки LLM-rec (не только NDCG)
- Position bias, rerank vs full rank, calibration
- **В презентации:** перенесено в **appendix** (по совету научрука — не перегружать основной доклад)

---

### 4.7. Persona4Rec (2026)

**Kim et al., *Offline Reasoning for Efficient Recommendation: LLM-Empowered Persona-Profiled Item Indexing***  
[arXiv:2602.21756](https://arxiv.org/abs/2602.21756) · код: [PERSONA4REC](https://github.com/legenduck/PERSONA4REC)

- Offline LLM reasoning → persona profiles → online **dot-product**, ~**0.75 ms**
- Ответ на latency LLM4Rerank
- **Линейка:** «убрать LLM из online полностью»

---

### 4.8. Think When Needed (2026) — SOTA в презентации

**Guo et al., *Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking***  
[arXiv:2601.18146](https://arxiv.org/abs/2601.18146)

- Router **до** генерации: Think vs Non-Think mode
- **Checklist probing + isolation masking** (§4.2): Yes/No вопросы в suffix prompt, изолированы от ranking generation → model-aware difficulty signals
- Закрывает фокусы доклада: **калибровка, мало запросов, когда reasoning окупается**

---

### 4.9. Другие статьи из списка научрука (кратко)

| Paper | arXiv | Суть |
|-------|-------|------|
| Reason4Rec | [2502.02061](https://arxiv.org/abs/2502.02061) | Reasoning для explainable rec |
| (из review command) | [2405.18616](https://arxiv.org/abs/2405.18616) | Отдельная работа из cursor command |

---

### 4.10. Фокусы обзора (из ТЗ доклада)

- Как **калибровать** LLM-ranker
- Как **не тратить** слишком много запросов, но получить качество
- Достаточна ли **экспертиза LLM** для задачи
- **Ограниченный контекст**
- Близкие референсы в литературе
- **Идемпотентность** (повторный запрос → тот же augment)

---

## 5. Презентация: как делали и финальная структура

### Процесс

1. Взяли **формат** референс-презы (картинка + буллеты)
2. Сначала 3 метода + SOTA + выводы
3. Научник попросил **убрать Beyond Utility** из основной части → appendix
4. Добавили **LettinGo, Exp3RT, Persona4Rec** для полной линейки Data→Model→latency→routing
5. Написали **единый текст выступления ~45 мин** по слайдам
6. Финальная проверка экспорта Slides → «можно отправлять»

### Финальная структура (основная часть)

| № | Слайд | Paper |
|---|-------|-------|
| 1 | Титул: LLM в ранжировании выдачи объявлений · Гончаров Артём · Будылин | — |
| 2 | План доклада | — |
| 3 | Data / Model / Task | Hou et al. 2025 |
| 4–5 | LettinGo | Wang et al. KDD 2025 |
| 6–7 | UR4Rec | Zhang et al. COLING 2025 |
| 8–9 | Exp3RT | Kim et al. SIGIR 2025 |
| 10–11 | LLM4Rerank | Gao et al. WWW 2025 |
| 12–13 | Persona4Rec | Kim et al. 2026 |
| 14–15 | Think When Needed (SOTA) | Guo et al. 2026 |
| 16 | Выводы (4 буллета) | — |
| 17 | Спасибо + литература | — |

**Appendix:** Beyond Utility (Jiang et al.)

### Выводы на слайде (итог обзора)

1. LLM в **Model**, не вместо LTR (offline Data → rerank top-K)
2. **Rerank** — общий паттерн (UR4Rec, LLM4Rerank)
3. Сигнал LLM нужно **отбирать** (UR4Rec retriever; TWn routing)
4. Метрики и **bias** обязательны (Beyond Utility)

### Текст выступления

Полный сценарий ~45 мин был сгенерирован в чате (слайд за слайдом, с таймингом).  
Ключевые имена в устной части: **Гончаров Артём, М05-513в, научрук Роман Будылин**.

---

## 6. Данные Avito (parquet)

### Файлы в корне `avito/`

| Файл | Строк | Суть |
|------|-------|------|
| `items_with_attrs.parquet` | 44 736 | **SERP** — показ объявления в выдаче `serp_x` |
| `users_with_history.parquet` | 2 028 | **Контакты** user → item (`contact_date`) |

### Ключевые поля items

- `serp_x`, `item_id`, `user_id` (продавец)
- `block`, `block_pos` — позиция в выдаче
- `query_infm_logical_category` (UsedCars / NewCars), `query_loc`
- `title`, `description_short`, `brand`, `model_name`, `price`, `mileage_km`
- `clicks_daily`, `contacts_daily`, `serp_is_positive`

### Ключевые поля users

- `user_id`, `item_id`, `contact_date`, `brand`, `model_name`

### Важные факты EDA (`reports/summary.json`)

- ~2000 уникальных SERP, mean size ~22, max 435
- Категории: UsedCars 43676, NewCars 1060
- **`overlap_item_id` между файлами = 0** — разные срезы, не joint по item
- `overlap_user_id` = 274 — можно связать историю с SERP **по user**, не по item label

### Как смотреть

```bash
conda activate diplom_avito
cd /home/artem-gon/MIPT/DIPLOM/avito
python scripts/generate_eda_report.py   # → reports/eda_report.html
jupyter notebook notebooks/01_eda_avito_data.ipynb
```

---

## 7. Воспроизведение UR4Rec: код и архитектура

### Официального кода нет

Ближайший open-source: **KAR** [Open-World-Knowledge-Augmented-Recommendation](https://github.com/YunjiaXi/Open-World-Knowledge-Augmented-Recommendation).  
Мы реализовали UR4Rec **строго по §3 + Algorithm 1 (Appendix A)**.

### Pipeline (`scripts/run_ur4rec.py`)

```text
Stage 1: knowledge   — offline LLM (Qwen2.5 / DeepSeek / Llama2 / template)
Stage 2: backbone    — DLCM reranker без augmentation
Stage 3: pretrain    — retriever L_CL + L_CF
Stage 4: joint       — retriever + backbone с e_aug
Stage 5: eval        — MAP/NDCG@1,5,10
```

### Модули `src/ur4rec/`

| Модуль | Назначение |
|--------|------------|
| `retriever.py` | Transformer retriever, proxies P, cross-attention |
| `masks.py` | Isolation masks (Fig. 3c) |
| `text_encoder.py` | Frozen BERT для LLM-текстов |
| `backbone.py` | DLCM reranker |
| `losses.py` | InfoNCE + listwise CE |
| `metrics.py` | NDCG, MAP |
| `data/ml1m.py` | MovieLens-1M rerank samples |
| `data/amazon_books.py`, `steam.py` | Paper datasets |
| `data/avito.py` | SERP rerank из parquet |
| `llm/prompts.py` | Промпты §3.1 (ML1M / Amazon / Steam / Avito) |
| `llm/hf_chat_generator.py` | Qwen / generic HF chat |
| `llm/knowledge_batch.py` | Batch generation + resume + ETA |

### LLM choice

| Backend | Model | Примечание |
|---------|-------|------------|
| **`qwen`** (default) | `Qwen/Qwen2.5-7B-Instruct` | Рекомендуется, 4bit на V100/A100 |
| `deepseek` | DeepSeek-R1-Distill-Qwen-7B | Хуже для коротких 128-token промптов |
| `llama2` | Llama-2-7b-chat | Как в paper, нужен HF approve |

Paper использует **Llama2-Chat**; мы заменили на **Qwen2.5** для открытого доступа.

### Промпты (§3.1)

- **User:** keywords summarizing preference from history (title + genre/category per item)
- **Item:** overview from title + categories (Steam в paper; ML1M = title + genres)

Код: `src/ur4rec/llm/prompts.py`

---

## 8. Эксперименты, метрики, конфиги

### Датасеты UR4Rec (Table 1 paper)

| Dataset | Путь | Статус на сервере |
|---------|------|-------------------|
| MovieLens-1M | `data/movielens-1m/` | ✅ OK |
| Amazon-Books 5-core | `data/amazon-books/` | ⏳ скачать с Mac, rsync |
| Steam | `data/steam/` | ⏳ скачать с Mac, rsync |

```bash
python scripts/download_ur4rec_datasets.py --status
```

Split: 8:1:1, history_len 10, rerank top-100 (paper) / top-50 (beat_base) / top-20 (smoke).

### Smoke ML-1M (`ur4rec_smoke_qwen.yaml`)

**Намеренно урезано:** 200 train users, 20 candidates, caps 250 items / 300 users LLM, 1 epoch.

| Model | NDCG@10 |
|-------|---------|
| Base DLCM | **0.362** |
| UR4Rec | **0.337** ← хуже base |
| Paper full | base ~0.315, UR4Rec ~**0.661** |

**Почему UR4Rec хуже:** неполное knowledge → template fallback, 1 epoch retriever, 50 test users, 20 candidates (легче для base).

### Smoke Avito (`ur4rec_avito_smoke_qwen.yaml`)

| Model | NDCG@10 |
|-------|---------|
| Base | 0.929 |
| UR4Rec | 0.930 |

Задача слишком лёгкая (20 cand, 100 SERP) — **не сравнимо** с paper.

### Beat baseline — variant 1 (текущий основной run)

**Конфиг:** `configs/ur4rec_ml1m_beat_base.yaml`  
**Скрипт:** `bash scripts/run_beat_baseline.sh`

| Параметр | Значение |
|----------|----------|
| Knowledge | full, no caps, 128 tokens, Qwen 4bit |
| ~LLM calls | ~3883 items + ~6037 users |
| Candidates | 50 |
| Train | backbone 3 / pretrain 3 / joint 5 epochs |
| Test users | 600 |
| GPU time | ~**8–10 h** V100; быстрее на A100 (batch_size 8 опционально) |
| Output | `checkpoints/ur4rec_ml1m_beat_base/metrics_test.json` |

**Knowledge dir:** `data/movielens-1m/knowledge_qwen25_beat_base/`  
**Состояние при переносе:** ~**100 items** сгенерировано на V100, users пусто, checkpoint every 50 — **resume продолжит**.

### Full paper reproduction

**Конфиг:** `configs/ur4rec_ml1m_full.yaml`  
512 tokens, 100 candidates, ~9920 LLM calls, **~25–35 h** knowledge + **~6–10 h** train.

### Оценки GPU-часов (из обсуждений)

| Бюджет | Ожидание NDCG@10 UR4Rec vs base |
|--------|----------------------------------|
| 2 h | ~0.35–0.40 vs base 0.37–0.42, **не гарантия** победы |
| 8–10 h (beat_base) | UR4Rec > base **вероятно** (>90%) |
| 35+ h (full) | Близко к paper ~0.66 |

---

## 9. Идея собственного вклада: C-UR4Rec

Рабочее название: **C-UR4Rec: Context-aware User Preference Retrieval**

### Слабости UR4Rec → мотивация

| Проблема | Для Avito |
|----------|-----------|
| Один статический `s_u` на весь list | Интересы зависят от query/category |
| Нет search query в retriever | SERP = query + candidates |
| Item knowledge только history | Нужен knowledge **кандидата** |
| Нет gating при плохом LLM | Smoke: UR4Rec < base |
| Дорогая full knowledge generation | Partial knowledge ломает метрики |

### Три вклада (план)

1. **Multi-aspect user memory** — K аспектов (цена, бренд, geo…) вместо одного concat
2. **Candidate–context conditioned retrieval** — query = concat(h_cand, e_query, e_category)
3. **Confidence gating** — e_aug = gate · Retriever(...); gate→0 если retriever не уверен

### План экспериментов

| Этап | Ablation |
|------|----------|
| Baseline | DLCM, KAR-full-aug, UR4Rec repro |
| +1 | multi-aspect memory |
| +2 | candidate item knowledge |
| +3 | confidence gate |
| Main | C-UR4Rec full на ML-1M / Amazon / Steam |
| Transfer | Avito SERP |

**Доп. метрика:** NDCG при 25% / 50% / 100% knowledge coverage (robustness).

---

## 10. Инфраструктура: V100, A100, SSH, перенос

### Машины

| Alias | Хост | GPU | Роль |
|-------|------|-----|------|
| `ads-4v100` | ads-4v100.sas.yp-c.yandex.net | 4× V100 32GB | Старые эксперименты, beat run начат |
| `ads-8a100` | a100-1.vla.yp-c.yandex.net | A100 | **Целевая** для всех дальнейших run |

Подключение с Mac: `ssh ads-4v100` / `ssh ads-8a100` (интерактивно OK).

### SSH/rsync проблема

**Non-interactive** `ssh host 'cmd'` и **rsync через ssh** на V100 **зависают** (zsh/skotty login shell).  
Симптом: `rsync --server --sender` висит после `Warning: Permanently added...`

**Не использовать** rsync A100↔V100 для проверки переноса.

**Обход:** интерактивный ssh; или manifest на каждой машине + diff локально; или `ssh -T /bin/bash --norc --noprofile -c '...'`.

### Перенос V100 → A100

- ✅ **Правильно:** только `/home/artem-gon/MIPT/DIPLOM/avito/` ≈ **1.7G**
- ❌ **Неправильно:** весь `DIPLOM/` = **81G** (старый диплом `symbolic-music-discrete-diffusion` + zip)

**Размеры на A100 (проверено):**

```text
avito/        1.7G
  checkpoints 1.6G
  data        31M
  src         300K
  scripts     64K
  configs     44K
  logs        ~300K
```

### Conda на A100

```bash
cd ~/MIPT/DIPLOM/avito
conda env create -f environment.yml   # если env не копировали
conda activate diplom_avito
python -c "from ur4rec.data.ml1m import MovieLens1M; print('OK')"
```

### Cursor на A100

Remote-SSH → open folder `~/MIPT/DIPLOM/avito` → новый Agent chat → `@docs/AGENT_HANDOFF.md`

---

## 11. Структура репозитория и команды

```text
avito/
├── items_with_attrs.parquet
├── users_with_history.parquet
├── environment.yml
├── requirements.txt
├── README.md
├── configs/
│   ├── ur4rec_smoke_qwen.yaml
│   ├── ur4rec_ml1m_beat_base.yaml      # ← текущий main run
│   ├── ur4rec_ml1m_full.yaml
│   ├── ur4rec_avito_smoke_qwen.yaml
│   ├── ur4rec_amazon_smoke_qwen.yaml
│   └── ur4rec_steam_smoke_qwen.yaml
├── scripts/
│   ├── run_ur4rec.py
│   ├── run_beat_baseline.sh
│   ├── download_ur4rec_datasets.py
│   └── generate_eda_report.py
├── src/ur4rec/                         # реализация paper
├── data/
│   ├── movielens-1m/
│   │   └── knowledge_qwen25_beat_base/ # partial ~100 items
│   └── avito/knowledge_qwen25/
├── checkpoints/
├── logs/
├── notebooks/01_eda_avito_data.ipynb
├── reports/eda_report.html
└── docs/
    ├── AGENT_HANDOFF.md                # этот файл
    └── UR4Rec_code_and_reproduction.md
```

### Частые команды

```bash
# Статус датасетов
python scripts/download_ur4rec_datasets.py --status

# Smoke ML-1M
CUDA_VISIBLE_DEVICES=0 python scripts/run_ur4rec.py \
  --config configs/ur4rec_smoke_qwen.yaml --stage all

# Beat baseline (resume knowledge)
bash scripts/run_beat_baseline.sh

# Логи beat run
tail -f logs/beat_baseline_master.log
tail -f logs/beat_baseline_knowledge.log
tail -f logs/beat_baseline_train.log

# Метрики
cat checkpoints/ur4rec_ml1m_beat_base/metrics_test.json
```

### Чеклист «перенос OK» (локально на A100)

```bash
cd ~/MIPT/DIPLOM/avito
du -sh . data checkpoints
for f in scripts/run_ur4rec.py configs/ur4rec_ml1m_beat_base.yaml \
  data/movielens-1m/ml-1m/ratings.dat \
  data/movielens-1m/knowledge_qwen25_beat_base/items.json; do
  [ -e "$f" ] && echo OK $f || echo MISS $f
done
```

---

## 12. Что сделано / что дальше

### ✅ Сделано

- [x] Тема, problem statement, план диплома под критерии Avito
- [x] Глубокий разбор 7+ статей + survey Hou et al.
- [x] Презентация Google Slides + текст 45 мин
- [x] EDA Avito + conda `diplom_avito`
- [x] Полная реализация UR4Rec pipeline
- [x] Smoke ML-1M + Smoke Avito
- [x] Документация reproduction
- [x] Beat baseline config + launch script
- [x] Перенос проекта на A100 (~1.7G)

### 🔄 В процессе

- [ ] **Beat baseline run** — knowledge ~100/9920 items, resume на A100
- [ ] Amazon-Books + Steam datasets на сервер
- [ ] Аннотация / related work в Overleaf

### ⏳ Следующие шаги (приоритет)

1. `conda activate diplom_avito` на A100
2. `bash scripts/run_beat_baseline.sh` — resume knowledge + train
3. С Mac: rsync Amazon + Steam в `data/`
4. После beat base: решить full repro vs **C-UR4Rec MVP** на Avito
5. 3-й семестр: ablations, таблицы, текст Experiments

### Задание на практику (черновик обсуждался)

Цель: обзор LLM в ranking, product context «Авто», выбор baseline (LightGBM lambdarank + UR4Rec repro), план НИР.

---

## 13. Ссылки

### Презентация и документы

- Google Slides: https://docs.google.com/presentation/d/11wAZZad4MuDdk8JIZy_AySQ7jVjIxX82lJl0EyGJRiU/edit
- UR4Rec reproduction notes: [UR4Rec_code_and_reproduction.md](./UR4Rec_code_and_reproduction.md)
- README проекта: [../README.md](../README.md)

### Ключевые papers

| Short | URL |
|-------|-----|
| Generative Rec Survey | https://arxiv.org/abs/2510.27157 |
| UR4Rec | https://aclanthology.org/2025.coling-main.45/ |
| LettinGo | https://arxiv.org/abs/2506.18309 |
| Exp3RT | https://arxiv.org/abs/2408.06276 |
| LLM4Rerank | https://arxiv.org/abs/2406.12433 |
| Beyond Utility | https://arxiv.org/abs/2411.00331 |
| Persona4Rec | https://arxiv.org/abs/2602.21756 |
| Think When Needed | https://arxiv.org/abs/2601.18146 |
| KAR (baseline pred) | https://github.com/YunjiaXi/Open-World-Knowledge-Augmented-Recommendation |

### Датасеты

- MovieLens-1M: https://grouplens.org/datasets/movielens/1m/
- Amazon Reviews: https://mcauleylab.ucsd.edu/public_datasets/data/amazon/
- Steam (SASRec repo): https://github.com/kang205/SASRec

---

*Последнее обновление: май 2026. При изменении экспериментов дополняй секции 8, 10, 12.*
