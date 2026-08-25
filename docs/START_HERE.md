# Диплом: актуальная точка входа

Обновлено: **2026-08-25**. Этот файл — краткий текущий контекст проекта. Он не заменяет статьи, код и реестр экспериментов, а задаёт правильный порядок чтения и отделяет валидные результаты от legacy.

Новый Codex получает корневой `AGENTS.md` автоматически. Для полного onboarding следует использовать repo-skill `$diplom-context`; если skills недоступны, выполнить `bash scripts/project_context.sh` и следовать маршрутам чтения ниже.

## Суть диплома

Тема: **LLM в ранжировании выдачи объявлений**. Исследуется cost-aware многостадийный pipeline `retrieval → rank/LTR → rerank`, где LLM не ранжирует весь каталог online, а offline строит знания/профили/память, которые использует дешёвый candidate-aware reranker.

Основной воспроизводимый baseline — **UR4Rec** (COLING 2025), дополнительный — **Exp3RT** (SIGIR 2025). Целевой домен — Avito Auto. Планируемый собственный вклад — **C-UR4Rec**: UR4Rec с query/context-conditioned retrieval, multi-aspect memory и confidence gate.

Главный исследовательский вопрос: когда семантическая память от LLM улучшает top-K rerank по сравнению с обычным ranker и окупает стоимость генерации, особенно при короткой/шумной истории и неполном поисковом контексте.

## Иерархия источников истины

При расхождении файлов использовать такой приоритет:

1. Код и выбранный YAML-конфиг — фактическое поведение.
2. `experiments/registry.yaml` — статус запуска и путь к артефактам.
3. `checkpoints/*/metrics*.json` и `results/current/manifest.json` — численные результаты.
4. Этот файл и узкие документы в `docs/` — объяснение.
5. `docs/AGENT_HANDOFF.md` — только история до актуального аудита; не источник текущего статуса.

Статус запущенного процесса проверять по логам, а не по дате документа:

```bash
bash scripts/status_runs.sh
tail -f logs/ur4rec_corrected_v3/master.log
tail -f logs/ur4rec_corrected_v3/knowledge_shard0.log
```

## Текущее состояние экспериментов

| Эксперимент | Статус | Что можно использовать |
|---|---|---|
| UR4Rec ML-1M corrected-v3 | **running**, stage `knowledge` запущен 2026-08-25 на GPU `2,4,5,6` | Пока только protocol/config/code; финальных метрик ещё нет |
| UR4Rec ML-1M beat-base / guaranteed | done, **legacy** | Только как история отладки; веса и метрики предшествуют correctness fixes |
| Exp3RT Amazon rating-only | done | Отдельный shortcut baseline, не выдавать за полный paper pipeline |
| Exp3RT Amazon chained paper-full | done | Expected RMSE `0.5624`, MAE `0.3496`, `n=11743` |
| Exp3RT-style Avito | done | Leakage-free graded NDCG@10 `0.3413` против position baseline `0.3126` |
| UR4Rec Avito smoke | done, **legacy** | Нужен повторный запуск от `knowledge` |
| C-UR4Rec | design only | Нельзя утверждать, что метод реализован или проверен |

Точный реестр: `experiments/registry.yaml`. Валидные агрегированные результаты: `results/current/manifest.json`.

## Что изменено в corrected UR4Rec

Текущая реализация исправляет ошибки, из-за которых ранние checkpoints нельзя сравнивать с paper:

- Eq. 5 передаёт user preference и knowledge истории как последовательность memory tokens, а не как один ключ cross-attention;
- mask Figure 3(c) изолирует item tokens друг от друга, сохраняя self-attention и связь с proxies;
- self-attention, FFN и norm retriever инициализируются из BERT; cross-attention остаётся случайной;
- HF causal generation корректно срезает left-padded prompt;
- cache v2 проверяет generator/model/tokens/shards и `complete`, поддерживает resumable shards и явный merge;
- статический user profile строится из train-history и не видит validation/test targets;
- ML-1M corrected-v3 использует temporal-per-user train/val/test targets и 100 случайных кандидатов;
- Frozen BERT находится в `eval()`, кодирует тексты батчами и кэширует embeddings;
- contrastive negatives исключают известные positives, а NDCG поддерживает graded labels;
- deterministic template + hashing encoder разрешены только для correctness smoke/CI.

Corrected-v3 — **честный исправленный rerun, но не paper-exact**: candidate set пока random top-100; отдельный temporal MF protocol ещё не реализован. Также локальный `DLCMReranker` — компактная DLCM-style реализация, а не импорт официального backbone.

## Карта кода

### UR4Rec

```text
configs/ur4rec/                         experiment protocols
scripts/ur4rec/run_ur4rec.py            stages and orchestration
scripts/ur4rec/run_corrected_v3.sh      current 4-GPU launch
src/data/{ml1m,amazon_books,steam,avito}.py
src/models/ur4rec/backbone.py            DLCM-style GRU reranker
src/models/ur4rec/retriever.py           proxies, self/cross attention, PIM
src/models/ur4rec/{masks,losses,text_encoder}.py
src/common/llm/                          prompts, HF generation, cache/shards
tests/test_correctness_guards.py         regression guards
```

Stages: `knowledge → merge_knowledge` при нескольких shards, затем `backbone → pretrain → joint → eval`. `finish_joint` восстанавливает подбор blend alpha после сохранённого joint checkpoint. `--stage all` рассчитан на single-process knowledge; текущий multi-GPU run использует shell orchestrator.

### Exp3RT

```text
configs/exp3rt/
scripts/exp3rt/run_exp3rt.py
scripts/exp3rt/run_paper_full.sh         sequential adapter chaining
src/models/exp3rt/{data_prep,train,test,evaluate}.py
papers/exp3rt/assets/github_repo/        upstream git submodule
```

Generic stages: `prepare`, `train`, `test`, `eval`, `all`; train tasks: `preference`, `user`, `item`, `rating`. Для paper-full использовать shell pipeline, потому что generic `all` сам по себе не гарантирует наследование adapter между стадиями.

### Avito

- `items_with_attrs.parquet`: 44 736 строк, 2 000 SERP; candidate/listing attrs и post-exposure signals.
- `users_with_history.parquet`: 2 028 контактов, 274 пользователя.
- `user_id` пересекается, `item_id` между файлами не пересекается.
- Полного текста query/фильтров нет; доступны category/location proxies.
- `serp_is_positive` константный и не является label ранжирования.
- `contacts_daily` используется как graded target только в evaluation и не должен попадать в features.

Текущая UR4Rec Avito-ветка трактует SERP как internal user и ещё не использует реальную contact history как последовательность. Поэтому C-UR4Rec нельзя честно подтвердить на этом extract без дополнительного query/history protocol.

### Остальные рабочие артефакты

| Путь | Назначение |
|---|---|
| `scripts/status_runs.sh` | GPU/process/log summary; доступ к GPU из sandbox может быть ограничен |
| `scripts/project_context.sh` | read-only snapshot Git, registry, active logs и рекомендуемых документов для нового агента |
| `scripts/check_project_docs.py` | проверка onboarding-файлов, Markdown links, YAML, registry paths и manifest |
| `scripts/snapshot_experiment_results.py` | перенос небольших метрик/логов в `results/current/` |
| `scripts/generate_eda_report.py`, `notebooks/01_eda_avito_data.ipynb` | EDA локальных Avito parquet |
| `reports/` | HTML/JSON EDA outputs |
| `teach/` | учебный материал по валидности offline ranking experiments; не часть model runtime |
| `papers/exp3rt/` | deep-reading notes и исторические reproduction snapshots; актуальные статусы брать из `docs/` |
| `results/current/` | небольшой Git snapshot проверенных результатов; старые UR4Rec файлы явно legacy |

## Канонические команды

```bash
source /home/artem-gon/miniconda3/etc/profile.d/conda.sh
conda activate diplom_avito

# Контекст текущего checkout/run для новой сессии
bash scripts/project_context.sh

# Быстрый offline correctness smoke без HF download/GPU
python scripts/ur4rec/run_ur4rec.py \
  --config configs/ur4rec/ur4rec_correctness_smoke.yaml --stage all

# Тесты regression guards
python -m unittest -q tests/test_correctness_guards.py

# Согласованность документации и experiment artifacts
python scripts/check_project_docs.py

# Текущий полный corrected run
KNOWLEDGE_GPUS=2,4,5,6 TRAIN_GPU=2 \
  bash scripts/ur4rec/run_corrected_v3.sh

# Exp3RT chained paper-full
git submodule update --init --recursive
bash scripts/exp3rt/run_paper_full.sh

# Leakage-free Avito baseline
bash scripts/exp3rt/run_avito_eval.sh
```

Корневые `scripts/run_ur4rec.py` и `scripts/run_exp3rt.py` сохранены как compatibility wrappers. В новой документации и командах использовать paper-specific пути под `scripts/ur4rec/` и `scripts/exp3rt/`.

## Что читать дальше

| Задача | Читать |
|---|---|
| Понять UR4Rec и воспроизвести | `docs/UR4Rec_code_and_reproduction.md`, затем runner и `src/models/ur4rec/` |
| Понять Exp3RT | `docs/exp3rt_reproduction.md`, `papers/exp3rt/phase2_deep_notes.md` |
| Понять задание 26.06, датасеты и литературу | `docs/task_2026-06-26_artem.md` |
| Сравнить UR4Rec / Exp3RT / LLM4Rerank | `docs/llm4rerank_vs_ur4rec_exp3rt.md` |
| Понять Avito и C-UR4Rec | `docs/avito_preferences.md`, `docs/paper_improvements_backlog.md` |
| Восстановить хронологию и презентацию | `docs/AGENT_HANDOFF.md` (archive), единственный `../Гончаров_*.pdf` |

## Правила для нового агента

1. Сначала прочитать этот файл, registry и конфиг затронутого эксперимента.
2. Перед утверждением о результате открыть сам JSON с test metrics; не переносить число из markdown без проверки.
3. Всегда маркировать `legacy`, `rating-only`, `paper-full`, `corrected` и `paper-exact` как разные протоколы.
4. Не менять код/config активного run без явного согласования: это разрушает воспроизводимость уже запущенного процесса.
5. Не коммитить новые benchmark datasets, weights, logs, HF cache и секреты. Исключение уже зафиксировано в истории: два небольших исходных Avito parquet snapshot в корне. Обычно Git содержит код, configs, docs и небольшие snapshots результатов.
6. После эксперимента обновлять registry, snapshot/manifest и документы в одном commit.
