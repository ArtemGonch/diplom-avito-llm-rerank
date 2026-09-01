# UR4Rec: реализация и воспроизведение

Обновлено: **2026-09-01**. Основной эксперимент `ur4rec_ml1m_corrected_v3` завершён 2026-08-26: knowledge cache объединён (`3 883` items / `5 802` users), пройдены `backbone → pretrain → joint → eval`. Это исправленный rerun, **не paper-exact reproduction**.

Статья: Zhang et al., [*Enhancing Reranking for Recommendation with LLMs through User Preference Retrieval*](https://aclanthology.org/2025.coling-main.45/) (COLING 2025). Официального репозитория статьи не найдено, поэтому код в этом проекте — самостоятельная реализация по paper с компактным DLCM-style backbone.

## Что реализовано

```text
offline Qwen knowledge
  ├─ user preference text
  └─ knowledge text для items истории
          │
          ▼
Frozen BERT mean pooling → memory [user + history items]
          │
candidate ID embeddings + K learned proxies
          │
          ▼
6-layer retriever: self-attention → cross-attention(memory) → FFN
          │
          ▼
per-candidate augmentation → DLCM-style GRU listwise reranker
```

Код:

| Компонент | Путь |
|---|---|
| Runner и stages | `scripts/ur4rec/run_ur4rec.py` |
| Текущий 4-GPU orchestrator | `scripts/ur4rec/run_corrected_v3.sh` |
| Resume после готовых knowledge shards | `scripts/ur4rec/run_corrected_v3_resume.sh` |
| Dataset protocols | `src/data/ml1m.py`, `amazon_books.py`, `steam.py`, `avito.py` |
| DLCM-style backbone | `src/models/ur4rec/backbone.py` |
| Retriever | `src/models/ur4rec/retriever.py` |
| Figure 3 masks | `src/models/ur4rec/masks.py` |
| Losses | `src/models/ur4rec/losses.py` |
| Frozen encoder | `src/models/ur4rec/text_encoder.py` |
| LLM prompts/generation/cache | `src/common/llm/` |
| Correctness tests | `tests/test_correctness_guards.py` |

`src/ur4rec/` оставлен только как re-export compatibility layer. Корневой `scripts/run_ur4rec.py` — compatibility wrapper; новый код и команды должны использовать `scripts/ur4rec/run_ur4rec.py`.

## Stages и артефакты

| Stage | Что делает | Выход corrected-v3 |
|---|---|---|
| `knowledge` | Qwen генерирует item knowledge и user preference; один worker пишет свой shard | `data/movielens-1m/knowledge_qwen25_corrected_v3/shards/shard_N/` |
| `merge_knowledge` | проверяет полноту shards и собирает cache v2 | `users.json`, `items.json`, `meta.json` |
| `backbone` | обучает base DLCM-style reranker с listwise CE | `checkpoints/ur4rec_ml1m_corrected_v3/backbone.pt` |
| `pretrain` | обучает retriever на `L_CL + α·L_CF` | `retriever_pretrain.pt` |
| `joint` | совместно обучает backbone и retriever, early stopping по val NDCG@10 | `ur4rec_joint.pt`, `ur4rec_joint_meta.pt` |
| `finish_joint` | восстанавливает только val-подбор blend alpha после сохранённого joint checkpoint | `ur4rec_joint_meta.pt` |
| `eval` | test base, pure UR4Rec и val-selected blend | `metrics_test.json` |

`--stage all` выполняет `knowledge → backbone → pretrain → joint → eval` в одном процессе. При sharded generation сначала запускать несколько `knowledge` workers, затем `merge_knowledge`; это делает `run_corrected_v3.sh`.

## Corrected-v3 protocol

Конфиг: `configs/ur4rec/ur4rec_ml1m_corrected_v3.yaml`.

| Параметр | Значение |
|---|---|
| Dataset | MovieLens-1M, positive rating threshold из loader, 5-core |
| Split | один chronological train/val/test target на каждого подходящего пользователя |
| History | последние 10 событий строго до соответствующего target |
| Candidates | 100: target + 99 unseen random negatives |
| LLM | Qwen2.5-7B-Instruct, 4-bit, greedy, max 512 new tokens, batch 4 |
| Knowledge workers | 4 deterministic hash shards |
| Encoder | frozen `bert-base-uncased`, max length 512, batch 64 |
| Retriever | 8 proxies, 6 layers, 12 heads, hidden 768 |
| Backbone | local DLCM-style GRU, hidden 768, 5 epochs |
| Retriever pretrain | 3 epochs, 10 negatives, lr `1e-4` |
| Joint | до 8 epochs, patience 3, pure UR4Rec `blend_alpha_grid: [1.0]` |

Temporal-per-user split нужен потому, что user embeddings на validation/test должны быть обучены. В старом user-wise split validation/test users не встречались в train и их embeddings оставались случайными.

Random top-100 выбран как честный первоначальный corrected protocol. Он ещё не повторяет paper candidate generator: для temporal MF/BPR top-100 требуется обучать candidate model только на событиях до temporal cutoff. Legacy `mf_topk` работает с user-wise split и не превращает текущий run в paper-exact.

## Критические correctness invariants

### Memory и attention

- BERT возвращает отдельные векторы `[user preference, item knowledge 1, …, item knowledge H]` формы `[H+1, 768]`.
- Cross-attention получает несколько key/value tokens. Legacy flatten в один token делал softmax attention константой `1` и фактически отключал retrieval.
- В Figure 3(c) proxies видят весь список, а разные item tokens не видят друг друга; diagonal item self-attention разрешён.
- Self-attention projections, FFN и layer norms копируются из BERT; cross-attention инициализируется случайно.

### Knowledge generation и cache v2

- HF decoder работает с left padding, а generated continuation срезается по общей padded input width.
- `meta.json` фиксирует cache version, generator, model, `max_new_tokens`, число shards и `complete`.
- Незавершённый cache не считается hit. Каждый shard можно продолжить после остановки; root становится complete только после успешного merge.
- User profile для temporal protocol строится по самой ранней train sample history, поэтому статический profile не видит val/test target.
- `KnowledgeStore` и frozen BERT используют in-memory caches, а BERT texts заранее прогреваются батчами.

### Losses и evaluation

- Backbone и joint используют listwise softmax cross-entropy с одним relevant item на ML-1M list.
- Retriever pretrain использует InfoNCE по max cosine similarity proxies и preference-item matching BCE.
- Negatives не включают известные positive items пользователя.
- Test JSON содержит base, pure UR4Rec (`alpha=1`) и selected blend. В corrected-v3 grid содержит только `1.0`, чтобы fallback на base не выдавался за улучшение UR4Rec.
- Общий NDCG сохраняет graded labels; для ML-1M labels бинарные.

## Запуск

### Correctness smoke без GPU/Hugging Face

```bash
conda activate diplom_avito
python scripts/ur4rec/run_ur4rec.py \
  --config configs/ur4rec/ur4rec_correctness_smoke.yaml \
  --stage all
python -m unittest -q tests/test_correctness_guards.py
```

Smoke использует deterministic template generator и hashing encoder. Это тест plumbing/invariants, не эксперимент для дипломной таблицы.

### Corrected-v3

```bash
conda activate diplom_avito
KNOWLEDGE_GPUS=2,4,5,6 TRAIN_GPU=2 \
  bash scripts/ur4rec/run_corrected_v3.sh
```

На запуске 2026-08-25 `knowledge` распределён по физическим GPU `2,4,5,6`.
После завершения shards исходный orchestrator не продолжил merge; 2026-08-26
cache успешно объединён и pipeline возобновлён через resume-скрипт на GPU `2`.
Список свободных GPU всегда проверять заново: он не является свойством репозитория.

Логи:

```bash
tail -f logs/ur4rec_corrected_v3/master.log
tail -f logs/ur4rec_corrected_v3/knowledge_shard0.log
tail -f logs/ur4rec_corrected_v3/merge.log
tail -f logs/ur4rec_corrected_v3/train.log
```

Если четыре knowledge shards завершены, но orchestrator не дошёл до merge,
продолжить без повторной Qwen-генерации можно командой:

```bash
TRAIN_GPU=2 bash scripts/ur4rec/run_corrected_v3_resume.sh
```

Текущая стадия и PID пишутся в `logs/ur4rec_corrected_v3/resume.status` и
`resume.pid`. Resume идемпотентно пропускает уже существующие checkpoints,
использует `finish_joint`, если joint checkpoint сохранён без metadata, и перед
успешным завершением проверяет структуру `metrics_test.json`.

На стадии `knowledge` `master.log` содержит orchestration events, а живой progress bar находится в `knowledge_shard*.log`. Сообщение Transformers о неиспользуемых generation flags при greedy decoding не означает падение; ориентироваться на рост progress и exit status shard.

Финальный результат находится в:

```text
checkpoints/ur4rec_ml1m_corrected_v3/metrics_test.json
```

Проверенный snapshot: `results/current/metrics/ur4rec_ml1m_corrected_v3.json`.

| Test method | NDCG@5 | NDCG@10 | MAP@10 |
|---|---:|---:|---:|
| local DLCM-style base | **0.175297** | **0.214796** | **0.162291** |
| pure UR4Rec, `alpha=1.0` | 0.143121 | 0.183334 | 0.134540 |

Pure UR4Rec хуже base на `0.031462` NDCG@10, или `−14.65%` относительно
base. Это валидный negative result на текущем protocol. Он не противоречит
paper напрямую: paper DLCM Table 6 использует другой candidate-generation
protocol и сообщает NDCG@10 `0.315 → 0.661` (`0.359 → 0.678` — NDCG@20, а не
NDCG@10).

## Другие конфиги

| Config | Назначение | Статус |
|---|---|---|
| `ur4rec_correctness_smoke.yaml` | быстрый offline invariant test | current, не для quality claim |
| `ur4rec_ml1m_corrected_v3.yaml` | основной corrected ML-1M run | current, done; negative result |
| `ur4rec_ml1m_beat_base.yaml` | 50-candidate отладочный run | legacy |
| `ur4rec_ml1m_guaranteed.yaml` | blend мог выбрать pure base | legacy |
| `ur4rec_ml1m_full.yaml`, `paper_v2`, `paper_exact` | прежние попытки приблизить paper protocol | названия не гарантируют paper exactness |
| `ur4rec_*_smoke_qwen.yaml` | Qwen smoke на ML-1M/Amazon/Steam/Avito | старые caches/metrics невалидны после fixes |

## Датасеты

Loader поддерживает MovieLens-1M, Amazon Books, Steam и Avito. Download helper:

```bash
python scripts/ur4rec/download_ur4rec_datasets.py --datasets all
```

Большие datasets, generated knowledge и weights не входят в git. Пути задаются YAML-конфигами.

Avito имеет отдельное ограничение: текущий loader строит sample по SERP и
internal user id, а `users_with_history.parquet` не превращён в leakage-free
temporal history для UR4Rec. Дополнительный schema audit показал, что history
`brand`/`model_name` — sparse category-like значения с нулевым vocabulary
overlap с listing brand/model; price и остальные Auto attrs отсутствуют.
Результаты старого Avito smoke — legacy; актуальные controls и границы
персонализации описаны в `docs/avito_preferences.md`.

## Границы утверждений

Можно утверждать:

- реализован полный исследовательский pipeline offline knowledge → retriever pretrain → joint listwise rerank → test;
- correctness bugs memory/mask/generation/cache/split исправлены и покрыты regression tests;
- corrected-v3 использует temporal targets и честный pure-UR4Rec output;
- на этом protocol base NDCG@10 `0.214796`, UR4Rec `0.183334`.

Пока нельзя утверждать:

- что локальная реализация численно воспроизводит Table 6 paper;
- что corrected-v3 воспроизводит paper Table 6 или превосходит backbone;
- что random candidate protocol paper-exact;
- что C-UR4Rec реализован или подтверждён на Avito.

Для актуального статуса и списка валидных чисел начинать с `docs/START_HERE.md` и `experiments/registry.yaml`.
