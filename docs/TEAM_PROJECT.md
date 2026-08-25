# Общий рабочий проект: LLM rerank (UR4Rec + Exp3RT + Avito)

> **Начинать здесь:** [START_HERE.md](START_HERE.md). Подробный аудит задания 26.06, состояния кода и валидности метрик: [task_2026-06-26_artem.md](task_2026-06-26_artem.md).

## Формат для команды

Единая структура — **реестр экспериментов + артефакты + доки**:

```text
avito/
├── AGENTS.md                    # автоматический bootstrap нового Codex
├── .agents/skills/diplom-context/ # task-aware onboarding workflow
├── experiments/registry.yaml    # ← статус всех прогонов (обновляем вместе)
├── configs/{ur4rec,exp3rt}/      # воспроизводимые конфиги
├── scripts/                     # entrypoints (run_*.sh)
├── checkpoints/<exp_id>/        # веса + metrics_test.json / metrics.json
├── results/current/             # снимок для отчёта/презентации
│   ├── metrics/
│   ├── tables/
│   └── manifest.json
├── logs/                        # master + train логи
└── docs/
    ├── START_HERE.md            # актуальная точка входа для человека/агента
    ├── TEAM_PROJECT.md          # этот файл
    └── avito_preferences.md     # преференсы на Авто + C-UR4Rec
```

### Правила

1. **Один эксперимент = одна запись** в `experiments/registry.yaml` (`status`: planned | running | done | failed).
2. **Метрики только из JSON** в `checkpoints/` или `results/current/metrics/` — не из val во время train.
3. **После test** — `python scripts/snapshot_experiment_results.py` и обновить registry.
4. **Не коммитить** `.parquet` >10MB без LFS; пути в README.

### Быстрые команды

```bash
conda activate diplom_avito
cd ~/MIPT/DIPLOM/avito

# Статус прогонов
bash scripts/status_runs.sh

# Текущий UR4Rec corrected-v3
tail -f logs/ur4rec_corrected_v3/master.log
tail -f logs/ur4rec_corrected_v3/knowledge_shard0.log

# Exp3RT Amazon paper-full → test без ожидания 5 эпох
bash scripts/exp3rt/finish_paper_full_test.sh

# Exp3RT-style на Avito (full test)
bash scripts/exp3rt/run_avito_eval.sh
```

## Целевые датасеты (2 прогона)

| # | Метод | Датасет | Статус | Куда смотреть |
|---|--------|---------|--------|----------------|
| 1 | UR4Rec | MovieLens-1M | corrected-v3 **running** (`knowledge`); прежние artifacts legacy | `logs/ur4rec_corrected_v3/`, затем `checkpoints/ur4rec_ml1m_corrected_v3/metrics_test.json` |
| 2 | Exp3RT | Amazon-Books | `rating_only` ✓, `paper_full` test ✓ | `checkpoints/exp3rt/amazon_book_qwen_*` |

## Avito (transfer)

| Метод | Скрипт | Статус |
|-------|--------|--------|
| Exp3RT-style heuristic | `run_avito_eval.sh` | ✓ leakage-free graded NDCG@10 **0.3413** vs position **0.3126** |
| UR4Rec smoke | `ur4rec_avito_smoke_qwen` | legacy; rerun required |
| C-UR4Rec (план) | — | design only |

Подробнее: [avito_preferences.md](avito_preferences.md)
