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

1. **Один эксперимент = одна запись** в `experiments/registry.yaml` (`status`:
   planned | running | done | failed | invalid); `invalid` сохраняет artifact
   для аудита, но запрещает использовать его как результат.
2. **Метрики только из JSON** в `checkpoints/` или `results/current/metrics/` — не из val во время train.
3. **После test** — `python scripts/snapshot_experiment_results.py` и обновить registry.
4. **Не коммитить** `.parquet` >10MB без LFS; пути в README.

### Быстрые команды

```bash
conda activate diplom_avito
cd ~/MIPT/DIPLOM/avito

# Статус прогонов
bash scripts/status_runs.sh

# Завершённый UR4Rec corrected-v3
cat checkpoints/ur4rec_ml1m_corrected_v3/metrics_test.json

# Exp3RT Amazon paper-full → test без ожидания 5 эпох
bash scripts/exp3rt/finish_paper_full_test.sh

# Исторический Exp3RT-style diagnostic; invalid for claims
bash scripts/exp3rt/run_avito_eval.sh
```

## Целевые датасеты (2 прогона)

| # | Метод | Датасет | Статус | Куда смотреть |
|---|--------|---------|--------|----------------|
| 1 | UR4Rec | MovieLens-1M | corrected-v3 **done**; base `0.214796` > UR4Rec `0.183334` NDCG@10; прежние artifacts legacy | `results/current/metrics/ur4rec_ml1m_corrected_v3.json` |
| 2 | Exp3RT | Amazon-Books | `rating_only` ✓, `paper_full` test ✓ | `checkpoints/exp3rt/amazon_book_qwen_*` |
| 3 | BLaIR + CLIP | Amazon-C4 Automotive | retrieval и multimodal control ✓; late fusion NDCG@10 `0.226064` | `results/current/metrics/amazon_c4_automotive_multimodal.json` |

## Avito (transfer)

| Метод | Скрипт | Статус |
|-------|--------|--------|
| Exp3RT-style heuristic | `run_avito_eval.sh` | **invalid**: 200/200 SERP имеют одинаковые scores после schema fix; NDCG tie/order-driven |
| local CatBoost diagnostic | `scripts/avito/run_local_catboost.py` | ✓ ensemble NDCG@10 **0.653349**; не baseline Ромы |
| local Qwen L0 diagnostic | `scripts/avito/run_llm_pointwise_diagnostic.py` | ✓ no-history/position **0.353667**; простой gate не перенёс dev gain на test |
| UR4Rec smoke | `ur4rec_avito_smoke_qwen` | legacy; rerun required |
| C-UR4Rec (план) | — | design only |

Подробнее: [avito_preferences.md](avito_preferences.md)
