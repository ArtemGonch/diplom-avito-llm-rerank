# Общий рабочий проект: LLM rerank (UR4Rec + Exp3RT + Avito)

## Формат для команды

Единая структура — **реестр экспериментов + артефакты + доки**:

```text
avito/
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
    ├── TEAM_PROJECT.md          # этот файл
    └── avito_preferences.md     # преференсы на Авто + C-UR4Rec
```

### Правила

1. **Один эксперiment = одна строка** в `experiments/registry.yaml` (`status`: planned | running | done | failed).
2. **Метрики только из JSON** в `checkpoints/` или `results/current/metrics/` — не из val во время train.
3. **После test** — `python scripts/snapshot_experiment_results.py` и обновить registry.
4. **Не коммитить** `.parquet` >10MB без LFS; пути в README.

### Быстрые команды

```bash
conda activate diplom_avito
cd ~/MIPT/DIPLOM/avito

# Статус прогонов
bash scripts/status_runs.sh

# UR4Rec ML-1M (guaranteed)
tail -f logs/guaranteed_master.log

# Exp3RT Amazon paper-full → test без ожидания 5 эпох
bash scripts/exp3rt/finish_paper_full_test.sh

# Exp3RT-style на Avito (full test)
bash scripts/exp3rt/run_avito_eval.sh
```

## Целевые датасеты (2 прогона)

| # | Метод | Датасет | Статус | Куда смотреть |
|---|--------|---------|--------|----------------|
| 1 | UR4Rec | MovieLens-1M | `beat_base` ✓, `guaranteed` 🔄 pretrain e2 | `checkpoints/ur4rec_ml1m_*` |
| 2 | Exp3RT | Amazon-Books | `rating_only` ✓, `paper_full` 🔄 merge+test | `checkpoints/exp3rt/amazon_book_qwen_*` |

## Avito (transfer)

| Метод | Скрипт | Статус |
|-------|--------|--------|
| Exp3RT-style heuristic | `run_avito_eval.sh` | ✓ NDCG@10 0.942 |
| UR4Rec smoke | `ur4rec_avito_smoke_qwen` | done |
| C-UR4Rec (план) | — | design only |

Подробнее: [avito_preferences.md](avito_preferences.md)
