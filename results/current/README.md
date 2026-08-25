# Experiment snapshots

Updated: **2026-08-25**. `manifest.json` is the current machine-readable summary. The UR4Rec files below are a **legacy 2026-06-23 snapshot** created before memory/mask/generation/cache fixes; do not use them as corrected reproduction results.

## Current validated results

- Exp3RT Amazon chained paper-full: `metrics/exp3rt_paper_full_test_metrics.json`
- Exp3RT-style Avito leakage-free: `metrics/exp3rt_avito_full_leakage_free.json`
- UR4Rec ML-1M corrected-v3: running; final snapshot will be added only after `metrics_test.json` exists

## UR4Rec ML-1M beat_base (legacy, completed)

- Metrics: `metrics/ur4rec_ml1m_beat_base.json`
- Training log copy: `logs/beat_baseline_train.log`
- Plots: `plots/ur4rec_*.png`
- Comparison table: `tables/comparison_metrics.md`

## Exp3RT Amazon rating (completed)

- Train log: `logs/exp3rt_rating_train.log`
- Checkpoint: `checkpoints/exp3rt/amazon_book_qwen/amazon-book_rating_r128_alpha32_seed425/`
- Plot: `plots/exp3rt_training_loss.png`

## Paper target (DLCM, ML-1M; reference only)

- Base NDCG@10 ≈ 0.359 → UR4Rec ≈ 0.678 (Appendix Table 6)
- Current corrected run: `configs/ur4rec/ur4rec_ml1m_corrected_v3.yaml` (random top-100, not paper-exact)
