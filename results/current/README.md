# Experiment snapshots

Updated: **2026-09-01**. `manifest.json` is the current machine-readable summary. The `beat_base` UR4Rec files below are a **legacy 2026-06-23 snapshot** created before memory/mask/generation/cache fixes; do not use them as corrected reproduction results.

## Current validated results

- Exp3RT Amazon chained paper-full: `metrics/exp3rt_paper_full_test_metrics.json`
- UR4Rec ML-1M corrected-v3: `metrics/ur4rec_ml1m_corrected_v3.json` — base NDCG@10 `0.214796`, UR4Rec `0.183334`
- Amazon-C4 Automotive: `metrics/amazon_c4_automotive_retrieval.json`, `metrics/amazon_c4_automotive_image_coverage.json`, `metrics/amazon_c4_automotive_multimodal.json`
- Avito local CatBoost diagnostic: `metrics/avito_local_catboost_diagnostic.json` — not Roma/team A1
- Avito local no-history Qwen L0/gate: `metrics/avito_local_llm_diagnostic.json` — diagnostic only; gate rejected on test

## Invalid diagnostics retained for audit

- Exp3RT-style Avito: `metrics/exp3rt_avito_full_leakage_free.json` — direct
  target leakage removed, but history schema is incompatible and 200/200 SERP
  have constant scores; `valid_for_claims=false`.

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

- Base NDCG@10 ≈ 0.315 → UR4Rec ≈ 0.661 (Appendix Table 6); `0.359 → 0.678` относится к NDCG@20
- Current corrected run: `configs/ur4rec/ur4rec_ml1m_corrected_v3.yaml` (random top-100, not paper-exact)
