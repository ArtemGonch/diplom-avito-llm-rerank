# Experiment snapshot (2026-06-23T150946Z)

## UR4Rec ML-1M beat_base (completed)

- Metrics: `metrics/ur4rec_ml1m_beat_base.json`
- Training log copy: `logs/beat_baseline_train.log`
- Plots: `plots/ur4rec_*.png`
- Comparison table: `tables/comparison_metrics.md`

## Exp3RT Amazon rating (completed)

- Train log: `logs/exp3rt_rating_train.log`
- Checkpoint: `checkpoints/exp3rt/amazon_book_qwen/amazon-book_rating_r128_alpha32_seed425/`
- Plot: `plots/exp3rt_training_loss.png`

## Paper target (DLCM, ML-1M)

- Base NDCG@10 ≈ 0.359 → UR4Rec ≈ 0.678 (Appendix Table 6)
- Full repro run: `configs/ur4rec/ur4rec_ml1m_full.yaml`
