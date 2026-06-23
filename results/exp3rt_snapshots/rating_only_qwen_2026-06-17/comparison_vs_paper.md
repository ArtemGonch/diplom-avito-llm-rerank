# Exp3RT Amazon-Books: test vs paper (Table 2, Total)

Generated: 2026-06-17T09:39:12.035168+00:00

| Method | RMSE | MAE | n |
|--------|------|-----|---|
| Ours (max_prob_rating) | 0.6314 | 0.3116 | 11743 |
| Ours (expected_rating) | 0.5906 | 0.3431 | 11743 |

## Paper reference

| Exp3RT (paper) | 0.6508 | 0.4297 | test |
| MF | 0.6683 | 0.4572 |
| LLMRec GPT-3.5 FS | 0.7852 | 0.4562 |

## Delta (ours expected_rating vs paper Exp3RT)

- RMSE: -0.0602 (lower is better)
- MAE: -0.0866 (lower is better)

## Protocol notes

- **Paper:** Llama-3-8B, 3-stage SFT, cutoff 1200, up to 10 epochs, vLLM test
- **Ours:** Qwen2.5-7B, rating stage only (profiles prebuilt), 1 epoch, cutoff 1024
- Metric: RMSE/MAE on full `rating_bias/test.json` (11,743 rows)