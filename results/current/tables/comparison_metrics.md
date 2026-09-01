# UR4Rec MovieLens-1M: ours vs paper (DLCM backbone)

Corrected paper-column mapping: 2026-09-01. Local values remain the legacy
`beat_base` run and are not the corrected-v3 result.

| Model | Metric | Ours (beat_base) | Paper | Δ |
|-------|--------|------------------|-------|---|
| DLCM base (ours) | ndcg@1 | 0.1067 | 0.1580 | -0.0513 |
| DLCM base (ours) | ndcg@5 | 0.2219 | 0.2710 | -0.0491 |
| DLCM base (ours) | ndcg@10 | 0.2853 | 0.3150 | -0.0297 |
| DLCM base (ours) | map@1 | 0.1067 | 0.1580 | -0.0513 |
| DLCM base (ours) | map@5 | 0.1839 | 0.2350 | -0.0511 |
| DLCM base (ours) | map@10 | 0.2100 | 0.2530 | -0.0430 |
| UR4Rec (ours) | ndcg@1 | 0.1217 | 0.4840 | -0.3623 |
| UR4Rec (ours) | ndcg@5 | 0.2334 | 0.6310 | -0.3976 |
| UR4Rec (ours) | ndcg@10 | 0.2996 | 0.6610 | -0.3614 |
| UR4Rec (ours) | map@1 | 0.1217 | 0.4840 | -0.3623 |
| UR4Rec (ours) | map@5 | 0.1979 | 0.5880 | -0.3901 |
| UR4Rec (ours) | map@10 | 0.2248 | 0.6010 | -0.3762 |

## Config notes

- **Ours:** `configs/ur4rec/ur4rec_ml1m_beat_base.yaml` — 50 candidates, Qwen 128 tok, 600 test users
- **Paper:** 100 candidates, Llama2-Chat 512 tok, full test split (Table 2 for @1/@5; Appendix Table 6 for @10)

## Other runs (reference)
