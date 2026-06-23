# UR4Rec MovieLens-1M: ours vs paper (DLCM backbone)

Generated: 2026-06-17T09:39:11.309453+00:00

| Model | Metric | Ours (beat_base) | Paper | Δ |
|-------|--------|------------------|-------|---|
| DLCM base (ours) | ndcg@1 | 0.1067 | 0.1580 | -0.0513 |
| DLCM base (ours) | ndcg@5 | 0.2219 | 0.3150 | -0.0931 |
| DLCM base (ours) | ndcg@10 | 0.2853 | 0.3590 | -0.0737 |
| DLCM base (ours) | map@1 | 0.1067 | 0.1580 | -0.0513 |
| DLCM base (ours) | map@5 | 0.1839 | 0.2530 | -0.0691 |
| DLCM base (ours) | map@10 | 0.2100 | 0.2650 | -0.0550 |
| UR4Rec (ours) | ndcg@1 | 0.1217 | 0.3480 | -0.2263 |
| UR4Rec (ours) | ndcg@5 | 0.2334 | 0.6610 | -0.4276 |
| UR4Rec (ours) | ndcg@10 | 0.2996 | 0.6780 | -0.3784 |
| UR4Rec (ours) | map@1 | 0.1217 | 0.3480 | -0.2263 |
| UR4Rec (ours) | map@5 | 0.1979 | 0.6010 | -0.4031 |
| UR4Rec (ours) | map@10 | 0.2248 | 0.6060 | -0.3812 |

## Config notes

- **Ours:** `configs/ur4rec/ur4rec_ml1m_beat_base.yaml` — 50 candidates, Qwen 128 tok, 600 test users
- **Paper:** 100 candidates, Llama2-Chat 512 tok, full test split (Appendix Table 6)

## Other runs (reference)
