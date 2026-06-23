# Source layout

Shared code used by all reproduced papers:

| Path | Contents |
|------|----------|
| `src/common/` | `metrics`, `device_util`, shared HF LLM backends (`llm/`) |
| `src/data/` | Dataset loaders: MovieLens, Amazon, Steam, Avito |

Paper-specific model code:

| Path | Paper |
|------|-------|
| `src/models/ur4rec/` | UR4Rec (COLING 2025) — backbone, retriever, losses |
| `src/models/exp3rt/` | Exp3RT (SIGIR 2025) — Qwen QLoRA train + vLLM inference |

`src/ur4rec/` — thin re-export shims so old `import ur4rec.*` still works.

Configs and scripts mirror the same split:

- `configs/ur4rec/`, `scripts/ur4rec/`
- `configs/exp3rt/`, `scripts/exp3rt/`

Checkpoints:

- `checkpoints/ur4rec/`
- `checkpoints/exp3rt/`
