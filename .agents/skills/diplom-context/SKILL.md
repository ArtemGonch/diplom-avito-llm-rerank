---
name: diplom-context
description: Load and verify the MIPT thesis context for requests about UR4Rec, Exp3RT, Avito reranking, experiments, metrics, papers, presentation, or project code. Use for onboarding and any diploma-related task in this repository; do not use for unrelated generic work.
---

# Diplom Context

Build an evidence-backed view of the project without loading every historical artifact by default.

## Bootstrap

1. Resolve the repository root and read `docs/START_HERE.md` completely.
2. Run `bash scripts/project_context.sh` and inspect `experiments/registry.yaml`.
3. Identify the request mode below and read only its focused sources.
4. Before reporting a number or status, open the underlying config, test metrics JSON, or live log that supports it.

If a referenced local artifact is unavailable because datasets, logs, or checkpoints are intentionally ignored by Git, say which evidence is missing. Do not silently replace it with an old Markdown value.

## Route by task

- **UR4Rec implementation or experiment:** read `docs/UR4Rec_code_and_reproduction.md`, the selected `configs/ur4rec/*.yaml`, `scripts/ur4rec/run_ur4rec.py`, and only the affected files under `src/models/ur4rec/`, `src/common/llm/`, or `src/data/`. Read `tests/test_correctness_guards.py` before changing an invariant.
- **Exp3RT:** read `docs/exp3rt_reproduction.md`, the selected `configs/exp3rt/*.yaml`, the relevant runner under `scripts/exp3rt/`, and the affected code under `src/models/exp3rt/`. Use `papers/exp3rt/` for paper analysis; treat reproduction notes marked archive as history.
- **Avito or C-UR4Rec:** read `docs/avito_preferences.md` and the Avito/code-audit sections of `docs/task_2026-06-26_artem.md`. Verify that evaluation targets or post-exposure signals are not model features.
- **Literature, thesis text, or presentation:** read `docs/task_2026-06-26_artem.md`, `docs/llm4rerank_vs_ur4rec_exp3rt.md`, and the relevant paper notes. Use `docs/AGENT_HANDOFF.md` only when chronology is needed. Inspect the thesis PDF when claims must match the presentation.
- **Experiment status or results:** inspect registry, `results/current/manifest.json`, the actual metrics JSON, and the current log. Distinguish `running`, `done`, `failed`, and `legacy` explicitly.

## Evidence and change rules

- Code plus selected YAML defines behavior; registry defines lifecycle status; test JSON defines numbers.
- Do not call corrected-v3 paper-exact: it currently uses temporal-per-user targets and random top-100 candidates.
- Do not cite legacy UR4Rec metrics as corrected results or Avito `0.9417` as valid.
- Do not claim C-UR4Rec is implemented or validated without matching code and test artifacts.
- Do not edit active-run code/config without explicit approval. Read-only monitoring is safe.
- When implementation changes affect claims or reproduction, update focused docs, registry, and manifest in the same change.

## Verification

Run checks proportional to the task. For onboarding or documentation changes, use:

```bash
python scripts/check_project_docs.py
python -m unittest -q tests/test_correctness_guards.py
```

For UR4Rec plumbing changes, also run the offline correctness smoke from `docs/UR4Rec_code_and_reproduction.md`. Do not interpret smoke quality as a thesis result.

End an onboarding pass with a compact readiness report covering: thesis objective, model/data flow, current validated experiments, active runs, invalid/legacy claims, open blockers, and which sources were inspected.
