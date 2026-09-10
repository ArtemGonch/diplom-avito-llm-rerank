# Diploma repository instructions

This repository contains Artem Goncharov's MIPT thesis project on LLM-assisted reranking: UR4Rec, Exp3RT, Avito Auto experiments, literature notes, and presentation artifacts.

For every thesis-related request:

1. Read `docs/START_HERE.md` completely before reasoning about the project.
2. Run `bash scripts/project_context.sh` for the current Git/experiment/run state.
3. Read `experiments/registry.yaml` and the exact config, metrics JSON, and focused document relevant to the request.
4. Use the repository skill `$diplom-context` when available; it defines task-specific reading routes and evidence rules.
5. For building, serving, sharing, or interpreting the Avito ranking interface, use the repository skill `$avito-case-browser` and read `docs/avito_case_browser.md`.

Treat code plus the selected YAML config as runtime truth, the registry as experiment-status truth, and test metrics JSON as numerical truth. Never present validation metrics, legacy snapshots, or Markdown summaries as current test results without checking the underlying artifact.

Keep these protocol labels distinct: `legacy`, `rating-only`, `paper-full`, `corrected`, and `paper-exact`. Corrected UR4Rec does not imply paper-exact reproduction. C-UR4Rec is a proposed contribution until code and evaluated artifacts demonstrate otherwise.

Do not change code or configs used by an active run unless the user explicitly approves invalidating or restarting that run. Datasets, row-level metrics or identifiers, executed notebook outputs, generated knowledge, weights, logs, model caches, credentials, and secrets stay out of Git. Avito data is stored only on the private VM and in the designated Google Drive folder; there are no dataset exceptions.

Read `docs/AGENT_HANDOFF.md` only for historical chronology or literature context. Its old paths, infrastructure notes, and metrics are not current state.

When changing code or experiment claims, update the focused documentation, registry, and result manifest together. Run the smallest relevant tests plus `python scripts/check_project_docs.py` before committing.
