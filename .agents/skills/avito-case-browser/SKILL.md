---
name: avito-case-browser
description: Build, validate, serve, and explain the thesis project's self-contained Avito ranking case browser. Use when asked to open or rebuild the UI, refresh its Qwen insights, add item/image overlays, inspect user history, compare source/CatBoost/Qwen/Gate rankings, or explain what an interface field means; do not use for unrelated frontends.
---

# Avito Case Browser

Operate the browser as an auditable experiment artifact, not as a generic demo.

## Establish the source of truth

1. Resolve the repository root and read `docs/avito_case_browser.md` completely.
2. Read the private local `results/current/metrics/avito_case_browser.json`
   before reporting counts, metrics, build status, protocol version, or insight
   coverage. It is intentionally absent from Git; if unavailable, use only the
   aggregate `results/current/manifest.json` and state the limitation.
3. Use `apps/avito_case_browser/template.html` for visible UI behavior,
   `scripts/avito/build_case_browser.py` for build behavior, and
   `src/avito_case_browser/` for data/validation semantics.
4. If implementation changes are requested, read
   `tests/test_avito_case_browser.py` first. Keep the focused documentation,
   result snapshot, registry, and manifest consistent with the change.

Do not rebuild merely to view an existing non-empty
`reports/avito_case_browser/index.html`. Generated HTML, caches, images, and ZIP
bundles are local artifacts excluded from Git.

## Build modes

Run commands from the repository root in the `diplom_avito` environment.

### Use the current validated cache

This is the normal reproducible build and does not run the LLM:

```bash
python scripts/avito/build_case_browser.py
```

Accept the build as shareable only when both the command output and
`results/current/metrics/avito_case_browser.json` say `status: done`, every
current insight request validates, and the command prints a share-bundle path.
`needs_insights` means that the HTML may be inspected, but the final ZIP was
intentionally not produced.

### Generate or refresh Qwen insights

Only do this when the cache is missing/stale or the user requests regeneration:

```bash
python scripts/avito/build_case_browser.py
CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=0 \
  python scripts/avito/generate_case_insights.py --device cuda:0 --batch-size 1
python scripts/avito/build_case_browser.py
```

Choose an actually free physical GPU and map it through
`CUDA_VISIBLE_DEVICES`; inside the process it becomes `cuda:0`. The canonical
audit contract requires batch size 1, the pinned model revision, deterministic
flags, and matching tokenizer/code provenance. Do not weaken cache validation
to make a stale result pass. The generator defaults to local model files; ask
before downloading a missing model or incurring a new external cost.

### Add titles, descriptions, or images from an overlay

Fill the generated
`data/avito/case_browser/item_overlay_template.csv`, then rebuild with:

```bash
python scripts/avito/build_case_browser.py \
  --item-overlay /absolute/path/to/item_overlay.csv
```

Use only the supported `item_id`, title/description, `image_path`, and
`image_url` contract described in `docs/avito_case_browser.md`. Keep the
builder's path, symlink, format, frame, pixel, and size guards enabled.

## Serve and open the interface

Serve over loopback; do not publish pseudonymous user history through an
unauthenticated public tunnel without explicit permission:

```bash
python -m http.server 8765 --bind 127.0.0.1 \
  --directory reports/avito_case_browser
```

Leave this process running and verify that `http://127.0.0.1:8765/` returns
HTTP 200. In a local session, return `http://localhost:8765/` to the user.

For a VS Code Remote workspace, handle forwarding for the user when possible.
Merge, rather than overwrite, these local workspace settings in the actual
opened workspace root; `.vscode/` is ignored by this repository:

```json
{
  "remote.autoForwardPorts": true,
  "remote.autoForwardPortsSource": "process",
  "remote.restoreForwardedPorts": true,
  "remote.portsAttributes": {
    "8765": {
      "label": "Avito Case Browser",
      "onAutoForward": "openBrowser",
      "protocol": "http",
      "requireLocalPort": true
    }
  }
}
```

Restart the HTTP process after adding the settings so process-based discovery
runs again. Confirm a browser `GET /` in the server log, then give the clickable
link `http://localhost:8765/`. Do not claim forwarding succeeded solely because
the remote HTTP process started.

## Explain and use the UI

The unit of navigation is one `serp_x` (one candidate list). The left search
matches SERP, user, item, and title. Filters mean:

| Control | Meaning |
|---|---|
| `★ Отобранные` | Eight automatically selected contrastive cases, not hand-picked wins |
| `С историей` | The exported `user_id` joins to at least one contact-history row |
| `С TC-event` | At least one candidate has observed `has_tc_events=1` |
| `Qwen лучше` | Per-SERP graded NDCG@10 of Qwen L0 exceeds local CatBoost |
| `Targets ≠` | The graded-target preference and first observed TC rank point in different directions |
| `Δ Qwen − CB` | Per-SERP `NDCG@10(Qwen) - NDCG@10(CatBoost)` |

The five metric cards are per selected SERP. `NDCG@10` uses normalized
`contacts_daily` and deterministic `item_id` tie-breaking. `Первый TC` is a
separate descriptive rank of a post-exposure marker. `История` is the number
of exported contact rows and is not guaranteed to predate the SERP.

Ranking tabs reorder the same candidates:

| Tab | Meaning |
|---|---|
| `Выдача` | Exported block-aware source order |
| `CatBoost` | Leakage-guarded local diagnostic, not the missing team/Roma baseline |
| `Qwen L0` | Pointwise Qwen2.5-7B score without history or position |
| `Gate` | Rejected dev-selected blend; show it only as a diagnostic |

On a candidate card, `benchmark rel.` is graded evaluation relevance; a red
`TC event` badge is the distinct observed `has_tc_events` marker. `Подробнее`
shows all method ranks/scores, exported description, and the validated insight.
`guarded fallback` means the free LLM response failed a guard and the displayed
summary was reconstructed deterministically from evidence. `confidence low`
is deliberate, not a model probability.

History `brand`/`model_name` values are raw category-like taxonomy, not vehicle
brand/model. `Metadata coverage` is the fraction of history rows enriched with
at least a title, description, or image. Never infer purchase, intent certainty,
income, profession, age, gender, or causality from this screen.

## Verification and handoff

After code, schema, metric, or documentation changes, run:

```bash
python -m unittest -q tests/test_avito_case_browser.py
python scripts/check_project_docs.py
```

When editing the skill itself, also run the installed `skill-creator`
`quick_validate.py` if that system skill is available; its installation path is
environment-specific and must not be hard-coded into the repository workflow.

The seven-file handoff was uploaded on 2026-09-10 to the owner-controlled
Drive folder named `Диплом — Avito Case Browser — 2026-09-10`. Treat Drive as
storage, not static hosting: tell recipients to download the standalone HTML or
ZIP and open `index.html` locally. Before reporting delivery, re-list the
folder and distinguish upload success from verified access by a specific
recipient. Do not commit Drive URLs, file IDs, account addresses, or
permissions to the repository.

For the current folder, the owner explicitly confirmed on 2026-09-10 that
anyone with the link should retain `writer` access so Roma can return the
overlay. Share the link only with Roma, report that anyone holding it can
replace or delete files, and do not narrow this permission without a new
explicit owner request.

When refreshing a delivered build, replace the existing HTML, ZIP, metrics,
and instruction file in place instead of creating same-named duplicates, then
re-list the folder and verify names, sizes, and modification times. The current
external dependency is Roma's item/image overlay; do not report images as
available until that input has actually been received and rebuilt into the UI.

In the handoff, report the HTTP link, server state, build status, insight
coverage, artifact paths, and any missing score/cache/overlay input. Keep the
two target layers and all interpretation limitations explicit.
