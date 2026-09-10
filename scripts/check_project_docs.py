#!/usr/bin/env python3
"""Validate thesis onboarding docs, registry references, and result manifests."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment error
    raise SystemExit(
        "PyYAML is required. Activate the diplom_avito environment or install requirements.txt."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
VALID_STATUSES = {"planned", "running", "done", "failed", "invalid"}

errors: list[str] = []
warnings: list[str] = []


def error(message: str) -> None:
    errors.append(message)


def warning(message: str) -> None:
    warnings.append(message)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        error(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
        return {}


def avito_featured_cases_block(metrics: dict) -> str:
    """Render the exact report table whose values must follow current metrics."""
    lines = [
        "<!-- BEGIN AVITO FEATURED CASES -->",
        "| reason_id | history | delta_qwen | delta_gate | CatBoost TC rank | Qwen TC rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in metrics.get("featured_cases", []):
        reasons = case.get("reasons") or []
        reason_id = reasons[0].get("id", "missing") if reasons else "missing"
        values = case.get("metrics") or {}
        actual = values.get("actual_tc") or {}

        def rank(method: str) -> str:
            value = (actual.get(method) or {}).get("first_rank")
            return "—" if value is None else str(int(value))

        lines.append(
            "| {reason} | {history} | {qwen:+.6f} | {gate:+.6f} | {cb} | {llm} |".format(
                reason=reason_id,
                history=int(case.get("history_count", 0)),
                qwen=float(values.get("delta_qwen_vs_catboost_ndcg@10", 0.0)),
                gate=float(values.get("delta_gate_vs_catboost_ndcg@10", 0.0)),
                cb=rank("catboost"),
                llm=rank("qwen_l0"),
            )
        )
    lines.append("<!-- END AVITO FEATURED CASES -->")
    return "\n".join(lines)


required = [
    ROOT / "AGENTS.md",
    ROOT / ".agents/skills/diplom-context/SKILL.md",
    ROOT / ".agents/skills/avito-case-browser/SKILL.md",
    ROOT / ".agents/skills/avito-case-browser/agents/openai.yaml",
    ROOT / "docs/START_HERE.md",
    ROOT / "docs/avito_case_browser.md",
    ROOT / "experiments/registry.yaml",
    ROOT / "results/current/manifest.json",
]
for path in required:
    if not path.exists():
        error(f"missing required onboarding file: {path.relative_to(ROOT)}")

try:
    tracked_paths = set(
        subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    )
except (OSError, subprocess.CalledProcessError) as exc:
    warning(f"could not audit tracked private paths: {exc}")
    tracked_paths = set()

private_tracked_paths = {
    "items_with_attrs.parquet",
    "users_with_history.parquet",
    "results/current/metrics/avito_case_browser.json",
}
for path in sorted(private_tracked_paths & tracked_paths):
    error(f"private Avito artifact is tracked by Git: {path}")
for path in sorted(tracked_paths):
    if path.startswith(("data/", "reports/")):
        error(f"generated/private artifact is tracked by Git: {path}")

notebook_path = ROOT / "notebooks/01_eda_avito_data.ipynb"
if notebook_path.exists():
    notebook = load_json(notebook_path)
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("outputs") or cell.get("execution_count") is not None:
            error(f"Avito EDA notebook contains executed output in cell {index}")

markdown_files = [ROOT / "README.md", ROOT / "src/README.md", ROOT / "results/current/README.md"]
markdown_files.extend((ROOT / "docs").rglob("*.md"))
papers_root = ROOT / "papers/exp3rt"
if papers_root.exists():
    markdown_files.extend(
        path for path in papers_root.rglob("*.md") if "assets" not in path.parts
    )

for path in sorted(set(markdown_files)):
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8")
    for raw_target in LINK_RE.findall(text):
        target = raw_target.strip().strip("<>")
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = unquote(target.split("#", 1)[0])
        if not target or "*" in target:
            continue
        resolved = Path(target) if Path(target).is_absolute() else path.parent / target
        if not resolved.exists():
            error(f"broken Markdown link: {path.relative_to(ROOT)} -> {target}")

registry_path = ROOT / "experiments/registry.yaml"
try:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
except (OSError, yaml.YAMLError) as exc:
    error(f"invalid YAML experiments/registry.yaml: {exc}")
    registry = {}

for experiment_id, experiment in (registry.get("experiments") or {}).items():
    status = experiment.get("status")
    if status not in VALID_STATUSES:
        error(f"{experiment_id}: invalid status {status!r}")
    for key in ("config", "script"):
        value = experiment.get(key)
        if value and not (ROOT / value).exists():
            error(f"{experiment_id}: missing {key} path {value}")
    metrics = experiment.get("metrics")
    if status == "done" and metrics and not (ROOT / metrics).exists():
        warning(f"{experiment_id}: done metrics are unavailable locally: {metrics}")
    if status == "running":
        log_path = experiment.get("log")
        if not log_path or not (ROOT / log_path).exists():
            warning(f"{experiment_id}: running log is unavailable: {log_path}")

for config_path in sorted((ROOT / "configs").rglob("*.yaml")):
    try:
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        error(f"invalid YAML {config_path.relative_to(ROOT)}: {exc}")

manifest_path = ROOT / "results/current/manifest.json"
manifest = load_json(manifest_path)
entrypoint = manifest.get("project_entrypoint")
if entrypoint and not (ROOT / entrypoint).exists():
    error(f"manifest project_entrypoint does not exist: {entrypoint}")
for key, value in manifest.items():
    if not isinstance(value, dict):
        continue
    snapshot_path = value.get("path")
    if snapshot_path and not (manifest_path.parent / snapshot_path).exists():
        error(f"manifest {key}.path does not exist: {snapshot_path}")

agents_text = (ROOT / "AGENTS.md").read_text(encoding="utf-8") if (ROOT / "AGENTS.md").exists() else ""
for marker in (
    "docs/START_HERE.md",
    "$diplom-context",
    "$avito-case-browser",
    "docs/avito_case_browser.md",
    "scripts/project_context.sh",
):
    if marker not in agents_text:
        error(f"AGENTS.md does not reference required bootstrap marker: {marker}")

case_metrics_path = ROOT / "results/current/metrics/avito_case_browser.json"
case_report_path = ROOT / "docs/task_2026-09-01_execution_report.md"
if case_metrics_path.exists() and case_report_path.exists():
    case_metrics = load_json(case_metrics_path)
    report_text = case_report_path.read_text(encoding="utf-8")
    expected_block = avito_featured_cases_block(case_metrics)
    block_match = re.search(
        r"<!-- BEGIN AVITO FEATURED CASES -->.*?<!-- END AVITO FEATURED CASES -->",
        report_text,
        flags=re.DOTALL,
    )
    if block_match is None:
        error("Avito execution report lacks the machine-checked featured-case block")
    elif block_match.group(0) != expected_block:
        error(
            "Avito execution report featured-case values differ from "
            "results/current/metrics/avito_case_browser.json"
        )

for message in warnings:
    print(f"WARNING: {message}")
for message in errors:
    print(f"ERROR: {message}")

if errors:
    print(f"FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
    sys.exit(1)
print(f"OK: onboarding/docs validation passed ({len(warnings)} warning(s))")
