#!/usr/bin/env python3
"""Archive one Exp3RT run (metrics, predictions, checkpoint meta, logs)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RUNS = {
    "rating_only_qwen_2026-06-17": {
        "label": "Qwen2.5-7B rating-only, 1 epoch, cutoff 1024 (beats paper on test)",
        "config": "configs/exp3rt/amazon_book_qwen.yaml",
        "checkpoint": (
            "checkpoints/exp3rt/amazon_book_qwen/"
            "amazon-book_rating_r128_alpha32_seed425"
        ),
        "logs": [
            "logs/exp3rt_rating_train.log",
            "logs/exp3rt_rating_test.log",
        ],
    },
}


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def snapshot_run(run_id: str) -> Path:
    meta = RUNS[run_id]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out = ROOT / "results" / "exp3rt_snapshots" / run_id
    out.mkdir(parents=True, exist_ok=True)

    ckpt = ROOT / meta["checkpoint"]
    for name in ("metrics.json", "predictions_test.json", "adapter_config.json"):
        _copy_if_exists(ckpt / name, out / name)
    # full merged weights stay in checkpoint dir (multi-GB); path recorded in manifest

    for log_rel in meta["logs"]:
        _copy_if_exists(ROOT / log_rel, out / "logs" / Path(log_rel).name)

    _copy_if_exists(ROOT / meta["config"], out / "config.yaml")
    _copy_if_exists(
        ROOT / "results/current/tables/exp3rt_amazon_vs_paper.md",
        out / "comparison_vs_paper.md",
    )
    _copy_if_exists(
        ROOT / "results/current/tables/exp3rt_amazon_vs_paper.json",
        out / "comparison_vs_paper.json",
    )

    manifest = {
        "run_id": run_id,
        "snapshot_time": ts,
        "label": meta["label"],
        "config": meta["config"],
        "checkpoint": meta["checkpoint"],
    }
    metrics_path = out / "metrics.json"
    if metrics_path.exists():
        manifest["test_metrics"] = json.loads(metrics_path.read_text())

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = f"""# Exp3RT snapshot: {run_id}

{meta['label']}

- Config: `{meta['config']}`
- Checkpoint source: `{meta['checkpoint']}`
- Test metrics: `metrics.json`
- Paper comparison: `comparison_vs_paper.md`

Frozen at {ts} UTC.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(f"Archived to {out}")
    return out


if __name__ == "__main__":
    for rid in RUNS:
        snapshot_run(rid)
