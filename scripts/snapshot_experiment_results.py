#!/usr/bin/env python3
"""Snapshot metrics, training curves, comparison tables, and plots."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "current"

PAPER_DLCM_ML1M = {
    "base": {
        "map@5": 0.253,
        "map@10": 0.265,
        "ndcg@5": 0.315,
        "ndcg@10": 0.359,
        "map@1": 0.158,
        "ndcg@1": 0.158,
    },
    "ur4rec": {
        "map@5": 0.601,
        "map@10": 0.606,
        "ndcg@5": 0.661,
        "ndcg@10": 0.678,
        "map@1": 0.348,
        "ndcg@1": 0.348,
    },
    "source": "UR4Rec paper Appendix Table 6 (DLCM, MovieLens-1M)",
}

METRIC_SOURCES = {
    "ur4rec_ml1m_beat_base": ROOT / "checkpoints/ur4rec_ml1m_beat_base/metrics_test.json",
    "ur4rec_smoke_qwen": ROOT / "checkpoints/ur4rec_smoke_qwen/metrics_test.json",
    "ur4rec_avito_smoke_qwen": ROOT / "checkpoints/ur4rec_avito_smoke_qwen/metrics_test.json",
}


def _parse_ur4rec_curves(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    curves = {"backbone": [], "pretrain": [], "joint": []}
    for m in re.finditer(
        r"\[backbone\] epoch (\d+) loss=([\d.]+) val_ndcg@10=([\d.]+)", text
    ):
        curves["backbone"].append(
            {
                "epoch": int(m.group(1)),
                "loss": float(m.group(2)),
                "val_ndcg@10": float(m.group(3)),
            }
        )
    for m in re.finditer(
        r"\[pretrain\] epoch (\d+) L_CL=([\d.]+) L_CF=([\d.]+)", text
    ):
        curves["pretrain"].append(
            {
                "epoch": int(m.group(1)),
                "L_CL": float(m.group(2)),
                "L_CF": float(m.group(3)),
            }
        )
    for m in re.finditer(r"\[joint\] epoch (\d+) loss=([\d.]+)", text):
        curves["joint"].append(
            {"epoch": int(m.group(1)), "loss": float(m.group(2))}
        )
    return curves


def _parse_exp3rt_curves(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    losses = [
        float(m.group(1))
        for m in re.finditer(r"\{'loss': ([\d.]+), 'grad_norm':", text)
    ]
    eval_match = re.search(
        r"\{'eval_loss': ([\d.]+), 'eval_rmse': ([\d.]+), 'eval_mae': ([\d.]+)",
        text,
    )
    train_match = re.search(
        r"\{'train_runtime': ([\d.]+).*'train_loss': ([\d.]+), 'epoch': ([\d.]+)\}",
        text,
    )
    summary = {}
    if eval_match:
        summary["eval_loss"] = float(eval_match.group(1))
        summary["eval_rmse"] = float(eval_match.group(2))
        summary["eval_mae"] = float(eval_match.group(3))
    if train_match:
        summary["train_runtime_sec"] = float(train_match.group(1))
        summary["train_loss"] = float(train_match.group(2))
        summary["epochs"] = float(train_match.group(3))
        summary["train_hours"] = summary["train_runtime_sec"] / 3600
    return {"train_loss_steps": losses, "summary": summary}


def _load_metrics() -> dict:
    out = {}
    for name, path in METRIC_SOURCES.items():
        if path.exists():
            out[name] = json.loads(path.read_text())
    return out


def _build_comparison_rows(metrics: dict) -> list[dict]:
    rows = []
    beat = metrics.get("ur4rec_ml1m_beat_base", {})
    for model_key, label in [("base", "DLCM base (ours)"), ("ur4rec", "UR4Rec (ours)")]:
        ours = beat.get(model_key, {})
        paper = PAPER_DLCM_ML1M[model_key]
        for metric in ["ndcg@1", "ndcg@5", "ndcg@10", "map@1", "map@5", "map@10"]:
            rows.append(
                {
                    "model": label,
                    "metric": metric,
                    "ours_beat_base": ours.get(metric),
                    "paper": paper.get(metric),
                    "delta": (
                        ours.get(metric) - paper.get(metric)
                        if ours.get(metric) is not None
                        else None
                    ),
                }
            )
    return rows


def _save_table(rows: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "comparison_metrics.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("model,metric,ours_beat_base,paper,delta\n")
        for r in rows:
            f.write(
                f"{r['model']},{r['metric']},"
                f"{r['ours_beat_base']},{r['paper']},{r['delta']}\n"
            )

    md_path = out_dir / "comparison_metrics.md"
    lines = [
        "# UR4Rec MovieLens-1M: ours vs paper (DLCM backbone)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Model | Metric | Ours (beat_base) | Paper | Δ |",
        "|-------|--------|------------------|-------|---|",
    ]
    for r in rows:
        o = r["ours_beat_base"]
        p = r["paper"]
        d = r["delta"]
        o_s = f"{o:.4f}" if o is not None else "—"
        d_s = f"{d:+.4f}" if d is not None else "—"
        lines.append(
            f"| {r['model']} | {r['metric']} | {o_s} | {p:.4f} | {d_s} |"
        )
    lines.extend(
        [
            "",
            "## Config notes",
            "",
            "- **Ours:** `configs/ur4rec/ur4rec_ml1m_beat_base.yaml` — 50 candidates, Qwen 128 tok, 600 test users",
            "- **Paper:** 100 candidates, Llama2-Chat 512 tok, full test split (Appendix Table 6)",
            "",
            "## Other runs (reference)",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_ur4rec_training(curves: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    bb = curves["backbone"]
    if bb:
        ep = [x["epoch"] for x in bb]
        axes[0].plot(ep, [x["loss"] for x in bb], "o-", label="train loss")
        ax2 = axes[0].twinx()
        ax2.plot(ep, [x["val_ndcg@10"] for x in bb], "s--", color="C1", label="val NDCG@10")
        axes[0].set_title("Backbone (DLCM)")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        ax2.set_ylabel("NDCG@10")

    pt = curves["pretrain"]
    if pt:
        ep = [x["epoch"] for x in pt]
        axes[1].plot(ep, [x["L_CL"] for x in pt], "o-", label="L_CL")
        axes[1].plot(ep, [x["L_CF"] for x in pt], "s-", label="L_CF")
        axes[1].set_title("Retriever pretrain")
        axes[1].set_xlabel("epoch")
        axes[1].legend()

    jt = curves["joint"]
    if jt:
        ep = [x["epoch"] for x in jt]
        axes[2].plot(ep, [x["loss"] for x in jt], "o-")
        axes[2].set_title("Joint training")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("loss")

    fig.suptitle("UR4Rec ML-1M beat_base — training curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_metrics_comparison(metrics: dict, out_path: Path) -> None:
    beat = metrics.get("ur4rec_ml1m_beat_base", {})
    if not beat:
        return

    keys = ["ndcg@5", "ndcg@10", "map@5", "map@10"]
    x = np.arange(len(keys))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_key, label, color) in enumerate(
        [
            ("base", "Base (ours)", "C0"),
            ("ur4rec", "UR4Rec (ours)", "C1"),
        ]
    ):
        vals = [beat[model_key].get(k, 0) for k in keys]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    paper_base = [PAPER_DLCM_ML1M["base"][k] for k in keys]
    paper_ur4 = [PAPER_DLCM_ML1M["ur4rec"][k] for k in keys]
    ax.bar(x + 2 * width, paper_base, width, label="Paper base", color="C2", alpha=0.6)
    ax.bar(x + 3 * width, paper_ur4, width, label="Paper UR4Rec", color="C3", alpha=0.6)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(keys)
    ax.set_ylabel("score")
    ax.set_title("MovieLens-1M DLCM: metrics vs UR4Rec paper")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 0.75)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_exp3rt_loss(curves: dict, out_path: Path) -> None:
    losses = curves.get("train_loss_steps", [])
    if not losses:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, linewidth=0.8)
    ax.set_xlabel("logged step (every ~10 steps)")
    ax.set_ylabel("train loss")
    summary = curves.get("summary", {})
    title = "Exp3RT rating stage — train loss"
    if summary:
        title += (
            f" | final loss={summary.get('train_loss', '?'):.3f}, "
            f"eval RMSE={summary.get('eval_rmse', '?'):.3f}"
        )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    dirs = {
        "root": OUT,
        "metrics": OUT / "metrics",
        "curves": OUT / "training_curves",
        "tables": OUT / "tables",
        "plots": OUT / "plots",
        "logs": OUT / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    metrics = _load_metrics()
    for name, data in metrics.items():
        (dirs["metrics"] / f"{name}.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    ur4rec_log = ROOT / "logs/beat_baseline_train.log"
    exp3rt_log = ROOT / "logs/exp3rt_rating_train.log"
    ur4rec_curves = _parse_ur4rec_curves(ur4rec_log) if ur4rec_log.exists() else {}
    exp3rt_curves = _parse_exp3rt_curves(exp3rt_log) if exp3rt_log.exists() else {}

    (dirs["curves"] / "ur4rec_beat_base.json").write_text(
        json.dumps(ur4rec_curves, indent=2), encoding="utf-8"
    )
    (dirs["curves"] / "exp3rt_rating.json").write_text(
        json.dumps(exp3rt_curves, indent=2), encoding="utf-8"
    )

    rows = _build_comparison_rows(metrics)
    _save_table(rows, dirs["tables"])

    _plot_ur4rec_training(ur4rec_curves, dirs["plots"] / "ur4rec_training_curves.png")
    _plot_metrics_comparison(metrics, dirs["plots"] / "ur4rec_metrics_vs_paper.png")
    _plot_exp3rt_loss(exp3rt_curves, dirs["plots"] / "exp3rt_training_loss.png")

    for log_name in [
        "beat_baseline_train.log",
        "exp3rt_rating_train.log",
        "beat_baseline_master.log",
    ]:
        src = ROOT / "logs" / log_name
        if src.exists():
            shutil.copy2(src, dirs["logs"] / log_name)

    manifest = {
        "snapshot_time": ts,
        "metrics_files": list(metrics.keys()),
        "paper_reference": PAPER_DLCM_ML1M,
        "exp3rt_summary": exp3rt_curves.get("summary", {}),
        "plots": [
            "plots/ur4rec_training_curves.png",
            "plots/ur4rec_metrics_vs_paper.png",
            "plots/exp3rt_training_loss.png",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = f"""# Experiment snapshot ({ts})

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
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"Snapshot saved to {OUT}")


if __name__ == "__main__":
    main()
