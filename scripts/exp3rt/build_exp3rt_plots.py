#!/usr/bin/env python3
"""Build Exp3RT training curves and test metrics vs SIGIR 2025 paper."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "current"
PLOTS = OUT / "plots"
CURVES = OUT / "training_curves"

PAPER_AMAZON_TOTAL = {"rmse": 0.6508, "mae": 0.4297}
PAPER_BASELINES = {
    "MF": {"rmse": 0.6683, "mae": 0.4572},
    "LLMRec GPT-3.5": {"rmse": 0.7852, "mae": 0.4562},
    "LLMRec GPT-4": {"rmse": 0.8386, "mae": 0.5004},
}

RATING_ONLY_METRICS = (
    ROOT / "results/exp3rt_snapshots/rating_only_qwen_2026-06-17/comparison_vs_paper.json"
)
PAPER_FULL_METRICS = OUT / "metrics/exp3rt_paper_full_test_metrics.json"
RATING_ONLY_LOG = ROOT / "logs/exp3rt_rating_train.log"
PAPER_FULL_LOG = ROOT / "logs/exp3rt_paper_full_train.log"

STAGE_ORDER = ["preference", "user", "item", "rating"]
STAGE_TITLES = {
    "preference": "Stage 1: preference",
    "user": "Stage 2: user profile",
    "item": "Stage 3: item profile",
    "rating": "Stage 4: rating (early stop @ ep 2)",
}


def _parse_single_stage_log(text: str) -> dict:
    losses = [
        float(m.group(1))
        for m in re.finditer(r"\{'loss': ([\d.]+), 'grad_norm':", text)
    ]
    evals = []
    for m in re.finditer(
        r"\{'eval_loss': ([\d.]+)(?:, 'eval_rmse': ([\d.]+))?"
        r"(?:, 'eval_mae': ([\d.]+))?[^}]*'epoch': ([\d.]+)\}",
        text,
    ):
        evals.append(
            {
                "eval_loss": float(m.group(1)),
                "eval_rmse": float(m.group(2)) if m.group(2) else None,
                "eval_mae": float(m.group(3)) if m.group(3) else None,
                "epoch": float(m.group(4)),
            }
        )
    train = re.search(
        r"\{'train_runtime': ([\d.]+)[^}]*'train_loss': ([\d.]+), 'epoch': ([\d.]+)\}",
        text,
    )
    summary = {}
    if train:
        summary = {
            "train_runtime_sec": float(train.group(1)),
            "train_loss": float(train.group(2)),
            "epochs": float(train.group(3)),
            "train_hours": float(train.group(1)) / 3600,
        }
    if evals:
        last = evals[-1]
        summary.setdefault("eval_loss", last["eval_loss"])
        if last.get("eval_rmse") is not None:
            summary["eval_rmse"] = last["eval_rmse"]
            summary["eval_mae"] = last["eval_mae"]
    return {"train_loss_steps": losses, "evals": evals, "summary": summary}


def _parse_paper_full_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    starts = [
        (m.start(), m.group(1))
        for m in re.finditer(r"Exp3RT train stage=(\w+)", text)
    ]
    segments: list[tuple[str, str]] = []
    for i, (pos, stage) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        segments.append((stage, text[pos:end]))

    best: dict[str, dict] = {}
    for stage, seg in segments:
        parsed = _parse_single_stage_log(seg)
        n_loss = len(parsed["train_loss_steps"])
        if n_loss >= best.get(stage, {}).get("_n_loss", -1):
            parsed["_n_loss"] = n_loss
            best[stage] = parsed

    return {stage: best[stage] for stage in STAGE_ORDER if stage in best}


def _load_comparison_metrics() -> dict:
    rating_only = {}
    if RATING_ONLY_METRICS.exists():
        rating_only = json.loads(RATING_ONLY_METRICS.read_text(encoding="utf-8"))["ours"]

    paper_full = {}
    if PAPER_FULL_METRICS.exists():
        paper_full = json.loads(PAPER_FULL_METRICS.read_text(encoding="utf-8"))

    return {"rating_only": rating_only, "paper_full": paper_full}


def _plot_rating_only_training(curves: dict, out_path: Path) -> None:
    losses = curves.get("train_loss_steps", [])
    if not losses:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, linewidth=0.9, color="C0")
    ax.set_xlabel("logged step (~every 10 optimizer steps)")
    ax.set_ylabel("train loss")
    summary = curves.get("summary", {})
    subtitle = ""
    if summary:
        subtitle = (
            f"final train loss={summary.get('train_loss', 0):.3f}, "
            f"val RMSE={summary.get('eval_rmse', 0):.3f}, "
            f"{summary.get('train_hours', 0):.1f} h"
        )
    ax.set_title(f"Exp3RT rating-only (1 epoch) — train loss\n{subtitle}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_paper_full_training(stages: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, stage in zip(axes, STAGE_ORDER):
        data = stages.get(stage, {})
        losses = data.get("train_loss_steps", [])
        evals = data.get("evals", [])
        if losses:
            ax.plot(losses, linewidth=0.8, color="C0", label="train loss")
        if evals:
            ep = [e["epoch"] for e in evals]
            ax2 = ax.twinx()
            ax2.plot(
                ep,
                [e["eval_loss"] for e in evals],
                "s--",
                color="C1",
                label="val loss",
            )
            if any(e.get("eval_rmse") for e in evals):
                ax2.plot(
                    ep,
                    [e["eval_rmse"] for e in evals],
                    "o-",
                    color="C2",
                    label="val RMSE",
                )
            ax2.set_ylabel("validation")
        summary = data.get("summary", {})
        note = ""
        if summary:
            note = f"train={summary.get('train_loss', 0):.3f}"
            if summary.get("eval_rmse"):
                note += f", val RMSE={summary['eval_rmse']:.3f}"
        ax.set_title(f"{STAGE_TITLES[stage]}\n{note}")
        ax.set_xlabel("logged step")
        ax.set_ylabel("train loss")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Exp3RT Amazon-Books — 4-stage SFT training curves (Qwen2.5-7B)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_rating_eval_curve(stages: dict, out_path: Path) -> None:
    rating = stages.get("rating", {})
    evals = rating.get("evals", [])
    if not evals:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ep = [e["epoch"] for e in evals]
    ax.plot(ep, [e["eval_rmse"] for e in evals], "o-", color="C2", label="val RMSE")
    ax.axhline(PAPER_AMAZON_TOTAL["rmse"], color="C3", linestyle="--", label="paper test RMSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("RMSE (validation)")
    ax.set_title("Exp3RT paper-full — rating stage validation RMSE")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_metrics_vs_paper(metrics: dict, out_path: Path) -> None:
    ro = metrics.get("rating_only", {}).get("expected_rating", {})
    pf = metrics.get("paper_full", {}).get("expected_rating", {})

    methods = [
        ("Exp3RT (paper)", PAPER_AMAZON_TOTAL),
        ("MF (paper)", PAPER_BASELINES["MF"]),
        ("LLMRec GPT-3.5 (paper)", PAPER_BASELINES["LLMRec GPT-3.5"]),
        ("Ours rating-only", ro),
        ("Ours paper-full", pf),
    ]
    labels = [m[0] for m in methods if m[1].get("rmse") is not None]
    rmse = [m[1]["rmse"] for m in methods if m[1].get("rmse") is not None]
    mae = [m[1]["mae"] for m in methods if m[1].get("mae") is not None]

    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors_rmse = ["C3" if "paper" in lb.lower() and "ours" not in lb.lower() else "C0" for lb in labels]
    colors_rmse = [
        "C1" if lb.startswith("Ours") else ("C3" if "Exp3RT (paper)" in lb else "C4")
        for lb in labels
    ]
    axes[0].bar(x, rmse, width=0.6, color=colors_rmse, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("RMSE ↓")
    axes[0].set_title("Amazon-Books test RMSE (n=11,743)")
    axes[0].grid(axis="y", alpha=0.25)

    colors_mae = colors_rmse
    axes[1].bar(x, mae, width=0.6, color=colors_mae, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("MAE ↓")
    axes[1].set_title("Amazon-Books test MAE (n=11,743)")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Exp3RT reproduction vs SIGIR 2025 Table 2 (expected_rating)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_metrics_delta(metrics: dict, out_path: Path) -> None:
    ro = metrics["rating_only"].get("expected_rating", {})
    pf = metrics["paper_full"].get("expected_rating", {})
    paper = PAPER_AMAZON_TOTAL

    entries = [
        ("Rating-only\nΔ vs paper", ro["rmse"] - paper["rmse"], ro["mae"] - paper["mae"]),
        ("Paper-full\nΔ vs paper", pf["rmse"] - paper["rmse"], pf["mae"] - paper["mae"]),
    ]
    labels = [e[0] for e in entries]
    rmse_d = [e[1] for e in entries]
    mae_d = [e[2] for e in entries]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, rmse_d, width, label="Δ RMSE", color="C0")
    ax.bar(x + width / 2, mae_d, width, label="Δ MAE", color="C1")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("delta (negative = better than paper)")
    ax.set_title("Exp3RT: improvement over paper Exp3RT (expected_rating)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    CURVES.mkdir(parents=True, exist_ok=True)

    rating_curves = (
        _parse_single_stage_log(RATING_ONLY_LOG.read_text(encoding="utf-8", errors="replace"))
        if RATING_ONLY_LOG.exists()
        else {}
    )
    paper_full_stages = (
        _parse_paper_full_log(PAPER_FULL_LOG) if PAPER_FULL_LOG.exists() else {}
    )
    comparison = _load_comparison_metrics()

    (CURVES / "exp3rt_rating_only.json").write_text(
        json.dumps(rating_curves, indent=2), encoding="utf-8"
    )
    paper_full_export = {
        stage: {k: v for k, v in data.items() if k != "_n_loss"}
        for stage, data in paper_full_stages.items()
    }
    (CURVES / "exp3rt_paper_full_stages.json").write_text(
        json.dumps(paper_full_export, indent=2), encoding="utf-8"
    )

    plots_created = []
    p1 = PLOTS / "exp3rt_training_loss_rating_only.png"
    _plot_rating_only_training(rating_curves, p1)
    if p1.exists():
        plots_created.append(str(p1.relative_to(OUT)))

    p2 = PLOTS / "exp3rt_training_curves_paper_full.png"
    _plot_paper_full_training(paper_full_stages, p2)
    if p2.exists():
        plots_created.append(str(p2.relative_to(OUT)))

    p3 = PLOTS / "exp3rt_rating_val_rmse_paper_full.png"
    _plot_rating_eval_curve(paper_full_stages, p3)
    if p3.exists():
        plots_created.append(str(p3.relative_to(OUT)))

    p4 = PLOTS / "exp3rt_metrics_vs_paper.png"
    if comparison["rating_only"] and comparison["paper_full"]:
        _plot_metrics_vs_paper(comparison, p4)
        plots_created.append(str(p4.relative_to(OUT)))

        p5 = PLOTS / "exp3rt_metrics_delta_vs_paper.png"
        _plot_metrics_delta(comparison, p5)
        plots_created.append(str(p5.relative_to(OUT)))

    meta = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "plots": plots_created,
        "metrics_sources": {
            "rating_only": str(RATING_ONLY_METRICS.relative_to(ROOT)),
            "paper_full": str(PAPER_FULL_METRICS.relative_to(ROOT)),
        },
        "logs": {
            "rating_only": str(RATING_ONLY_LOG.relative_to(ROOT)),
            "paper_full": str(PAPER_FULL_LOG.relative_to(ROOT)),
        },
        "paper_reference": PAPER_AMAZON_TOTAL,
        "ours_expected_rating": {
            "rating_only": comparison["rating_only"].get("expected_rating"),
            "paper_full": comparison["paper_full"].get("expected_rating"),
        },
    }
    (PLOTS / "exp3rt_plots_manifest.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"Wrote {len(plots_created)} plots to {PLOTS}")
    for p in plots_created:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
