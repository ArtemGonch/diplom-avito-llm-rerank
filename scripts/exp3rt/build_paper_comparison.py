#!/usr/bin/env python3
"""Build Exp3RT Amazon-Books test metrics vs SIGIR 2025 paper Table 2."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = Path(
    os.environ.get(
        "EXP3RT_METRICS_PATH",
        "checkpoints/exp3rt/amazon_book_qwen/"
        "amazon-book_rating_r128_alpha32_seed425/metrics.json",
    )
)
if not METRICS_PATH.is_absolute():
    METRICS_PATH = ROOT / METRICS_PATH
OUT_DIR = ROOT / "results/current/tables"

PAPER_AMAZON_TOTAL = {
    "rmse": 0.6508,
    "mae": 0.4297,
    "source": "Exp3RT SIGIR 2025 Table 2, Amazon-Book Total",
}
PAPER_BASELINES = {
    "MF": {"rmse": 0.6683, "mae": 0.4572},
    "LLMRec_GPT35_FS": {"rmse": 0.7852, "mae": 0.4562},
    "LLMRec_GPT4_FS": {"rmse": 0.8386, "mae": 0.5004},
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not METRICS_PATH.exists():
        print(f"Missing {METRICS_PATH} — run test+eval first")
        return

    ours = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    rows = []
    for method, key in [
        ("Ours (max_prob_rating)", "max_prob_rating"),
        ("Ours (expected_rating)", "expected_rating"),
    ]:
        m = ours[key]
        rows.append(
            {
                "method": method,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "n": m.get("n"),
            }
        )

    # Prefer expected_rating as paper-style (softmax over digit logprobs)
    primary = rows[1]

    lines = [
        "# Exp3RT Amazon-Books: test vs paper (Table 2, Total)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Method | RMSE | MAE | n |",
        "|--------|------|-----|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['rmse']:.4f} | {r['mae']:.4f} | {r.get('n', '—')} |"
        )
    lines.extend(
        [
            "",
            "## Paper reference",
            "",
            f"| Exp3RT (paper) | {PAPER_AMAZON_TOTAL['rmse']:.4f} | {PAPER_AMAZON_TOTAL['mae']:.4f} | test |",
            f"| MF | {PAPER_BASELINES['MF']['rmse']:.4f} | {PAPER_BASELINES['MF']['mae']:.4f} |",
            f"| LLMRec GPT-3.5 FS | {PAPER_BASELINES['LLMRec_GPT35_FS']['rmse']:.4f} | {PAPER_BASELINES['LLMRec_GPT35_FS']['mae']:.4f} |",
            "",
            "## Delta (ours expected_rating vs paper Exp3RT)",
            "",
            f"- RMSE: {primary['rmse'] - PAPER_AMAZON_TOTAL['rmse']:+.4f} (lower is better)",
            f"- MAE: {primary['mae'] - PAPER_AMAZON_TOTAL['mae']:+.4f} (lower is better)",
            "",
            "## Protocol notes",
            "",
            "- **Paper:** Llama-3-8B, 3-stage SFT, cutoff 1200, up to 10 epochs, vLLM test",
            "- **Ours:** Qwen2.5-7B, rating stage only (profiles prebuilt), 1 epoch, cutoff 1024",
            "- Metric: RMSE/MAE on full `rating_bias/test.json` (11,743 rows)",
        ]
    )

    md_path = OUT_DIR / "exp3rt_amazon_vs_paper.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "ours": ours,
        "paper_total": PAPER_AMAZON_TOTAL,
        "paper_baselines": PAPER_BASELINES,
        "delta_expected_vs_paper": {
            "rmse": primary["rmse"] - PAPER_AMAZON_TOTAL["rmse"],
            "mae": primary["mae"] - PAPER_AMAZON_TOTAL["mae"],
        },
    }
    json_path = OUT_DIR / "exp3rt_amazon_vs_paper.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    shutil_copy = ROOT / "results/current/metrics/exp3rt_amazon_test_metrics.json"
    shutil_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil_copy.write_text(json.dumps(ours, indent=2), encoding="utf-8")

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
