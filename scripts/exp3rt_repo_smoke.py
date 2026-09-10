#!/usr/bin/env python3
"""Validate cloned EXP3RT repo data pipeline without full Llama-3 training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "papers" / "exp3rt" / "assets" / "github_repo"


def _load_json(path: Path) -> list | dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sample_keys(obj: dict, n: int = 8) -> list[str]:
    return list(obj.keys())[:n] if isinstance(obj, dict) else []


def main() -> int:
    if not REPO.exists():
        print(f"ERROR: repo not found at {REPO}")
        return 1

    print(f"EXP3RT repo smoke: {REPO}\n")

    # 1) merge train shards (idempotent) — run from repo root
    import os

    orig = os.getcwd()
    os.chdir(REPO)
    try:
        bias_dir = REPO / "data/amazon-book/rating_bias"
        shards = sorted(bias_dir.glob("train_*.json"))
        merged = []
        for p in shards:
            merged.extend(_load_json(p))
        out = bias_dir / "train.json"
        if not out.exists() or out.stat().st_mtime < max(p.stat().st_mtime for p in shards):
            out.write_text(json.dumps(merged), encoding="utf-8")
        print(f"  [OK] merged train shards: {len(shards)} -> train.json ({len(merged)} rows)")
    finally:
        os.chdir(orig)

    ds = "amazon-book"
    paths = {
        "preference_train": REPO / f"data/{ds}/preference_extraction/preference_train_0.json",
        "user_train": REPO / f"data/{ds}/user_profile/user_train.json",
        "item_train": REPO / f"data/{ds}/item_profile/item_train.json",
        "rating_train": REPO / f"data/{ds}/rating_bias/train.json",
        "rating_valid": REPO / f"data/{ds}/rating_bias/valid.json",
        "rating_test": REPO / f"data/{ds}/rating_bias/test.json",
        "topk_train": REPO / f"data/topk/{ds}/train.txt",
    }

    ok = True
    for name, p in paths.items():
        if not p.exists():
            print(f"  [MISSING] {name}: {p}")
            ok = False
            continue
        if p.suffix == ".json":
            data = _load_json(p)
            n = len(data) if isinstance(data, list) else len(data.keys())
            print(f"  [OK] {name}: {n} records")
            if isinstance(data, list) and data:
                ex = data[0]
                if isinstance(ex, dict):
                    print(f"       keys: {list(ex.keys())[:10]}")
        else:
            lines = p.read_text(encoding="utf-8").strip().splitlines()
            print(f"  [OK] {name}: {len(lines)} lines")

    # Schema spot-check on rating example
    rating = _load_json(paths["rating_train"])
    if isinstance(rating, list) and rating:
        ex = rating[0]
        required = {"instruction", "input", "output"}
        missing = required - set(ex.keys())
        if missing:
            print(f"  [WARN] rating_train missing keys: {missing}")
        else:
            print("  [OK] rating_train Alpaca schema (instruction/input/output)")

    pref = _load_json(paths["preference_train"])
    if isinstance(pref, list):
        print(f"  [OK] preference_train sample output: {str(pref[0].get('output', ''))[:120]}...")

    print("\nDependencies for full train (not run here):")
    print("  - meta-llama/Meta-Llama-3-8B-Instruct (gated HF)")
    print("  - accelerate, peft, bitsandbytes, vllm")
    print("  - GPU: QLoRA 3 stages x 3 epochs")

    print("\nSmoke result:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
