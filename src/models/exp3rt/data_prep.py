"""Prepare Exp3RT bundled JSON (merge shards, validate paths)."""

from __future__ import annotations

import json
from pathlib import Path


def merge_json_shards(input_dir: Path, prefix: str, output_name: str) -> Path:
    input_dir = Path(input_dir)
    shards = sorted(p for p in input_dir.glob(f"{prefix}*.json") if p.name != output_name)
    merged: list = []
    for path in shards:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {path}")
        merged.extend(data)
    out = input_dir / output_name
    out.write_text(json.dumps(merged), encoding="utf-8")
    return out


def prepare_amazon_book(data_root: Path) -> dict[str, Path]:
    data_root = Path(data_root)
    ds = data_root / "amazon-book"
    rating_dir = ds / "rating_bias"
    pref_dir = ds / "preference_extraction"

    rating_train = merge_json_shards(rating_dir, "train_", "train.json")
    preference_train = merge_json_shards(pref_dir, "preference_train_", "preference_train.json")

    paths = {
        "preference_train": preference_train,
        "preference_valid": pref_dir / "preference_valid.json",
        "user_train": ds / "user_profile" / "user_train.json",
        "user_valid": ds / "user_profile" / "user_valid.json",
        "item_train": ds / "item_profile" / "item_train.json",
        "item_valid": ds / "item_profile" / "item_valid.json",
        "rating_train": rating_train,
        "rating_valid": rating_dir / "valid.json",
        "rating_test": rating_dir / "test.json",
        "topk_train": data_root / "topk" / "amazon-book" / "train.txt",
        "topk_test": data_root / "topk" / "amazon-book" / "test.txt",
    }
    missing = [k for k, p in paths.items() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Exp3RT data files: {missing}")
    return paths


def count_rows(path: Path) -> int:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return len(data)
