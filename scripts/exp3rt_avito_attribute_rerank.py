#!/usr/bin/env python3
"""
Exp3RT-style Avito MVP: pseudo-reviews + attribute-wise candidate scoring.

Builds text profiles from Avito parquet (no Amazon reviews) and reranks SERP
candidates by attribute fit. Modes:
  heuristic — fast attribute matching (default)
  llm       — Qwen compares items on attribute checklist (needs GPU)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ur4rec.data.avito import AvitoSERP  # noqa: E402
from ur4rec.metrics import evaluate_batch  # noqa: E402


def build_user_pseudo_profile(users: pd.DataFrame | None, seller_user_id: int | None) -> str:
    """Pseudo user profile from contact history (Exp3RT stage-2 analogue)."""
    if users is None or seller_user_id is None:
        return "The user has no recorded contact history on Avito."
    hist = users[users["user_id"] == seller_user_id]
    if hist.empty:
        return "The user has no recorded contact history on Avito."
    brands = Counter(str(b) for b in hist["brand"].dropna() if str(b).strip())
    models = Counter(str(m) for m in hist["model_name"].dropna() if str(m).strip())
    prices = hist["price"].dropna() if "price" in hist.columns else []
    top_brands = ", ".join(b for b, _ in brands.most_common(3)) or "various brands"
    top_models = ", ".join(m for m, _ in models.most_common(3)) or "various models"
    price_hint = ""
    if len(prices):
        price_hint = f" Typical contact price around {float(np.median(prices)):.0f}."
    lines = [
        f"The user frequently contacts listings from brands: {top_brands}.",
        f"Preferred models include: {top_models}.{price_hint}",
        f"Total past contacts in sample: {len(hist)}.",
    ]
    return " ".join(lines)


def item_pseudo_description(row: pd.Series) -> str:
    """Item 'review' proxy from listing attributes."""
    parts = [
        str(row.get("title") or "listing"),
        f"brand={row.get('brand', 'unknown')}",
        f"model={row.get('model_name', 'unknown')}",
        f"price={row.get('price', 0)}",
        f"mileage_km={row.get('mileage_km', 'n/a')}",
        f"fuel={row.get('fuel_text', '')}",
        f"body={row.get('body_type', '')}",
    ]
    return " | ".join(p for p in parts if p)


def heuristic_scores(
    profile: str,
    query: str,
    rows: pd.DataFrame,
    users: pd.DataFrame | None,
    seller_id: int | None,
) -> np.ndarray:
    """Score candidates by brand/price/query overlap with pseudo profile."""
    prof_lower = profile.lower()
    brand_counts: Counter[str] = Counter()
    if users is not None and seller_id is not None:
        hist = users[users["user_id"] == seller_id]
        brand_counts = Counter(str(b).lower() for b in hist["brand"].dropna())

    scores = []
    for _, row in rows.iterrows():
        s = 0.0
        brand = str(row.get("brand", "")).lower()
        model = str(row.get("model_name", "")).lower()
        title = str(row.get("title", "")).lower()
        price = float(row.get("price") or 0)
        if brand and brand in prof_lower:
            s += 2.0
        if brand_counts.get(brand, 0) > 0:
            s += 1.5 * np.log1p(brand_counts[brand])
        if model and model in prof_lower:
            s += 1.0
        qtoks = set(re.findall(r"[a-zа-я0-9]+", query.lower()))
        itoks = set(re.findall(r"[a-zа-я0-9]+", title + " " + brand + " " + model))
        s += 0.3 * len(qtoks & itoks)
        s += 0.001 * float(row.get("contacts_daily") or 0)
        s += 0.0005 * float(row.get("clicks_daily") or 0)
        if price > 0 and "price" in prof_lower:
            s -= 0.000001 * abs(price - _median_price_from_profile(profile))
        scores.append(s)
    return np.asarray(scores, dtype=np.float64)


def _median_price_from_profile(profile: str) -> float:
    nums = [float(x) for x in re.findall(r"around (\d+(?:\.\d+)?)", profile)]
    return float(np.median(nums)) if nums else 0.0


def position_baseline_scores(rows: pd.DataFrame) -> np.ndarray:
    """Higher score for better (lower) block position."""
    pos = rows["block_pos"].astype(float).values if "block_pos" in rows else np.arange(len(rows))
    return -pos


def llm_score_batch(
    profile: str,
    query: str,
    rows: pd.DataFrame,
    gen,
) -> np.ndarray:
    """Ask LLM for 1-5 fit score per item (Exp3RT stage-3 style, simplified)."""
    scores = []
    for _, row in rows.iterrows():
        desc = item_pseudo_description(row)
        prompt = (
            "Rate how well this Avito listing fits the user on a scale 1-5.\n"
            "Consider: brand preference, price fit, mileage, relevance to search query.\n"
            "Reply with ONLY one digit 1-5.\n\n"
            f"<Query>\n{query}\n\n"
            f"<User Profile>\n{profile}\n\n"
            f"<Item>\n{desc}\n"
        )
        out = gen.generate_user_preference(prompt).strip()
        m = re.search(r"[1-5]", out)
        scores.append(float(m.group()) if m else 3.0)
    return np.asarray(scores, dtype=np.float64)


def run_eval(
    data: AvitoSERP,
    serp_ids: list[int],
    mode: str,
    gen=None,
    max_samples: int | None = None,
) -> dict:
    scores_h, scores_p, labels_list = [], [], []
    n = 0
    for sid in serp_ids:
        serp_x = data.idx2serp[sid]
        grp = data.items[data.items["serp_x"] == serp_x]
        if len(grp) < data.min_serp_size:
            continue
        grp = grp.copy()
        grp["_idx"] = grp["item_id"].map(lambda x: data.item2idx[int(x)])
        labels = (data._labels_for_group(grp) > 0).astype(float)
        seller = None
        if "user_id" in grp.columns:
            uvals = grp["user_id"].dropna()
            if len(uvals):
                seller = int(uvals.iloc[0])
        profile = build_user_pseudo_profile(data.users, seller)
        query = data.serp_query_text(serp_x)

        if mode == "llm" and gen is not None:
            sc = llm_score_batch(profile, query, grp, gen)
        else:
            sc = heuristic_scores(profile, query, grp, data.users, seller)

        scores_h.append(sc)
        scores_p.append(position_baseline_scores(grp))
        labels_list.append(labels)
        n += 1
        if max_samples and n >= max_samples:
            break

    return {
        "n_serps": n,
        "exp3rt_style": evaluate_batch(scores_h, labels_list),
        "position_base": evaluate_batch(scores_p, labels_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3RT-style Avito attribute rerank MVP")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp3rt_avito_smoke.yaml")
    parser.add_argument("--mode", choices=("heuristic", "llm"), default="heuristic")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--max-serps", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "papers/exp3rt/reproduction/results/avito_exp3rt_mvp.json")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ds = cfg["dataset"]
    items_path = ROOT / ds.get("items_path", "items_with_attrs.parquet")
    users_path = ROOT / ds.get("users_path", "users_with_history.parquet")

    data = AvitoSERP.from_parquet(
        items_path,
        users_path if users_path.exists() else None,
        label_field=ds.get("label_field", "contacts"),
        min_serp_size=ds.get("min_serp_size", 10),
    )
    train_u, val_u, test_u = data.train_val_test_split(
        ds.get("train_ratio", 0.8), ds.get("val_ratio", 0.1), cfg.get("seed", 42)
    )
    if args.split == "val":
        cap = ds.get("max_val_serps")
        serp_ids = val_u[: cap or len(val_u)]
    else:
        cap = ds.get("max_test_serps")
        serp_ids = test_u[: cap or len(test_u)]

    gen = None
    if args.mode == "llm":
        from ur4rec.llm.generate import create_knowledge_generator  # noqa: E402

        gen = create_knowledge_generator(cfg)

    print(f"Exp3RT Avito MVP mode={args.mode} split={args.split} serps={len(serp_ids)}")
    metrics = run_eval(data, serp_ids, args.mode, gen, args.max_serps)
    print(json.dumps(metrics, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mode": args.mode, "split": args.split, **metrics}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
