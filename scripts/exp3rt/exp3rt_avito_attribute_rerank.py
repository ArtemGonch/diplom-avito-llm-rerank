#!/usr/bin/env python3
"""
Exp3RT-style Avito diagnostic: pseudo-reviews + attribute-wise candidate scoring.

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.avito import AvitoSERP  # noqa: E402
from common.metrics import evaluate_batch  # noqa: E402


def build_user_pseudo_profile(users: pd.DataFrame | None, user_id: int | None) -> str:
    """Pseudo profile from raw contact-taxonomy fields.

    Despite their exported names, ``brand`` and ``model_name`` in the current
    history snapshot contain category-like values (for example, household
    goods or machinery taxonomy), not the automotive brand/model vocabulary
    used by the candidate listings.  Keep that distinction explicit so this
    diagnostic is never described as an automotive preference profile.
    """
    if users is None or user_id is None:
        return "The user has no recorded contact history on Avito."
    hist = users[users["user_id"] == user_id]
    if hist.empty:
        return "The user has no recorded contact history on Avito."
    level_1 = Counter(str(v) for v in hist["brand"].dropna() if str(v).strip())
    level_2 = Counter(str(v) for v in hist["model_name"].dropna() if str(v).strip())
    top_level_1 = ", ".join(v for v, _ in level_1.most_common(3)) or "unknown"
    top_level_2 = ", ".join(v for v, _ in level_2.most_common(3)) or "unknown"
    lines = [
        f"Raw level-1 contact taxonomy values: {top_level_1}.",
        f"Raw level-2 contact taxonomy values: {top_level_2}.",
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
    user_id: int | None,
) -> np.ndarray:
    """Score candidates by observable query-token overlap only.

    The current contact export has no trustworthy mapping from its taxonomy
    values to the automotive candidate attributes, so it is intentionally not
    used as brand/model or price evidence here.
    """
    scores = []
    for _, row in rows.iterrows():
        s = 0.0
        brand = str(row.get("brand", "")).lower()
        model = str(row.get("model_name", "")).lower()
        title = str(row.get("title", "")).lower()
        qtoks = set(re.findall(r"[a-zа-я0-9]+", query.lower()))
        itoks = set(re.findall(r"[a-zа-я0-9]+", title + " " + brand + " " + model))
        s += 0.3 * len(qtoks & itoks)
        # contacts_daily is the evaluation target and clicks_daily is a
        # post-exposure behavioural signal.  Neither is a legal rank-time
        # feature in this offline experiment.
        scores.append(s)
    return np.asarray(scores, dtype=np.float64)


def score_diagnostics(scores_list: list[np.ndarray]) -> dict[str, int | float]:
    """Quantify degenerate all-tie rankings before metrics are interpreted."""
    n = len(scores_list)
    constant = sum(len(np.unique(scores)) <= 1 for scores in scores_list)
    all_zero = sum(bool(len(scores) and np.all(scores == 0)) for scores in scores_list)
    return {
        "evaluated_serps": n,
        "constant_score_serps": constant,
        "all_zero_score_serps": all_zero,
        "constant_score_fraction": float(constant / n) if n else 0.0,
    }


def history_schema_audit(data: AvitoSERP) -> dict[str, int | bool]:
    """Record facts that prevent the raw history fields being called brands/models."""
    if data.users is None:
        return {"history_available": False}
    listing_brands = set(data.items["brand"].dropna().astype(str).str.strip().str.lower())
    listing_models = set(data.items["model_name"].dropna().astype(str).str.strip().str.lower())
    history_level_1 = set(data.users["brand"].dropna().astype(str).str.strip().str.lower())
    history_level_2 = set(data.users["model_name"].dropna().astype(str).str.strip().str.lower())
    listing_item_ids = set(data.items["item_id"].dropna().astype(np.int64))
    history_item_ids = set(data.users["item_id"].dropna().astype(np.int64))
    return {
        "history_available": True,
        "history_rows": int(len(data.users)),
        "history_users": int(data.users["user_id"].nunique()),
        "history_has_price": bool("price" in data.users.columns),
        "exact_level1_to_listing_brand_vocab_overlap": int(len(history_level_1 & listing_brands)),
        "exact_level2_to_listing_model_vocab_overlap": int(len(history_level_2 & listing_models)),
        "exact_history_to_listing_item_overlap": int(len(history_item_ids & listing_item_ids)),
        "serp_rank_time_available": bool(
            {"rank_time", "serp_time", "serp_timestamp"} & set(data.items.columns)
        ),
    }


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
        labels = data._labels_for_group(grp).astype(float)
        user_id = None
        if "user_id" in grp.columns:
            uvals = grp["user_id"].dropna()
            if len(uvals):
                user_id = int(uvals.iloc[0])
        profile = build_user_pseudo_profile(data.users, user_id)
        query = data.serp_query_text(serp_x)

        if mode == "llm" and gen is not None:
            sc = llm_score_batch(profile, query, grp, gen)
        else:
            sc = heuristic_scores(profile, query, grp, data.users, user_id)

        scores_h.append(sc)
        scores_p.append(position_baseline_scores(grp))
        labels_list.append(labels)
        n += 1
        if max_samples and n >= max_samples:
            break

    diagnostics = score_diagnostics(scores_h)
    schema_audit = history_schema_audit(data)
    invalid_reasons = [
        "Current history fields are not mapped to automotive candidate attributes.",
        "No SERP rank-time timestamp is available for a temporal history cutoff.",
    ]
    if diagnostics["constant_score_fraction"] >= 0.95:
        invalid_reasons.append(
            "At least 95% of SERPs have constant candidate scores; reported NDCG is tie/order driven."
        )
    return {
        "n_serps": n,
        "claim_scope": "diagnostic only; not a valid personalized ranking result",
        "valid_for_claims": False,
        "invalid_reasons": invalid_reasons,
        "evaluation": {
            "relevance": f"graded normalised {data.label_field}_daily",
            "rank_time_features": [
                "category/location query proxy",
                "listing title/brand/model/price",
            ],
            "excluded_post_exposure_features": ["contacts_daily", "clicks_daily"],
            "history_usage": "not used for candidate matching: exported fields are incompatible taxonomy values",
        },
        "history_schema_audit": schema_audit,
        "score_diagnostics": diagnostics,
        "exp3rt_style": evaluate_batch(scores_h, labels_list),
        "position_base": evaluate_batch(scores_p, labels_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3RT-style Avito schema/score diagnostic")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exp3rt/exp3rt_avito_smoke.yaml")
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
        from common.llm.generate import create_knowledge_generator  # noqa: E402

        gen = create_knowledge_generator(cfg)

    print(f"Exp3RT Avito diagnostic mode={args.mode} split={args.split} serps={len(serp_ids)}")
    metrics = run_eval(data, serp_ids, args.mode, gen, args.max_serps)
    print(json.dumps(metrics, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mode": args.mode, "split": args.split, **metrics}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
