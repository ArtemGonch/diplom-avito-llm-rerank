#!/usr/bin/env python3
"""Train a leakage-safe *local diagnostic* CatBoost ranker on Avito SERPs.

This is not the team's production CatBoost baseline owned by Roma.  It exists
to validate the evaluator and provide an executable score artifact while the
authoritative split/config/scores are still missing.  Target and post-exposure
signals are never exposed to the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.metrics import evaluate_batch  # noqa: E402
from data.avito import AvitoSERP  # noqa: E402


FORBIDDEN_RANK_TIME_COLUMNS = {
    "contacts_daily",
    "clicks_daily",
    "has_tc_events",
    "has_x_events",
    "serp_is_positive",
}

NUMERIC_FEATURES = [
    "block_pos",
    "price",
    "image_count",
    "seller_is_private",
    "price_percentile",
    "price_diff_from_median",
    "seconds_age",
    "title_len",
    "desc_len",
    "mileage_km",
    "imv",
]

CATEGORICAL_FEATURES = [
    "block",
    "query_infm_logical_category",
    "query_loc",
    "brand",
    "model_name",
    "gearbox_text",
    "fuel_text",
    "body_type",
    "drive_text",
    "doors_text",
    "year_text",
    "rf_reg_text",
]


def rank_time_feature_columns() -> list[str]:
    features = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    overlap = FORBIDDEN_RANK_TIME_COLUMNS & set(features)
    if overlap:
        raise AssertionError(f"Post-exposure leakage in CatBoost features: {sorted(overlap)}")
    return features


def _prepare_frame(items: pd.DataFrame, serp_names: Sequence[str]) -> pd.DataFrame:
    frame = items[items["serp_x"].isin(set(serp_names))].copy()
    frame = frame.sort_values(["serp_x", "block_pos", "item_id"], kind="stable").reset_index(drop=True)
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        frame[column] = frame[column].fillna(0.0).astype(np.float32)
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].fillna("<missing>").astype(str)
    relevance = frame["contacts_daily"].clip(lower=0).astype(np.float64)
    maxima = relevance.groupby(frame["serp_x"]).transform("max")
    frame["_relevance"] = np.where(maxima > 0, relevance / maxima, 0.0).astype(np.float32)
    return frame


def _pool(frame: pd.DataFrame):
    from catboost import Pool

    features = rank_time_feature_columns()
    return Pool(
        frame[features],
        label=frame["_relevance"],
        group_id=frame["serp_x"],
        cat_features=CATEGORICAL_FEATURES,
        feature_names=features,
    )


def _evaluate_frame(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    score_groups: list[np.ndarray] = []
    label_groups: list[np.ndarray] = []
    offset = 0
    for _, group in frame.groupby("serp_x", sort=False):
        length = len(group)
        score_groups.append(np.asarray(scores[offset : offset + length], dtype=np.float64))
        label_groups.append(group["_relevance"].to_numpy(dtype=np.float64))
        offset += length
    if offset != len(scores):
        raise ValueError("Score count does not match test frame")
    return evaluate_batch(score_groups, label_groups, ks=(5, 10))


def _position_scores(frame: pd.DataFrame) -> np.ndarray:
    return -frame["block_pos"].to_numpy(dtype=np.float64)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=ROOT / "items_with_attrs.parquet")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument(
        "--reuse-existing-models",
        action="store_true",
        help="Load seed_<seed>.cbm when present (caller must keep hyperparameters identical).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/current/metrics/avito_local_catboost_diagnostic.json",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=ROOT / "data/avito/benchmark/local_catboost_eval_scores.parquet",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "checkpoints/avito_local_catboost",
    )
    args = parser.parse_args()

    from catboost import CatBoostRanker, __version__ as catboost_version

    started = time.time()
    data = AvitoSERP.from_parquet(args.items, users_path=None, label_field="contacts", min_serp_size=10)
    train_ids, val_ids, test_ids = data.train_val_test_split(seed=args.split_seed)
    train_names = [data.idx2serp[index] for index in train_ids]
    val_names = [data.idx2serp[index] for index in val_ids]
    test_names = [data.idx2serp[index] for index in test_ids]
    train = _prepare_frame(data.items, train_names)
    val = _prepare_frame(data.items, val_names)
    test = _prepare_frame(data.items, test_names)
    train_pool = _pool(train)
    val_pool = _pool(val)
    test_pool = _pool(test)
    dataset_sha256 = _file_sha256(args.items)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    seed_scores: list[np.ndarray] = []
    seed_val_scores: list[np.ndarray] = []
    seed_metrics: dict[str, dict[str, float]] = {}
    seed_val_metrics: dict[str, dict[str, float]] = {}
    best_iterations: dict[str, int] = {}
    for seed in args.seeds:
        print(f"Training local diagnostic CatBoost seed={seed}", flush=True)
        model_path = args.model_dir / f"seed_{seed}.cbm"
        model_meta_path = args.model_dir / f"seed_{seed}.meta.json"
        model_kwargs = dict(
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            loss_function="YetiRankPairwise",
            eval_metric="NDCG:top=10",
            random_seed=seed,
            l2_leaf_reg=5.0,
            random_strength=0.5,
            verbose=100,
            allow_writing_files=False,
            thread_count=-1,
        )
        if args.task_type == "GPU":
            model_kwargs.update(task_type="GPU", devices=args.devices)
        expected_model_meta = {
            "version": 1,
            "dataset_sha256": dataset_sha256,
            "split_seed": args.split_seed,
            "train_serps": len(train_ids),
            "features": rank_time_feature_columns(),
            "seed": seed,
            "catboost_version": catboost_version,
            "training": {
                "iterations": args.iterations,
                "depth": args.depth,
                "learning_rate": args.learning_rate,
                "loss_function": "YetiRankPairwise",
                "l2_leaf_reg": 5.0,
                "random_strength": 0.5,
                "task_type": args.task_type,
                "early_stopping_rounds": (
                    None if args.task_type == "GPU" else args.early_stopping_rounds
                ),
            },
        }
        model = CatBoostRanker(**model_kwargs)
        if args.reuse_existing_models and model_path.exists():
            if not model_meta_path.exists():
                raise ValueError(
                    f"Refusing unverified model reuse: missing {model_meta_path}. "
                    "Retrain without --reuse-existing-models."
                )
            actual_model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
            if actual_model_meta != expected_model_meta:
                raise ValueError(
                    f"Refusing stale/mismatched model reuse for seed={seed}: "
                    f"metadata differs at {model_meta_path}"
                )
            print(f"Reusing {model_path}", flush=True)
            model.load_model(model_path)
        elif args.task_type == "GPU":
            # CatBoost 1.2.x computes NDCG validation on CPU, making this tiny
            # diagnostic slower than CPU training.  Keep a fixed, recorded
            # iteration budget on GPU; authoritative model selection belongs
            # to the missing team CatBoost protocol.
            model.fit(train_pool)
        else:
            model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=args.early_stopping_rounds,
                use_best_model=True,
            )
        model.save_model(model_path)
        _write_json(model_meta_path, expected_model_meta)
        val_scores = np.asarray(model.predict(val_pool), dtype=np.float64)
        scores = np.asarray(model.predict(test_pool), dtype=np.float64)
        seed_val_scores.append(val_scores)
        seed_scores.append(scores)
        seed_val_metrics[str(seed)] = _evaluate_frame(val, val_scores)
        seed_metrics[str(seed)] = _evaluate_frame(test, scores)
        raw_best_iteration = model.get_best_iteration()
        best_iterations[str(seed)] = (
            int(raw_best_iteration)
            if raw_best_iteration is not None and raw_best_iteration >= 0
            else args.iterations - 1
        )

    mean_scores = np.mean(np.stack(seed_scores), axis=0)
    mean_val_scores = np.mean(np.stack(seed_val_scores), axis=0)
    position_val_metrics = _evaluate_frame(val, _position_scores(val))
    position_metrics = _evaluate_frame(test, _position_scores(test))
    ensemble_val_metrics = _evaluate_frame(val, mean_val_scores)
    ensemble_metrics = _evaluate_frame(test, mean_scores)
    metric_keys = sorted(next(iter(seed_metrics.values())))
    stability = {
        metric: {
            "mean": float(np.mean([values[metric] for values in seed_metrics.values()])),
            "std": float(np.std([values[metric] for values in seed_metrics.values()])),
        }
        for metric in metric_keys
    }

    score_parts: list[pd.DataFrame] = []
    for split, split_frame, split_seed_scores, split_mean_scores in (
        ("dev", val, seed_val_scores, mean_val_scores),
        ("test", test, seed_scores, mean_scores),
    ):
        part = split_frame[["serp_x", "item_id", "block_pos", "_relevance"]].copy()
        part = part.rename(columns={"_relevance": "evaluation_relevance"})
        part.insert(0, "split", split)
        for seed, scores in zip(args.seeds, split_seed_scores):
            part[f"catboost_seed_{seed}"] = scores
        part["catboost_score"] = split_mean_scores
        score_parts.append(part)
    score_artifact = pd.concat(score_parts, ignore_index=True)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    score_artifact.to_parquet(args.scores_output, index=False)

    payload = {
        "claim_scope": "local diagnostic only; not Roma/team CatBoost",
        "protocol": {
            "version": 1,
            "dataset": str(args.items.relative_to(ROOT)),
            "dataset_sha256": dataset_sha256,
            "split": "SERP-disjoint deterministic 80/10/10",
            "split_seed": args.split_seed,
            "candidate_set": "all rows of each fixed SERP; no resampling",
            "relevance": "contacts_daily / max contacts_daily within SERP; train/eval label, never a feature",
            "features": rank_time_feature_columns(),
            "excluded_post_exposure_features": sorted(FORBIDDEN_RANK_TIME_COLUMNS),
            "seeds": args.seeds,
            "catboost_version": catboost_version,
            "model": {
                "loss": "YetiRankPairwise",
                "iterations_cap": args.iterations,
                "depth": args.depth,
                "learning_rate": args.learning_rate,
                "early_stopping_rounds": args.early_stopping_rounds,
                "task_type": args.task_type,
                "gpu_fixed_iteration_budget": args.task_type == "GPU",
            },
        },
        "counts": {
            "train_serps": len(train_ids),
            "val_serps": len(val_ids),
            "test_serps": len(test_ids),
            "train_candidates": len(train),
            "val_candidates": len(val),
            "test_candidates": len(test),
        },
        "best_iterations": best_iterations,
        "position_baseline_dev": position_val_metrics,
        "position_baseline_test": position_metrics,
        "catboost_by_seed_dev": seed_val_metrics,
        "catboost_by_seed_test": seed_metrics,
        "catboost_stability_test": stability,
        "catboost_ensemble_dev": ensemble_val_metrics,
        "catboost_ensemble_test": ensemble_metrics,
        "scores_artifact": str(args.scores_output.relative_to(ROOT)),
        "elapsed_seconds": time.time() - started,
        "blocking_dependency": {
            "team_catboost_metric": "TBD",
            "required_from_roma": [
                "test SERP ids",
                "candidate ids and scores",
                "model/config revision and seed",
                "exact relevance formula",
                "rank-time cutoff for user history",
            ],
        },
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
