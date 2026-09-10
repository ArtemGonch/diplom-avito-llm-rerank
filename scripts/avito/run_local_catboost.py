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
import os
import platform
import subprocess
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

BLOCK_PRIORITY = {
    "items": 0,
    "extra-reg": 1,
    "extra-candidates-extra_regions": 2,
}

# The current frozen ``canonical_v2`` models predate the overwrite/reuse and
# deterministic-evaluation guards below. Keep their complete producing-code
# identity as provenance while allowing this verifier/evaluator revision to
# load byte-identical artifacts. No unknown fingerprint is accepted and every
# non-code metadata field remains subject to exact comparison.
CANONICAL_V2_GENERATOR_CODE_FINGERPRINTS = {
    "runner_sha256": "3cf2ec0a792b8a53c289efcbde52d2577c4b85255fc88d69fde34e95c809353e",
    "dataset_loader_sha256": "615ae224b34cc92732526c60e4edd014e38caabb2ce96898d44b3b43dd4fb6e3",
    "metrics_sha256": "2656be02e6e43e35af30de17d066efe21cf0f6b2afac64551b692c467162556e",
}


def rank_time_feature_columns() -> list[str]:
    features = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    overlap = FORBIDDEN_RANK_TIME_COLUMNS & set(features)
    if overlap:
        raise AssertionError(f"Post-exposure leakage in CatBoost features: {sorted(overlap)}")
    return features


def _prepare_frame(items: pd.DataFrame, serp_names: Sequence[str]) -> pd.DataFrame:
    frame = items[items["serp_x"].isin(set(serp_names))].copy()
    frame = frame.sort_values(["serp_x", "block_pos", "item_id"], kind="stable").reset_index(drop=True)
    frame["_source_block_pos"] = pd.to_numeric(frame["block_pos"], errors="coerce")
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        frame[column] = frame[column].fillna(0.0).astype(np.float32)
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].fillna("<missing>").astype(str)
    relevance = frame["contacts_daily"].clip(lower=0).astype(np.float64)
    maxima = relevance.groupby(frame["serp_x"]).transform("max")
    evaluation_relevance = np.where(maxima > 0, relevance / maxima, 0.0).astype(np.float64)
    # Preserve the exact float32 training-label path used by the frozen
    # canonical-v2 producer. Evaluation uses the separate float64 column in
    # every Avito evaluator, avoiding dtype-dependent NDCG drift.
    frame["_relevance"] = evaluation_relevance.astype(np.float32)
    frame["_evaluation_relevance"] = evaluation_relevance
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
    item_id_groups: list[np.ndarray] = []
    offset = 0
    for _, group in frame.groupby("serp_x", sort=False):
        length = len(group)
        score_groups.append(np.asarray(scores[offset : offset + length], dtype=np.float64))
        label_groups.append(group["_evaluation_relevance"].to_numpy(dtype=np.float64))
        item_id_groups.append(group["item_id"].to_numpy(dtype=np.int64))
        offset += length
    if offset != len(scores):
        raise ValueError("Score count does not match test frame")
    return evaluate_batch(
        score_groups,
        label_groups,
        ks=(5, 10),
        item_ids_list=item_id_groups,
    )


def _source_order_scores(frame: pd.DataFrame) -> np.ndarray:
    """Encode block-aware exported order as unique scores.

    ``block_pos=-1`` means that a supplemental block has no within-block
    position.  It must sort after positioned ``items`` rows, not ahead of
    position zero as the former raw ``-block_pos`` baseline did.
    """
    scores = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    for _, group in frame.groupby("serp_x", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        block_priority = (
            group["block"].map(BLOCK_PRIORITY).fillna(max(BLOCK_PRIORITY.values()) + 1).to_numpy(int)
        )
        source_positions = group.get("_source_block_pos", group["block_pos"])
        positions = pd.to_numeric(source_positions, errors="coerce").fillna(-1).to_numpy(int)
        position_key = np.where(positions >= 0, positions, np.iinfo(np.int32).max)
        item_ids = group["item_id"].to_numpy(dtype=np.int64)
        order = np.lexsort((item_ids, position_key, block_priority))
        scores.loc[indices[order]] = np.arange(len(group), 0, -1, dtype=np.float64)
    if scores.isna().any():
        raise AssertionError("Source-order score assignment is incomplete")
    return scores.to_numpy(dtype=np.float64)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _split_fingerprints(
    train_names: Sequence[str], val_names: Sequence[str], test_names: Sequence[str]
) -> dict[str, object]:
    membership = {
        "train": sorted(map(str, train_names)),
        "val": sorted(map(str, val_names)),
        "test": sorted(map(str, test_names)),
    }
    return {
        "train_sha256": _json_sha256(membership["train"]),
        "val_sha256": _json_sha256(membership["val"]),
        "test_sha256": _json_sha256(membership["test"]),
        "combined_sha256": _json_sha256(membership),
    }


def _execution_environment(task_type: str, devices: str) -> dict[str, object]:
    """Record enough runtime/hardware identity to audit a GPU fit."""
    environment: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "task_type": task_type,
        "catboost_devices_argument": devices,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "nvidia_driver": "unavailable",
        "nvidia_gpus": [],
    }
    if task_type != "GPU":
        return environment
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        records = []
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 4:
                records.append(
                    {
                        "physical_index": fields[0],
                        "uuid": fields[1],
                        "name": fields[2],
                        "driver_version": fields[3],
                    }
                )
        environment["nvidia_gpus"] = records
        if records:
            environment["nvidia_driver"] = records[0]["driver_version"]
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return environment


def _display_path(path: Path) -> str:
    """Use a repo-relative path when possible and preserve valid external paths."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _verify_reusable_model(
    model_path: Path, model_meta_path: Path, expected_model_meta: dict[str, object]
) -> None:
    """Reject legacy, stale or byte-corrupted CatBoost checkpoints."""
    if (
        not model_path.is_file()
        or model_path.is_symlink()
        or not model_meta_path.is_file()
        or model_meta_path.is_symlink()
    ):
        raise ValueError(
            f"Refusing unverified model reuse: missing/non-regular model or sidecar "
            f"({model_path}, {model_meta_path}). "
            "Fetch the frozen artifacts or train into fresh model/score/metric paths."
        )
    actual_model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
    recorded_model_hash = actual_model_meta.pop("model_sha256", None)
    expected = json.loads(json.dumps(expected_model_meta))
    actual_code = actual_model_meta.get("code_fingerprints")
    expected_code = expected.get("code_fingerprints")
    if actual_code != expected_code:
        if actual_code != CANONICAL_V2_GENERATOR_CODE_FINGERPRINTS:
            raise ValueError(
                f"Refusing stale/mismatched model reuse: unknown producing code "
                f"fingerprints in {model_meta_path}"
            )
        # This exact fingerprint triple is the audited canonical-v2 producer.
        # The current changes preserve the producer's float32 training labels;
        # common.metrics is only used after prediction. All training metadata
        # and model bytes remain subject to exact checks below.
        expected["code_fingerprints"] = actual_code
    if actual_model_meta != expected:
        raise ValueError(
            f"Refusing stale/mismatched model reuse: metadata differs at {model_meta_path}"
        )
    actual_model_hash = _file_sha256(model_path)
    if recorded_model_hash != actual_model_hash:
        raise ValueError(
            f"Refusing corrupted model reuse: model SHA-256 differs at {model_path}"
        )


def _preflight_model_outputs(
    *,
    model_dir: Path,
    seeds: Sequence[int],
    scores_output: Path,
    metrics_output: Path,
    reuse_existing_models: bool,
    force_retrain: bool,
) -> None:
    """Fail closed before a fit can mutate a frozen or partially missing run."""
    if reuse_existing_models and force_retrain:
        raise ValueError("--reuse-existing-models and --force-retrain are mutually exclusive")
    pairs = [
        (model_dir / f"seed_{seed}.cbm", model_dir / f"seed_{seed}.meta.json")
        for seed in seeds
    ]
    if reuse_existing_models:
        missing = [
            str(path)
            for pair in pairs
            for path in pair
            if not path.is_file() or path.is_symlink()
        ]
        if missing:
            raise FileNotFoundError(
                "Refusing partial canonical reuse; every regular .cbm and sidecar is required: "
                + ", ".join(missing)
            )
        return
    existing = [
        str(path)
        for path in [
            *(path for pair in pairs for path in pair),
            scores_output,
            metrics_output,
        ]
        if path.exists() or path.is_symlink()
    ]
    if existing and not force_retrain:
        raise FileExistsError(
            "Refusing to overwrite an existing CatBoost run. Use fresh --model-dir, "
            "--scores-output and --output paths, or pass --force-retrain explicitly: "
            + ", ".join(existing)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=ROOT / "items_with_attrs.parquet")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument(
        "--reuse-existing-models",
        action="store_true",
        help=(
            "Reuse seed_<seed>.cbm only when its exact generated .meta.json contract matches; "
            "legacy checkpoints without metadata are deliberately rejected."
        ),
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help=(
            "Explicitly allow a fit to replace existing model/score/metric artifacts. "
            "Prefer fresh output paths; this flag is deliberately never implied."
        ),
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
        default=ROOT / "checkpoints/avito_local_catboost_300_canonical_v2",
    )
    args = parser.parse_args()

    _preflight_model_outputs(
        model_dir=args.model_dir,
        seeds=args.seeds,
        scores_output=args.scores_output,
        metrics_output=args.output,
        reuse_existing_models=args.reuse_existing_models,
        force_retrain=args.force_retrain,
    )

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
    split_fingerprints = _split_fingerprints(train_names, val_names, test_names)
    execution_environment = _execution_environment(args.task_type, args.devices)
    code_fingerprints = {
        "runner_sha256": _file_sha256(Path(__file__)),
        "dataset_loader_sha256": _file_sha256(ROOT / "src/data/avito.py"),
        "metrics_sha256": _file_sha256(ROOT / "src/common/metrics.py"),
    }
    normalized_training_command = [
        "python",
        "scripts/avito/run_local_catboost.py",
        "--split-seed",
        str(args.split_seed),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--iterations",
        str(args.iterations),
        "--depth",
        str(args.depth),
        "--learning-rate",
        str(args.learning_rate),
        "--early-stopping-rounds",
        str(args.early_stopping_rounds),
        "--task-type",
        args.task_type,
        "--devices",
        args.devices,
        "--items",
        _display_path(args.items),
        "--model-dir",
        _display_path(args.model_dir),
        "--scores-output",
        _display_path(args.scores_output),
        "--output",
        _display_path(args.output),
    ]

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
            metric_period=100,
            allow_writing_files=False,
            thread_count=-1,
        )
        if args.task_type == "GPU":
            model_kwargs.update(task_type="GPU", devices=args.devices)
        expected_model_meta = {
            "version": 2,
            "dataset_sha256": dataset_sha256,
            "split_seed": args.split_seed,
            "split_fingerprints": split_fingerprints,
            "split_counts": {
                "train_serps": len(train_ids),
                "val_serps": len(val_ids),
                "test_serps": len(test_ids),
            },
            "features": rank_time_feature_columns(),
            "seed": seed,
            "catboost_version": catboost_version,
            "code_fingerprints": code_fingerprints,
            "execution_environment": execution_environment,
            "normalized_training_command": normalized_training_command,
            "training": {
                "iterations": args.iterations,
                "depth": args.depth,
                "learning_rate": args.learning_rate,
                "loss_function": "YetiRankPairwise",
                "l2_leaf_reg": 5.0,
                "random_strength": 0.5,
                "task_type": args.task_type,
                "devices": args.devices,
                "metric_period": 100,
                "early_stopping_rounds": (
                    None if args.task_type == "GPU" else args.early_stopping_rounds
                ),
            },
        }
        model = CatBoostRanker(**model_kwargs)
        reused = False
        if args.reuse_existing_models:
            _verify_reusable_model(model_path, model_meta_path, expected_model_meta)
            print(f"Reusing {model_path}", flush=True)
            model.load_model(model_path)
            reused = True
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
        if not reused:
            model.save_model(model_path)
            _write_json(
                model_meta_path,
                {**expected_model_meta, "model_sha256": _file_sha256(model_path)},
            )
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
    source_val_metrics = _evaluate_frame(val, _source_order_scores(val))
    source_metrics = _evaluate_frame(test, _source_order_scores(test))
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
        part = split_frame[
            ["serp_x", "item_id", "block_pos", "_evaluation_relevance"]
        ].copy()
        part = part.rename(columns={"_evaluation_relevance": "evaluation_relevance"})
        part.insert(0, "split", split)
        for seed, scores in zip(args.seeds, split_seed_scores):
            part[f"catboost_seed_{seed}"] = scores
        part["catboost_score"] = split_mean_scores
        score_parts.append(part)
    score_artifact = pd.concat(score_parts, ignore_index=True)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    score_artifact.to_parquet(args.scores_output, index=False)
    score_artifact_sha256 = _file_sha256(args.scores_output)
    model_artifacts = []
    for seed in args.seeds:
        model_path = args.model_dir / f"seed_{seed}.cbm"
        meta_path = args.model_dir / f"seed_{seed}.meta.json"
        stored_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_artifacts.append(
            {
                "seed": seed,
                "model": _display_path(model_path),
                "model_sha256": _file_sha256(model_path),
                "metadata": _display_path(meta_path),
                "metadata_sha256": _file_sha256(meta_path),
                "generator_runner_sha256": stored_meta.get("code_fingerprints", {}).get(
                    "runner_sha256"
                ),
                "generator_code_fingerprints": stored_meta.get("code_fingerprints"),
            }
        )

    payload = {
        "claim_scope": "local diagnostic only; not Roma/team CatBoost",
        "protocol": {
            "version": 3,
            "dataset": _display_path(args.items),
            "dataset_sha256": dataset_sha256,
            "split": "SERP-disjoint deterministic 80/10/10",
            "split_seed": args.split_seed,
            "split_fingerprints": split_fingerprints,
            "candidate_set": "all rows of each fixed SERP; no resampling",
            "source_order": "block priority, valid block_pos ascending, item_id ascending",
            "score_tie_break": "score descending, item_id ascending",
            "training_label": "contacts_daily / max contacts_daily within SERP; float32 as frozen producer; never a feature",
            "evaluation_relevance": "contacts_daily / max contacts_daily within SERP; float64; never a feature",
            "evaluation_relevance_dtype": "float64",
            "features": rank_time_feature_columns(),
            "excluded_post_exposure_features": sorted(FORBIDDEN_RANK_TIME_COLUMNS),
            "seeds": args.seeds,
            "catboost_version": catboost_version,
            "code_fingerprints": code_fingerprints,
            "frozen_artifact_generator_runner_sha256": sorted(
                {
                    str(artifact["generator_runner_sha256"])
                    for artifact in model_artifacts
                }
            ),
            "frozen_artifact_generator_code_fingerprints": [
                json.loads(value)
                for value in sorted(
                    {
                        json.dumps(
                            artifact["generator_code_fingerprints"],
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        for artifact in model_artifacts
                    }
                )
            ],
            "current_verifier_runner_sha256": code_fingerprints["runner_sha256"],
            "execution_environment": execution_environment,
            "normalized_training_command": normalized_training_command,
            "model": {
                "loss": "YetiRankPairwise",
                "iterations_cap": args.iterations,
                "depth": args.depth,
                "learning_rate": args.learning_rate,
                "early_stopping_rounds": args.early_stopping_rounds,
                "task_type": args.task_type,
                "gpu_fixed_iteration_budget": args.task_type == "GPU",
                "gpu_fit_reproducibility": (
                    "CatBoost GPU fitting is not claimed bitwise deterministic; exact "
                    "evaluation is frozen by immutable model and score SHA-256 artifacts."
                    if args.task_type == "GPU"
                    else "CPU fit under the recorded environment"
                ),
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
        "source_order_baseline_dev": source_val_metrics,
        "source_order_baseline_test": source_metrics,
        "legacy_position_baseline": {
            "valid": False,
            "reason": "raw -block_pos ranked sentinel -1 rows ahead of positioned items; removed",
        },
        "catboost_by_seed_dev": seed_val_metrics,
        "catboost_by_seed_test": seed_metrics,
        "catboost_stability_test": stability,
        "catboost_ensemble_dev": ensemble_val_metrics,
        "catboost_ensemble_test": ensemble_metrics,
        "scores_artifact": _display_path(args.scores_output),
        "scores_artifact_sha256": score_artifact_sha256,
        "model_artifacts": model_artifacts,
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
