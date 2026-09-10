"""Build a leakage-audited, self-contained browser for Avito ranking cases.

The browser is an error-analysis artifact, not a new online ranker.  It keeps
three concepts deliberately separate:

* ``contacts_daily`` is the graded benchmark relevance used by the local
  CatBoost/Qwen diagnostic;
* ``has_tc_events`` is a post-exposure marker shown only as an observed outcome;
* contact history has unknown temporal relation to a SERP because the export
  contains no SERP rank timestamp.

No post-exposure value is included in an LLM insight prompt.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


BROWSER_VERSION = 5
INSIGHT_PROMPT_VERSION = 3
INSIGHT_VALIDATION_VERSION = 9

MAX_OVERLAY_IMAGE_BYTES = 20 * 1024 * 1024
MAX_OVERLAY_IMAGE_PIXELS = 40_000_000
MAX_OVERLAY_IMAGE_FRAMES = 32
MAX_OVERLAY_IMAGE_TOTAL_PIXELS = 40_000_000
ALLOWED_IMAGE_FORMATS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "GIF": ".gif",
}

SCORE_KEYS = ["split", "serp_x", "item_id"]
REQUIRED_SCORE_COLUMNS = {
    *SCORE_KEYS,
    "evaluation_relevance",
    "catboost_score",
    "llm_score",
    "gated_blend_score",
    "llm_called",
}
REQUIRED_ITEM_COLUMNS = {
    "serp_x",
    "item_id",
    "block",
    "block_pos",
    "query_infm_logical_category",
    "query_loc",
    "user_id",
    "contacts_daily",
    "has_tc_events",
    "has_x_events",
    "title",
    "description_short",
    "brand",
    "model_name",
    "price",
    "mileage_km",
    "year_text",
    "body_type",
    "fuel_text",
    "gearbox_text",
    "drive_text",
    "doors_text",
    "image_count",
}
REQUIRED_HISTORY_COLUMNS = {
    "user_id",
    "item_id",
    "contact_date",
    "brand",
    "model_name",
}

POST_EXPOSURE_FIELDS = {
    "contacts_daily",
    "clicks_daily",
    "has_tc_events",
    "has_x_events",
    "serp_is_positive",
    "evaluation_relevance",
}

BLOCK_PRIORITY = {
    "items": 0,
    "extra-reg": 1,
    "extra-candidates-extra_regions": 2,
}

OVERLAY_ALIASES = {
    "description": "description_short",
    "description_text": "description_short",
    "image": "image_path",
    "image_local_path": "image_path",
}
OVERLAY_COLUMNS = ["item_id", "title", "description_short", "image_path", "image_url"]


def _missing_columns(frame: pd.DataFrame, required: set[str]) -> list[str]:
    return sorted(required - set(frame.columns))


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = _missing_columns(frame, required)
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clean_text(value: Any, limit: int = 400) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = " ".join(str(value).split())
    if not text:
        return None
    return text[:limit]


def normalize_user_id(value: Any) -> str | None:
    """Normalise parquet int/float user identifiers without a trailing ``.0``."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None
        if not number.is_integer():
            raise ValueError(f"user_id must be integer-like, got {value!r}")
        return str(int(number))
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    raise ValueError(f"user_id must be integer-like, got {value!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_order(scores: Sequence[float], item_ids: Sequence[int]) -> np.ndarray:
    """Return descending-score order with explicit ascending item-id ties."""
    score_array = np.asarray(scores, dtype=np.float64)
    id_array = np.asarray(item_ids, dtype=np.int64)
    if score_array.shape != id_array.shape:
        raise ValueError("scores and item_ids must have the same shape")
    safe_scores = np.where(np.isfinite(score_array), score_array, -np.inf)
    return np.lexsort((id_array, -safe_scores))


def _rank_positions(order: np.ndarray) -> np.ndarray:
    positions = np.empty(len(order), dtype=np.int64)
    positions[order] = np.arange(1, len(order) + 1, dtype=np.int64)
    return positions


def _dcg(relevance: np.ndarray, k: int) -> float:
    rel = np.asarray(relevance, dtype=np.float64)[:k]
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2, dtype=np.float64))
    return float(np.sum((np.power(2.0, rel) - 1.0) / discounts))


def _ndcg_for_order(relevance: np.ndarray, order: np.ndarray, k: int = 10) -> float:
    ideal = _dcg(np.sort(np.asarray(relevance, dtype=np.float64))[::-1], k)
    return _dcg(np.asarray(relevance, dtype=np.float64)[order], k) / ideal if ideal > 0 else 0.0


def _source_display_order(group: pd.DataFrame) -> np.ndarray:
    block_priority = (
        group["block"].map(BLOCK_PRIORITY).fillna(max(BLOCK_PRIORITY.values()) + 1).to_numpy(int)
    )
    positions = pd.to_numeric(group["block_pos"], errors="coerce").fillna(-1).to_numpy(int)
    position_key = np.where(positions >= 0, positions, np.iinfo(np.int32).max)
    item_ids = group["item_id"].to_numpy(np.int64)
    return np.lexsort((item_ids, position_key, block_priority))


def _actual_event_metrics(events: np.ndarray, order: np.ndarray, k: int = 10) -> dict[str, Any]:
    binary = np.asarray(events, dtype=np.float64)
    positive_positions = np.flatnonzero(binary[order] > 0)
    if not len(positive_positions):
        return {"eligible": False, "first_rank": None, "hit@10": None, "mrr@10": None, "ndcg@10": None}
    first_rank = int(positive_positions[0] + 1)
    return {
        "eligible": True,
        "first_rank": first_rank,
        "hit@10": float(first_rank <= k),
        "mrr@10": 1.0 / first_rank if first_rank <= k else 0.0,
        "ndcg@10": _ndcg_for_order(binary, order, k),
    }


def load_item_overlay(path: Path | None) -> pd.DataFrame:
    """Load optional Roma-provided title/description/image metadata."""
    if path is None:
        return pd.DataFrame(columns=OVERLAY_COLUMNS)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Item overlay does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".jsonl":
        frame = pd.read_json(path, lines=True)
    elif suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(raw if isinstance(raw, list) else raw.get("items", []))
    else:
        raise ValueError(f"Unsupported overlay format: {suffix}")
    frame = frame.rename(columns={key: value for key, value in OVERLAY_ALIASES.items() if key in frame})
    if "item_id" not in frame:
        raise ValueError("Item overlay requires item_id")
    frame = frame.copy()
    frame["item_id"] = pd.to_numeric(frame["item_id"], errors="raise").astype(np.int64)
    if frame.duplicated("item_id").any():
        duplicates = frame.loc[frame.duplicated("item_id", keep=False), "item_id"].unique()[:5]
        raise ValueError(f"Item overlay contains duplicate item_id values: {duplicates.tolist()}")
    for column in OVERLAY_COLUMNS:
        if column not in frame:
            frame[column] = None
    overlay_root = path.parent.resolve()
    resolved_image_paths: list[str | None] = []
    for value in frame["image_path"]:
        raw_path = _clean_text(value, 2000)
        if not raw_path or raw_path.lower().startswith(("http://", "https://", "data:image/")):
            resolved_image_paths.append(raw_path)
            continue
        image_path = Path(raw_path).expanduser()
        if not image_path.is_absolute():
            image_path = overlay_root / image_path
        resolved = image_path.resolve()
        try:
            resolved.relative_to(overlay_root)
        except ValueError as exc:
            raise ValueError(
                f"Local image_path must stay below the overlay directory {overlay_root}: "
                f"{raw_path!r}"
            ) from exc
        resolved_image_paths.append(str(resolved))
    frame["image_path"] = resolved_image_paths
    return frame[OVERLAY_COLUMNS]


def _balanced_json_slice(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object from plain or fenced LLM output."""
    for start, char in enumerate(text):
        if char != "{":
            continue
        candidate = _balanced_json_slice(text, start)
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def load_insight_results(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid insight JSONL at line {line_number}: {exc}") from exc
        request_id = str(row.get("request_id", ""))
        if not request_id:
            raise ValueError(f"Insight row {line_number} lacks request_id")
        if request_id in results:
            raise ValueError(f"Duplicate insight request_id: {request_id}")
        results[request_id] = row
    return results


def _overlay_frame(frame: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    renamed = overlay.rename(
        columns={column: f"overlay_{column}" for column in OVERLAY_COLUMNS if column != "item_id"}
    )
    return frame.merge(renamed, on="item_id", how="left", validate="many_to_one", sort=False)


def _prefer_overlay(row: Mapping[str, Any], column: str, fallback: str | None = None) -> str | None:
    overlay = _clean_text(row.get(f"overlay_{column}"), 4000)
    if overlay:
        return overlay
    return _clean_text(row.get(column), 4000) or fallback


def _image_reference(row: Mapping[str, Any]) -> str | None:
    url = _clean_text(row.get("overlay_image_url"), 2000)
    if url and url.lower().startswith(("https://", "http://", "data:image/")):
        return url
    path = _clean_text(row.get("overlay_image_path"), 2000)
    if path and not path.lower().startswith(("javascript:", "data:text/")):
        return path
    return None


def _prepare_inputs(
    items: pd.DataFrame,
    history: pd.DataFrame,
    scores: pd.DataFrame,
    split: str,
    overlay: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _require_columns(items, REQUIRED_ITEM_COLUMNS, "items")
    _require_columns(history, REQUIRED_HISTORY_COLUMNS, "history")
    _require_columns(scores, REQUIRED_SCORE_COLUMNS, "scores")

    score_frame = scores.loc[scores["split"].astype(str).eq(split)].copy()
    if score_frame.empty:
        raise ValueError(f"Score artifact contains no rows for split={split!r}")
    score_frame["split"] = split
    score_frame["serp_x"] = score_frame["serp_x"].astype(str)
    score_frame["item_id"] = pd.to_numeric(score_frame["item_id"], errors="raise").astype(np.int64)
    if score_frame.duplicated(SCORE_KEYS).any():
        raise ValueError("Score artifact contains duplicate split/serp_x/item_id keys")

    item_frame = items.copy()
    item_frame["serp_x"] = item_frame["serp_x"].astype(str)
    item_frame["item_id"] = pd.to_numeric(item_frame["item_id"], errors="raise").astype(np.int64)
    item_frame = item_frame[item_frame["serp_x"].isin(set(score_frame["serp_x"]))].copy()
    item_keys = ["serp_x", "item_id"]
    if item_frame.duplicated(item_keys).any():
        examples = item_frame.loc[item_frame.duplicated(item_keys, keep=False), item_keys].head(5)
        raise ValueError(f"Items contain duplicate candidates in selected split: {examples.to_dict('records')}")

    key_audit = score_frame[SCORE_KEYS].merge(
        item_frame[item_keys].assign(split=split)[SCORE_KEYS],
        on=SCORE_KEYS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    counts = {str(key): int(value) for key, value in key_audit["_merge"].value_counts().items()}
    if counts.get("left_only", 0) or counts.get("right_only", 0):
        raise ValueError(
            "Items and score artifact candidate sets differ: "
            f"missing_items={counts.get('left_only', 0)}, missing_scores={counts.get('right_only', 0)}"
        )

    frame = score_frame.merge(item_frame, on=item_keys, how="left", validate="one_to_one", sort=False)
    contacts = (
        pd.to_numeric(frame["contacts_daily"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .astype(np.float64)
    )
    maxima = contacts.groupby(frame["serp_x"]).transform("max")
    computed_relevance = np.where(maxima > 0, contacts / maxima, 0.0)
    artifact_relevance = pd.to_numeric(frame["evaluation_relevance"], errors="raise").to_numpy(float)
    max_relevance_difference = float(np.max(np.abs(computed_relevance - artifact_relevance)))
    if max_relevance_difference > 1e-6:
        raise ValueError(
            "Score relevance does not match contacts_daily/max-within-SERP; "
            f"max_abs_difference={max_relevance_difference}"
        )
    frame["evaluation_relevance"] = computed_relevance.astype(np.float64)
    frame["user_key"] = [normalize_user_id(value) for value in frame["user_id"]]
    users_per_serp = frame.groupby("serp_x")["user_key"].nunique(dropna=True)
    if bool((users_per_serp > 1).any()):
        raise ValueError("At least one SERP maps to multiple non-null user_id values")
    frame = _overlay_frame(frame, overlay)

    history_frame = history.copy()
    history_frame["item_id"] = pd.to_numeric(history_frame["item_id"], errors="raise").astype(np.int64)
    history_frame["user_key"] = [normalize_user_id(value) for value in history_frame["user_id"]]
    selected_users = set(frame["user_key"].dropna())
    history_frame = history_frame[history_frame["user_key"].isin(selected_users)].copy()
    history_frame["contact_date"] = pd.to_datetime(history_frame["contact_date"], errors="coerce")
    if history_frame["contact_date"].isna().any():
        raise ValueError("History contains unparseable contact_date values for selected users")
    history_frame = _overlay_frame(history_frame, overlay)
    history_frame = history_frame.sort_values(
        ["user_key", "contact_date", "item_id"], ascending=[True, False, True], kind="stable"
    )

    audit = {
        "candidate_key_counts": counts,
        "max_relevance_abs_difference": max_relevance_difference,
        "score_rows": int(len(score_frame)),
        "serps": int(score_frame["serp_x"].nunique()),
        "users_with_exported_history": int(
            frame.groupby("serp_x")["user_key"].first().isin(set(history_frame["user_key"])).sum()
        ),
        "history_rows_for_split_users": int(len(history_frame)),
        "rank_time_available": False,
    }
    return frame, history_frame, audit


def _history_records(group: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in group.to_dict("records"):
        title = _prefer_overlay(row, "title")
        description = _prefer_overlay(row, "description_short")
        records.append(
            {
                "item_id": int(row["item_id"]),
                "contact_date": row["contact_date"].date().isoformat(),
                "raw_taxonomy_level_1": _clean_text(row.get("brand"), 160),
                "raw_taxonomy_level_2": _clean_text(row.get("model_name"), 160),
                "title": title,
                "description": description,
                "image": _image_reference(row),
                "metadata_available": bool(title or description or _image_reference(row)),
                "temporal_status": "unknown_no_serp_rank_time",
            }
        )
    return records


def _history_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = Counter(record.get("raw_taxonomy_level_1") for record in records if record.get("raw_taxonomy_level_1"))
    second = Counter(record.get("raw_taxonomy_level_2") for record in records if record.get("raw_taxonomy_level_2"))
    dates = sorted(record["contact_date"] for record in records)
    return {
        "count": len(records),
        "date_from": dates[0] if dates else None,
        "date_to": dates[-1] if dates else None,
        "top_raw_taxonomy_level_1": [[key, count] for key, count in first.most_common(5)],
        "top_raw_taxonomy_level_2": [[key, count] for key, count in second.most_common(5)],
        "metadata_coverage": (
            sum(bool(record["metadata_available"]) for record in records) / len(records)
            if records
            else 0.0
        ),
        "warning": (
            "Экспортированные brand/model_name истории являются raw category-like taxonomy, "
            "а не маркой/моделью автомобиля; временной cutoff относительно SERP неизвестен."
        ),
    }


def _safe_insight(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return None
    parsed = result.get("parsed")
    return {
        "status": str(result.get("status", "unknown")),
        "parsed": parsed if isinstance(parsed, dict) else None,
        "model": _clean_text(result.get("model"), 200),
        "model_revision": _clean_text(result.get("model_revision"), 200),
        "prompt_sha256": _clean_text(result.get("prompt_sha256"), 128),
        "llm_output_accepted": bool(result.get("llm_output_accepted", True)),
        "fallback_applied": bool(result.get("fallback_applied", False)),
        "generation_pass": _clean_text(result.get("generation_pass"), 100),
        "evidence_normalized_fields": list(result.get("evidence_normalized_fields", [])),
    }


def _candidate_record(
    row: Mapping[str, Any],
    ranks: Mapping[str, int],
    insight_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    item_id = int(row["item_id"])
    title = _prefer_overlay(row, "title", f"Объявление {item_id}")
    description = _prefer_overlay(row, "description_short")
    return {
        "item_id": item_id,
        "title": title,
        "description": description,
        "image": _image_reference(row),
        "image_count_reported": _finite_float(row.get("image_count")),
        "source": {
            "block": _clean_text(row.get("block"), 100),
            "block_pos": int(row["block_pos"]) if _finite_float(row.get("block_pos")) is not None else None,
        },
        "attributes": {
            "brand": _clean_text(row.get("brand"), 100),
            "model": _clean_text(row.get("model_name"), 140),
            "price": _finite_float(row.get("price")),
            "year": _clean_text(row.get("year_text"), 80),
            "mileage_km": _finite_float(row.get("mileage_km")),
            "body": _clean_text(row.get("body_type"), 80),
            "fuel": _clean_text(row.get("fuel_text"), 80),
            "gearbox": _clean_text(row.get("gearbox_text"), 80),
            "drive": _clean_text(row.get("drive_text"), 80),
            "doors": _clean_text(row.get("doors_text"), 40),
        },
        "scores": {
            "catboost": _finite_float(row.get("catboost_score")),
            "qwen_l0": _finite_float(row.get("llm_score")),
            "gated_blend": _finite_float(row.get("gated_blend_score")),
        },
        "ranks": {key: int(value) for key, value in ranks.items()},
        "outcomes": {
            "actual_tc_event": bool(int(row.get("has_tc_events", 0))),
            "other_x_event": bool(int(row.get("has_x_events", 0))),
            "contacts_daily": _finite_float(row.get("contacts_daily")),
            "benchmark_relevance": _finite_float(row.get("evaluation_relevance")),
            "usage": "display/evaluation only; never an LLM or CatBoost rank-time feature",
        },
        "llm_called_by_rejected_gate": bool(row.get("llm_called", False)),
        "insight": _safe_insight(insight_results.get(f"item:{item_id}")),
    }


def _case_orders(group: pd.DataFrame) -> dict[str, np.ndarray]:
    item_ids = group["item_id"].to_numpy(np.int64)
    return {
        "source": _source_display_order(group),
        "catboost": deterministic_order(group["catboost_score"].to_numpy(float), item_ids),
        "qwen_l0": deterministic_order(group["llm_score"].to_numpy(float), item_ids),
        "gated_blend": deterministic_order(group["gated_blend_score"].to_numpy(float), item_ids),
    }


def _case_tags(case: Mapping[str, Any]) -> list[str]:
    metrics = case["metrics"]
    history_count = int(case["history_summary"]["count"])
    delta_llm = float(metrics["delta_qwen_vs_catboost_ndcg@10"])
    delta_gate = float(metrics["delta_gate_vs_catboost_ndcg@10"])
    tags = ["with_history" if history_count else "cold_start"]
    tags.append("actual_tc_event" if metrics["actual_tc_event_count"] else "no_actual_tc_event")
    tags.append("qwen_beats_catboost" if delta_llm > 1e-12 else "qwen_below_catboost")
    tags.append("gate_helps" if delta_gate > 1e-12 else ("gate_hurts" if delta_gate < -1e-12 else "gate_unchanged"))
    if history_count >= 10:
        tags.append("long_history")
    actual = metrics["actual_tc"]
    if actual["catboost"]["eligible"]:
        cb_rank = actual["catboost"]["first_rank"]
        llm_rank = actual["qwen_l0"]["first_rank"]
        if cb_rank is not None and llm_rank is not None:
            outcome_prefers_llm = llm_rank < cb_rank
            benchmark_prefers_llm = delta_llm > 0
            if outcome_prefers_llm != benchmark_prefers_llm:
                tags.append("target_disagreement")
    return tags


def _build_cases(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    insight_results: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    history_by_user = {
        str(user_key): _history_records(group)
        for user_key, group in history.groupby("user_key", sort=False)
    }
    cases: list[dict[str, Any]] = []
    for serp_x, group in frame.groupby("serp_x", sort=True):
        group = group.reset_index(drop=True)
        item_ids = group["item_id"].to_numpy(np.int64)
        relevance = group["evaluation_relevance"].to_numpy(float)
        actual_events = group["has_tc_events"].to_numpy(int)
        orders = _case_orders(group)
        ranks_by_method = {name: _rank_positions(order) for name, order in orders.items()}
        user_values = [
            normalised
            for value in group["user_key"].tolist()
            if (normalised := normalize_user_id(value)) is not None
        ]
        user_key = str(user_values[0]) if user_values else None
        history_records = history_by_user.get(user_key or "", [])
        graded = {
            name: _ndcg_for_order(relevance, order, 10)
            for name, order in orders.items()
        }
        actual = {
            name: _actual_event_metrics(actual_events, order, 10)
            for name, order in orders.items()
        }
        metrics = {
            "graded_ndcg@10": graded,
            "delta_qwen_vs_catboost_ndcg@10": graded["qwen_l0"] - graded["catboost"],
            "delta_gate_vs_catboost_ndcg@10": graded["gated_blend"] - graded["catboost"],
            "actual_tc_event_count": int(np.sum(actual_events > 0)),
            "actual_tc": actual,
        }
        candidates: list[dict[str, Any]] = []
        for index, row in enumerate(group.to_dict("records")):
            candidate_ranks = {name: int(values[index]) for name, values in ranks_by_method.items()}
            candidates.append(_candidate_record(row, candidate_ranks, insight_results))
        first = group.iloc[0]
        case: dict[str, Any] = {
            "serp_x": str(serp_x),
            "user_id": user_key,
            "query": {
                "category_proxy": _clean_text(first.get("query_infm_logical_category"), 160),
                "location_id": _clean_text(first.get("query_loc"), 80),
                "limitation": "Полный текст запроса и фильтры в snapshot отсутствуют.",
            },
            "candidate_count": int(len(group)),
            "history": history_records,
            "history_summary": _history_summary(history_records),
            "user_insight": _safe_insight(insight_results.get(f"user:{user_key}")) if user_key else None,
            "metrics": metrics,
            "candidates": candidates,
            "featured_reasons": [],
        }
        case["tags"] = _case_tags(case)
        cases.append(case)
    return cases


def _choose_featured(cases: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = []

    def choose(
        reason: str,
        predicate: Any,
        key: Any,
        reverse: bool,
        description: Any,
    ) -> None:
        eligible = [case for case in cases if predicate(case) and case["serp_x"] not in selected]
        if not eligible:
            return
        eligible.sort(key=lambda case: (key(case), case["serp_x"]), reverse=reverse)
        chosen = eligible[0]
        chosen["featured_reasons"].append({"id": reason, "description": description(chosen)})
        chosen["tags"].append("featured")
        selected.append(chosen["serp_x"])

    llm_delta = lambda case: case["metrics"]["delta_qwen_vs_catboost_ndcg@10"]
    gate_delta = lambda case: case["metrics"]["delta_gate_vs_catboost_ndcg@10"]
    has_history = lambda case: case["history_summary"]["count"] > 0
    choose(
        "largest_qwen_gain",
        lambda case: llm_delta(case) > 0,
        llm_delta,
        True,
        lambda case: f"Максимальный выигрыш Qwen L0 над CatBoost по NDCG@10: {llm_delta(case):+.3f}.",
    )
    choose(
        "largest_qwen_failure",
        lambda case: llm_delta(case) < 0,
        llm_delta,
        False,
        lambda case: f"Максимальный проигрыш Qwen L0 CatBoost по NDCG@10: {llm_delta(case):+.3f}.",
    )
    choose(
        "history_qwen_gain",
        lambda case: has_history(case) and llm_delta(case) > 0,
        llm_delta,
        True,
        lambda case: (
            f"Лучший Qwen-vs-CatBoost кейс среди SERP с экспортированной историей "
            f"({case['history_summary']['count']} событий): {llm_delta(case):+.3f}."
        ),
    )
    choose(
        "history_qwen_failure",
        lambda case: has_history(case) and llm_delta(case) < 0,
        llm_delta,
        False,
        lambda case: (
            f"Худший Qwen-vs-CatBoost кейс при наличии истории "
            f"({case['history_summary']['count']} событий): {llm_delta(case):+.3f}."
        ),
    )
    choose(
        "longest_history",
        has_history,
        lambda case: case["history_summary"]["count"],
        True,
        lambda case: f"Самая длинная доступная история среди оставшихся кейсов: {case['history_summary']['count']} событий.",
    )
    choose(
        "largest_gate_gain",
        lambda case: gate_delta(case) > 0,
        gate_delta,
        True,
        lambda case: f"Наибольший локальный выигрыш отклонённого gate: {gate_delta(case):+.3f} NDCG@10.",
    )
    choose(
        "largest_gate_failure",
        lambda case: gate_delta(case) < 0,
        gate_delta,
        False,
        lambda case: f"Наибольший локальный вред отклонённого gate: {gate_delta(case):+.3f} NDCG@10.",
    )
    choose(
        "target_disagreement",
        lambda case: "target_disagreement" in case["tags"],
        lambda case: abs(llm_delta(case)),
        True,
        lambda case: (
            "Направление разницы по graded contacts_daily расходится с тем, "
            "какая модель выше поставила фактический TC-event."
        ),
    )
    return selected


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def _aggregate_summary(cases: Sequence[Mapping[str, Any]], frame: pd.DataFrame) -> dict[str, Any]:
    methods = ["source", "catboost", "qwen_l0", "gated_blend"]
    graded = {
        method: _mean(case["metrics"]["graded_ndcg@10"][method] for case in cases)
        for method in methods
    }
    actual: dict[str, dict[str, Any]] = {}
    for method in methods:
        rows = [case["metrics"]["actual_tc"][method] for case in cases]
        eligible = [row for row in rows if row["eligible"]]
        actual[method] = {
            "eligible_serps": len(eligible),
            "hit@10": _mean(row["hit@10"] for row in eligible),
            "mrr@10": _mean(row["mrr@10"] for row in eligible),
            "ndcg@10": _mean(row["ndcg@10"] for row in eligible),
        }
    correlations = frame[["contacts_daily", "has_tc_events"]].astype(float).corr()
    score_ties = {}
    for method, column in {
        "catboost": "catboost_score",
        "qwen_l0": "llm_score",
        "gated_blend": "gated_blend_score",
    }.items():
        duplicate_counts = [
            len(group) - group[column].nunique(dropna=False)
            for _, group in frame.groupby("serp_x", sort=False)
        ]
        score_ties[method] = {
            "serps_with_ties": int(sum(count > 0 for count in duplicate_counts)),
            "duplicate_score_entries": int(sum(duplicate_counts)),
        }
    return {
        "counts": {
            "serps": len(cases),
            "candidates": int(len(frame)),
            "serps_with_history": int(sum(case["history_summary"]["count"] > 0 for case in cases)),
            "history_rows": int(sum(case["history_summary"]["count"] for case in cases)),
            "serps_with_actual_tc_event": int(sum(case["metrics"]["actual_tc_event_count"] > 0 for case in cases)),
            "actual_tc_event_rows": int(sum(case["metrics"]["actual_tc_event_count"] for case in cases)),
            "qwen_beats_catboost_serps": int(
                sum(case["metrics"]["delta_qwen_vs_catboost_ndcg@10"] > 1e-12 for case in cases)
            ),
            "gate_beats_catboost_serps": int(
                sum(case["metrics"]["delta_gate_vs_catboost_ndcg@10"] > 1e-12 for case in cases)
            ),
        },
        "graded_contacts_daily": {
            "metric": "NDCG@10, explicit item_id ascending score-tie break",
            "values": graded,
        },
        "actual_tc_event_diagnostic": {
            "metric_scope": "only SERPs with at least one has_tc_events=1; descriptive, not the fixed benchmark target",
            "values": actual,
        },
        "contacts_daily_vs_actual_tc_row_correlation": float(
            correlations.loc["contacts_daily", "has_tc_events"]
        ),
        "score_tie_audit": score_ties,
    }


def _source_fingerprints(paths: Mapping[str, Path] | None) -> dict[str, Any]:
    if not paths:
        return {}
    result = {}
    for name, path in paths.items():
        resolved = Path(path)
        try:
            display_path = str(resolved.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            display_path = resolved.name
        result[name] = {
            "path": display_path,
            "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size,
        }
    return result


def build_browser_payload(
    items: pd.DataFrame,
    history: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    split: str = "test",
    overlay: pd.DataFrame | None = None,
    insight_results: Mapping[str, Mapping[str, Any]] | None = None,
    source_paths: Mapping[str, Path] | None = None,
    generated_at: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Return browser payload plus prepared candidate/history frames."""
    overlay = overlay if overlay is not None else load_item_overlay(None)
    insight_results = insight_results or {}
    frame, history_frame, audit = _prepare_inputs(items, history, scores, split, overlay)
    cases = _build_cases(frame, history_frame, insight_results)
    featured = _choose_featured(cases)
    insight_ok = [result for result in insight_results.values() if result.get("status") == "ok"]
    insight_fallback = [result for result in insight_ok if result.get("fallback_applied")]
    payload = {
        "meta": {
            "browser_version": BROWSER_VERSION,
            "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
            "split": split,
            "claim_scope": (
                "offline local diagnostic browser; Qwen L0 has no history; CatBoost is not Roma/team A1"
            ),
            "source_fingerprints": _source_fingerprints(source_paths),
            "audit": audit,
            "insight_results": {
                "loaded": len(insight_results),
                "ok": len(insight_ok),
                "failed": len(insight_results) - len(insight_ok),
                "direct_llm_accepted": len(insight_ok) - len(insight_fallback),
                "guarded_fallback": len(insight_fallback),
            },
            "limitations": [
                "SERP rank_time is absent, so exported history cannot be asserted to precede the displayed SERP.",
                "History brand/model_name fields are raw category-like taxonomy, not automotive brand/model.",
                "Qwen L0 ranking is no-history and no-position; user insights are descriptive and do not change its scores.",
                "CatBoost scores are a local diagnostic, not the missing Roma/team baseline.",
                "contacts_daily is benchmark relevance; has_tc_events is displayed separately as a post-exposure outcome.",
                "The Avito snapshot contains image_count but no image files or URLs; images require an optional overlay.",
            ],
        },
        "summary": _aggregate_summary(cases, frame),
        "featured_serps": featured,
        "cases": cases,
    }
    return payload, frame, history_frame


def _item_prompt(candidate: Mapping[str, Any]) -> str:
    source = {
        "structured_attributes": {
            "title": candidate.get("title"),
            **dict(candidate.get("attributes") or {}),
        },
        "seller_description": candidate.get("description"),
    }
    return (
        "Ты анализируешь автомобильное объявление для offline error analysis. "
        "Используй только вход ниже. structured_attributes считай наблюдаемыми полями экспорта. "
        "seller_description — рекламный текст продавца, а не проверенные факты. "
        "Не придумывай комплектацию, состояние, владельцев, ДТП или мотивы покупателя. "
        "summary сделай одним коротким предложением ТОЛЬКО из structured_attributes, без оценки состояния. "
        "observed_facts — максимум 4 точных структурных факта. seller_claims — максимум 3 коротких "
        "дословных фрагмента из seller_description без перефразирования; каждый записывай как "
        "«Продавец заявляет: «<точный фрагмент>»». Если такого фрагмента нет, верни пустой массив. "
        "risks_to_verify, suitable_for "
        "и limitations — максимум по 3 коротких пункта. "
        "Верни только JSON с ключами summary (строка), observed_facts (массив строк), "
        "seller_claims (массив строк), risks_to_verify (массив строк), suitable_for (массив строк), "
        "limitations (массив строк), confidence (low|medium|high). Значения пиши по-русски.\n"
        f"Наблюдаемые данные объявления:\n{json.dumps(source, ensure_ascii=False, allow_nan=False)}"
    )


def _user_prompt(user_id: str, history: Sequence[Mapping[str, Any]]) -> str:
    safe_history = [
        {
            "contact_date": row.get("contact_date"),
            "raw_category_level_1": row.get("raw_taxonomy_level_1"),
            "raw_category_level_2": row.get("raw_taxonomy_level_2"),
            "title": row.get("title"),
            "description": row.get("description"),
        }
        for row in history
    ]
    dates = sorted(str(row.get("contact_date")) for row in history if row.get("contact_date"))
    period = f"{dates[0]} — {dates[-1]}" if dates else "неизвестен"
    return (
        "Ты строишь осторожный описательный профиль по экспортированной истории контактов. "
        "Контакт означает только обращение к объявлению: это НЕ покупка и НЕ подтверждение намерения купить. "
        "Не называй контакт просмотром, кликом, заказом или покупкой. Не характеризуй активность/частоту "
        "словами «регулярно», «ежедневно» или «часто»: используй только точные counts и даты ниже. "
        "Временное отношение этой истории к текущей выдаче неизвестно. Поля raw_category являются "
        "товарной таксономией и не являются маркой/моделью автомобиля. Не делай выводов о возрасте, "
        "поле, доходе, профессии или других чувствительных/личных свойствах. Не пиши, что пользователь "
        "что-либо купил или приобрёл. Отделяй наблюдения от гипотез; гипотезы допускаются только о "
        "товарном интересе и обязательно с модальностью «может». "
        "Верни только JSON с ключами summary (строка), observed_signals (массив строк), "
        "hypotheses (массив строк), limitations (массив строк), confidence (low|medium|high). "
        "summary обязательно назови «историей контактов». observed_signals должны содержать только точное "
        "число контактов, период экспорта и raw category counts. Значения пиши только по-русски.\n"
        f"user_id={user_id}; число контактов={len(history)}; период экспорта={period}; история:\n"
        f"{json.dumps(safe_history, ensure_ascii=False, allow_nan=False)}"
    )


def build_insight_requests(
    payload: Mapping[str, Any],
    *,
    top_k_per_ranking: int = 3,
) -> list[dict[str, Any]]:
    """Build leakage-free item/user prompts for featured cases only."""
    featured = set(payload.get("featured_serps", []))
    cases = [case for case in payload.get("cases", []) if case.get("serp_x") in featured]
    requests: dict[str, dict[str, Any]] = {}
    for case in cases:
        user_id = case.get("user_id")
        if user_id and case.get("history"):
            request_id = f"user:{user_id}"
            prompt = _user_prompt(str(user_id), case["history"])
            requests.setdefault(
                request_id,
                {
                    "request_id": request_id,
                    "kind": "user",
                    "entity_id": str(user_id),
                    "prompt_version": INSIGHT_PROMPT_VERSION,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "evidence": {
                        "history": [
                            {
                                "contact_date": row.get("contact_date"),
                                "raw_category_level_1": row.get("raw_taxonomy_level_1"),
                                "raw_category_level_2": row.get("raw_taxonomy_level_2"),
                            }
                            for row in case["history"]
                        ],
                        "history_summary": case.get("history_summary", {}),
                    },
                    "selected_for_serps": [],
                },
            )["selected_for_serps"].append(case["serp_x"])

        selected_items: dict[int, set[str]] = defaultdict(set)
        candidates = list(case.get("candidates", []))
        for method in ("source", "catboost", "qwen_l0"):
            for candidate in sorted(candidates, key=lambda row: row["ranks"][method])[:top_k_per_ranking]:
                selected_items[int(candidate["item_id"])].add(f"top_{top_k_per_ranking}_{method}")
        for candidate in candidates:
            if candidate["outcomes"]["actual_tc_event"]:
                selected_items[int(candidate["item_id"])].add("actual_tc_event_error_analysis")
        if candidates:
            best_relevance = max(
                candidates,
                key=lambda row: (row["outcomes"]["benchmark_relevance"] or 0.0, -int(row["item_id"])),
            )
            selected_items[int(best_relevance["item_id"])].add("max_benchmark_relevance")

        by_id = {int(candidate["item_id"]): candidate for candidate in candidates}
        for item_id, reasons in selected_items.items():
            candidate = by_id[item_id]
            request_id = f"item:{item_id}"
            prompt = _item_prompt(candidate)
            if any(field in prompt for field in POST_EXPOSURE_FIELDS):
                raise AssertionError("Post-exposure field name leaked into item insight prompt")
            row = requests.setdefault(
                request_id,
                {
                    "request_id": request_id,
                    "kind": "item",
                    "entity_id": str(item_id),
                    "prompt_version": INSIGHT_PROMPT_VERSION,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "evidence": {
                        "title": candidate.get("title"),
                        "description": candidate.get("description"),
                        "attributes": candidate.get("attributes", {}),
                    },
                    "selected_for_serps": [],
                    "selection_reasons": [],
                },
            )
            row["selected_for_serps"].append(case["serp_x"])
            row["selection_reasons"].extend(sorted(reasons))

    for row in requests.values():
        row["selected_for_serps"] = sorted(set(row["selected_for_serps"]))
        if "selection_reasons" in row:
            row["selection_reasons"] = sorted(set(row["selection_reasons"]))
    return sorted(requests.values(), key=lambda row: row["request_id"])


def build_item_request_rows(
    payload: Mapping[str, Any],
    *,
    featured_only: bool,
) -> list[dict[str, Any]]:
    featured = set(payload.get("featured_serps", []))
    aggregate: dict[int, dict[str, Any]] = {}

    def add(item_id: int, role: str, case: Mapping[str, Any], title: str | None, description: str | None, image: str | None) -> None:
        row = aggregate.setdefault(
            item_id,
            {
                "item_id": item_id,
                "roles": set(),
                "serp_ids": set(),
                "featured_reasons": set(),
                "known_title": title or "",
                "has_description": bool(description),
                "has_image": bool(image),
            },
        )
        row["roles"].add(role)
        row["serp_ids"].add(str(case["serp_x"]))
        row["featured_reasons"].update(reason["id"] for reason in case.get("featured_reasons", []))
        row["known_title"] = row["known_title"] or title or ""
        row["has_description"] = bool(row["has_description"] or description)
        row["has_image"] = bool(row["has_image"] or image)

    for case in payload.get("cases", []):
        if featured_only and case.get("serp_x") not in featured:
            continue
        for candidate in case.get("candidates", []):
            add(
                int(candidate["item_id"]),
                "candidate",
                case,
                candidate.get("title"),
                candidate.get("description"),
                candidate.get("image"),
            )
        for event in case.get("history", []):
            add(
                int(event["item_id"]),
                "history_contact",
                case,
                event.get("title"),
                event.get("description"),
                event.get("image"),
            )

    rows = []
    for item_id in sorted(aggregate):
        row = aggregate[item_id]
        rows.append(
            {
                "item_id": item_id,
                "priority": "featured" if row["featured_reasons"] else "full_test",
                "roles": "|".join(sorted(row["roles"])),
                "serp_ids": "|".join(sorted(row["serp_ids"])),
                "featured_reasons": "|".join(sorted(row["featured_reasons"])),
                "known_title": row["known_title"],
                "need_title": not bool(row["known_title"]),
                "need_description": not bool(row["has_description"]),
                "need_image": not bool(row["has_image"]),
            }
        )
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["item_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_browser_html(template_path: Path, output_path: Path, payload: Mapping[str, Any]) -> None:
    """Render data as base64 so listing text cannot break out of a script tag."""
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(raw).decode("ascii")
    template = template_path.read_text(encoding="utf-8")
    marker = "__AVITO_BROWSER_PAYLOAD_BASE64__"
    if template.count(marker) != 1:
        raise ValueError(f"Template must contain exactly one {marker} marker")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.replace(marker, encoded), encoding="utf-8")


def copy_overlay_images(
    overlay: pd.DataFrame,
    output_dir: Path,
    requested_item_ids: set[int],
    *,
    allowed_source_root: Path,
) -> pd.DataFrame:
    """Copy verified requested images from the overlay sandbox into browser assets."""
    if overlay.empty:
        return overlay
    result = overlay.copy()
    assets = _secure_asset_directory(output_dir)
    for index, row in result.iterrows():
        item_id = int(row["item_id"])
        raw_path = _clean_text(row.get("image_path"), 2000)
        if item_id not in requested_item_ids or not raw_path:
            continue
        if raw_path.lower().startswith(("http://", "https://", "data:image/")):
            continue
        source, suffix = _verified_image_path(
            Path(raw_path), allowed_root=allowed_source_root, label="overlay image"
        )
        destination = assets / f"{item_id}{suffix}"
        _atomic_copy_regular_file(source, destination, output_dir=output_dir)
        result.at[index, "image_path"] = str(
            destination.relative_to(Path(output_dir).resolve(strict=True))
        )
    return result


def _secure_asset_directory(output_dir: Path) -> Path:
    """Create ``assets/items`` without traversing attacker-controlled symlinks."""
    output_dir = Path(output_dir)
    if output_dir.is_symlink():
        raise ValueError(f"browser output directory must not be a symlink: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve(strict=True)
    current = root
    for component in ("assets", "items"):
        current = current / component
        if current.is_symlink():
            raise ValueError(f"browser asset directory must not be a symlink: {current}")
        if current.exists() and not current.is_dir():
            raise ValueError(f"browser asset path is not a directory: {current}")
        current.mkdir(exist_ok=True)
        if current.is_symlink() or not current.is_dir():
            raise ValueError(f"browser asset directory is unsafe: {current}")
        try:
            current.resolve(strict=True).relative_to(root)
        except ValueError as exc:
            raise ValueError(f"browser asset directory escaped output root: {current}") from exc
    return current


def _atomic_copy_regular_file(source: Path, destination: Path, *, output_dir: Path) -> None:
    """Atomically replace a regular asset while refusing symlink destinations."""
    root = Path(output_dir).resolve(strict=True)
    parent = destination.parent
    if parent.is_symlink():
        raise ValueError(f"browser asset parent must not be a symlink: {parent}")
    try:
        parent.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ValueError(f"browser asset destination escaped output root: {destination}") from exc
    if destination.is_symlink():
        raise ValueError(f"browser asset destination must not be a symlink: {destination}")
    if destination.exists() and not destination.is_file():
        raise ValueError(f"browser asset destination is not a regular file: {destination}")

    descriptor, temporary_name = tempfile.mkstemp(prefix=".asset-", dir=parent)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(source_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if destination.is_symlink():
            raise ValueError(f"browser asset destination became a symlink: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        # ``fdopen`` owns the descriptor after it succeeds.  If opening the
        # source fails first, close the still-live descriptor explicitly.
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _verified_image_path(
    path: Path,
    *,
    allowed_root: Path,
    label: str,
) -> tuple[Path, str]:
    """Resolve and decode an image without allowing reads outside ``allowed_root``."""
    root = Path(allowed_root).resolve()
    try:
        resolved = Path(path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} is outside the allowed root {root}: {resolved}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    size = resolved.stat().st_size
    if size <= 0 or size > MAX_OVERLAY_IMAGE_BYTES:
        raise ValueError(
            f"{label} size must be between 1 and {MAX_OVERLAY_IMAGE_BYTES} bytes: "
            f"{resolved} ({size} bytes)"
        )
    try:
        from PIL import Image, ImageFile, ImageSequence, UnidentifiedImageError

        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(resolved) as image:
            image_format = str(image.format or "").upper()
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_OVERLAY_IMAGE_PIXELS:
                raise ValueError(
                    f"{label} dimensions are invalid or too large: {width}x{height}"
                )
            image.verify()
        # ``verify`` checks container structure but deliberately does not decode
        # pixel data.  Reopen and load every frame so truncated JPEG/GIF files
        # cannot be copied into the portable report.
        with Image.open(resolved) as image:
            decoded_format = str(image.format or "").upper()
            frame_count = 0
            total_pixels = 0
            for frame in ImageSequence.Iterator(image):
                width, height = frame.size
                if width <= 0 or height <= 0 or width * height > MAX_OVERLAY_IMAGE_PIXELS:
                    raise ValueError(
                        f"{label} frame dimensions are invalid or too large: {width}x{height}"
                    )
                frame_count += 1
                total_pixels += width * height
                if frame_count > MAX_OVERLAY_IMAGE_FRAMES:
                    raise ValueError(
                        f"{label} has too many animation frames: {frame_count} > "
                        f"{MAX_OVERLAY_IMAGE_FRAMES}"
                    )
                if total_pixels > MAX_OVERLAY_IMAGE_TOTAL_PIXELS:
                    raise ValueError(
                        f"{label} decoded animation is too large: {total_pixels} > "
                        f"{MAX_OVERLAY_IMAGE_TOTAL_PIXELS} pixels"
                    )
                frame.load()
            if frame_count == 0 or decoded_format != image_format:
                raise ValueError(f"{label} could not be decoded consistently: {resolved}")
        # Pillow can intentionally tolerate a missing GIF trailer (and some
        # JPEG decoders tolerate a missing EOI marker).  Portable report assets
        # use the complete wire format, so reject these truncated containers.
        with resolved.open("rb") as handle:
            handle.seek(max(0, size - 2))
            trailer = handle.read()
        if image_format == "JPEG" and not trailer.endswith(b"\xff\xd9"):
            raise ValueError(f"{label} is not a fully decodable image (truncated JPEG): {resolved}")
        if image_format == "GIF" and not trailer.endswith(b"\x3b"):
            raise ValueError(f"{label} is not a fully decodable image (truncated GIF): {resolved}")
    except (UnidentifiedImageError, OSError, EOFError) as exc:
        raise ValueError(f"{label} is not a fully decodable image: {resolved}") from exc
    suffix = ALLOWED_IMAGE_FORMATS.get(image_format)
    if suffix is None:
        raise ValueError(
            f"{label} format {image_format or '<unknown>'!r} is not allowed; "
            f"expected one of {sorted(ALLOWED_IMAGE_FORMATS)}"
        )
    return resolved, suffix


def create_share_bundle(
    bundle_path: Path,
    browser_html: Path,
    metrics_path: Path,
    featured_manifest_path: Path,
    overlay_template_path: Path,
    all_manifest_path: Path | None = None,
    referenced_assets: Sequence[Path] = (),
) -> None:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    browser_root = browser_html.parent.resolve()
    verified_assets: list[tuple[Path, Path]] = []
    for candidate in sorted({Path(path).resolve() for path in referenced_assets}):
        asset, suffix = _verified_image_path(
            candidate,
            allowed_root=browser_root / "assets",
            label="referenced browser asset",
        )
        relative = asset.relative_to(browser_root)
        if not relative.parts or relative.parts[0] != "assets":
            raise ValueError(f"Browser asset must live below assets/: {asset}")
        if relative.suffix.lower() != suffix:
            raise ValueError(
                f"Browser asset extension does not match decoded {suffix} image: {asset}"
            )
        verified_assets.append((asset, relative))
    readme = (
        "Avito case browser — share bundle\n\n"
        "1. Open index.html in a modern browser; it is self-contained.\n"
        "2. item_request_featured.csv lists item IDs that need images/metadata first.\n"
        "3. item_request_all_test.csv contains the complete test-split request.\n"
        "4. Fill item_overlay_template.csv and rebuild with --item-overlay; "
        "local image_path files are copied by default.\n"
        "5. Ranking outcomes inside the HTML are post-exposure evaluation data, never model inputs.\n"
    )
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(browser_html, "index.html")
        archive.write(metrics_path, "metrics.json")
        archive.write(featured_manifest_path, "item_request_featured.csv")
        if all_manifest_path is not None:
            archive.write(all_manifest_path, "item_request_all_test.csv")
        archive.write(overlay_template_path, "item_overlay_template.csv")
        for asset, relative in verified_assets:
            archive.write(asset, relative)
        archive.writestr("README.txt", readme)
