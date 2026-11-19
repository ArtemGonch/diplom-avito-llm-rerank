"""Audit the BX history delivery and run fixed pre-search ranking diagnostics."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from statistics import mean
from zipfile import ZipFile
from zoneinfo import ZoneInfo

from avito_bx_baseline import ABS_TOLERANCE, PRIMARY_LABEL, audit_label, audit_preview


ARCHIVE_SHA256 = "84e5722513aa56949de71de654dc7d9fd77b85d8f4ef10ec178172bfe6a58226"
ATTRIBUTES = (
    "price", "brand", "model", "year", "generation", "transmission",
    "equipment", "power_hp", "engine_volume_l", "mileage_km",
)


def check_history_time(search_at, event_at):
    if not search_at - timedelta(days=7) <= event_at < search_at:
        raise ValueError("History event is outside the strict seven-day window")


def brand_name(value):
    return value.strip().casefold() if value and value.strip() else None


def history_preferences(events):
    """Return seen items and a uniquely supported brand, using BTC only."""
    by_item = defaultdict(set)
    for event in events:
        if event["interaction_type"] == "btc":
            brands = by_item[event["item_id"]]
            brand = brand_name(event["attr_brand"])
            if brand:
                brands.add(brand)
    votes = Counter(next(iter(brands)) for brands in by_item.values() if len(brands) == 1)
    top = votes.most_common(2)
    leader = None
    if top and top[0][1] >= 2 and (len(top) == 1 or top[0][1] > top[1][1]):
        leader = top[0][0]
    return set(by_item), leader


def paired_order(scores, priority):
    """Improve priority only inside disjoint adjacent pairs of the BX ranking."""
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    swaps = 0
    for i in range(0, len(order) - 1, 2):
        left, right = order[i:i + 2]
        if priority[left] == 0 and priority[right] == 1:
            order[i:i + 2] = [right, left]
            swaps += 1
    return order, swaps


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read_bundle(path):
    from pyarrow import parquet

    source = path.read_bytes()
    if _sha(source) != ARCHIVE_SHA256:
        raise ValueError("Archive does not match the reviewed 19 November delivery")
    with ZipFile(io.BytesIO(source)) as archive:
        report = json.loads(archive.read("report.json"))
        names = ["serps.parquet", "candidates.parquet", "history_7d.parquet",
                 "serps_7d.jsonl.gz"]
        members = {name: archive.read(name) for name in names}
        members["serps_7d.jsonl"] = gzip.decompress(members["serps_7d.jsonl.gz"])
        for name, data in members.items():
            ref = report["files"][name]
            if ref["bytes"] != len(data) or ref["sha256"] != _sha(data):
                raise ValueError(f"Bundle member fingerprint mismatch: {name}")
        candidate_table = parquet.read_table(io.BytesIO(members["candidates.parquet"]))
        serps = parquet.read_table(io.BytesIO(members["serps.parquet"])).to_pylist()
        history = parquet.read_table(io.BytesIO(members["history_7d.parquet"])).to_pylist()
        json_groups = [json.loads(line) for line in members["serps_7d.jsonl"].splitlines()]
        reference = json.loads(archive.read("baseline_check.json"))
    return candidate_table, serps, history, json_groups, reference


def _group(rows):
    result = defaultdict(list)
    for row in rows:
        result[row["serp_x"]].append(row)
    return result


def _nullable(value):
    return None if isinstance(value, float) and math.isnan(value) else value


def _attributes(row):
    return {key: _nullable(row[f"attr_{key}"]) for key in ATTRIBUTES}


def _check_interval(row, at):
    if not (
        row["attributes_found"] is True
        and row["attributes_from"] <= at < datetime.fromisoformat(row["attributes_to"])
    ):
        raise ValueError("Invalid effective attribute interval")
    if row["price_at"] is not None and row["price_at"] > at:
        raise ValueError("Price effective date is after the cutoff")
    if row["price_loaded_at"] is not None and row["price_loaded_at"] > at:
        raise ValueError("Known price load date is after the cutoff")


def _check_bridge(rows, external, internal, *, optional=False):
    forward, backward = {}, {}
    for row in rows:
        outer, inner = _nullable(row[external]), row[internal]
        if optional and outer is None and inner is None:
            continue
        # The historical export typed the external user ID as double.
        if (
            external == "user_id"
            and isinstance(outer, float)
            and math.isfinite(outer)
            and outer.is_integer()
            and abs(outer) < 2**53
        ):
            outer = int(outer)
        if type(outer) is not int or type(inner) is not int:
            raise ValueError(f"Invalid exact identifier in {external} bridge")
        if (
            forward.setdefault(outer, inner) != inner
            or backward.setdefault(inner, outer) != outer
        ):
            raise ValueError(f"Conflicting mapping in {external} bridge")


def _check_alignment(serps, candidates, history, json_groups):
    c, h = _group(candidates), _group(history)
    meta, as_json = ({r["serp_x"]: r for r in rows} for rows in (serps, json_groups))
    if (
        len(meta) != len(serps) or len(as_json) != len(json_groups)
        or set(meta) != set(c) or set(meta) != set(as_json) or not set(h) <= set(meta)
    ):
        raise ValueError("Unexpected duplicate, missing or orphan search group")
    _check_bridge([*serps, *history], "user_id", "dwh_user_id", optional=True)
    _check_bridge([*candidates, *history], "item_id", "dwh_item_id")
    for key, request in meta.items():
        search = request["search_at"]
        moscow = datetime.fromtimestamp(request["features_timestamp"], ZoneInfo("Europe/Moscow"))
        if moscow.replace(tzinfo=None) != search:
            raise ValueError("Search timestamp disagrees with source cutoff")
        actor = _nullable(request["user_id"])
        events = h.get(key, [])
        pool = c[key]
        if (
            request["history_status"] != ("missing_user_id" if actor is None else "available")
            or request["history_event_count"] != len(events)
            or (actor is None and events)
            or len({r["event_key"] for r in events}) != len(events)
            or len({r["item_id"] for r in pool}) != len(pool)
        ):
            raise ValueError("Inconsistent history status, count or duplicate key")
        for row in pool:
            if (
                _nullable(row["user_id"]) != actor
                or row["features_timestamp"] != request["features_timestamp"]
                or row["query_text"] != request["query_text"]
            ):
                raise ValueError("Candidate disagrees with search metadata")
            _check_interval(row, search)
        for event in events:
            event_at = event["event_at"]
            if (
                _nullable(event["user_id"]) != actor
                or event["search_at"] != search
                or event["dwh_user_id"] != request["dwh_user_id"]
                or event["interaction_type"] not in {"btc", "contact", "dtb"}
            ):
                raise ValueError("History disagrees with search metadata")
            check_history_time(search, event_at)
            _check_interval(event, event_at)
        expected = {
            "serp_x": key, "user_id": actor,
            "search_at": search.isoformat(sep=" "),
            "history_status": request["history_status"], "query_text": request["query_text"],
            "candidates": [
                {"item_id": r["item_id"], "source_row_index": r["source_row_index"],
                 "attributes": _attributes(r)}
                for r in pool
            ],
            "history": [
                {"item_id": r["item_id"], "at": r["event_at"].isoformat(sep=" "),
                 "type": r["interaction_type"], "attributes": _attributes(r)}
                for r in events
            ],
        }
        if as_json[key] != expected:
            raise ValueError("Compact JSONL differs from the joined Parquet tables")
    return c, h


def _diagnostics(candidates, history):
    methods = ("seen_btc_item", "dominant_btc_brand")
    labels = (PRIMARY_LABEL, "target")
    changes = {method: Counter() for method in methods}
    paired = {(method, label): [] for method in methods for label in labels}
    coverage = Counter()
    for key, pool in candidates.items():
        events = history.get(key, [])
        seen, brand = history_preferences(events)
        coverage["groups_with_btc"] += bool(seen)
        coverage["groups_with_dominant_brand"] += brand is not None
        item_priority = [int(r["item_id"] in seen) for r in pool]
        candidate_brands = [brand_name(r["attr_brand"]) for r in pool]
        brand_priority = [
            int(value == brand) if value is not None and brand is not None else None
            for value in candidate_brands
        ]
        coverage["groups_with_seen_candidate"] += any(item_priority)
        coverage["groups_with_matching_brand_candidate"] += any(brand_priority)
        scores = [r["baseline_score"] for r in pool]
        group = [range(len(pool))]
        targets = {label: [r[label] for r in pool] for label in labels}
        base = {label: audit_label(group, scores, y)["ndcg"] for label, y in targets.items()}
        for method, priority in zip(methods, (item_priority, brand_priority)):
            order, swaps = paired_order(scores, priority)
            changes[method]["changed_groups"] += bool(swaps)
            changes[method]["swapped_pairs"] += swaps
            reordered = [0] * len(pool)
            for rank, index in enumerate(order):
                reordered[index] = -rank
            for label, original in base.items():
                if original is not None:
                    result = audit_label(group, reordered, targets[label])["ndcg"]
                    paired[method, label].append((original, result, bool(swaps)))
    metrics = {}
    for method in methods:
        metrics[method] = dict(changes[method], metrics={})
        for label in labels:
            values = paired[method, label]
            delta = [b - a for a, b, _ in values]
            metrics[method]["metrics"][label] = {
                "eligible_groups": len(values),
                "changed_groups_among_eligible": sum(change for _, _, change in values),
                "baseline": mean(a for a, _, _ in values),
                "reranked": mean(b for _, b, _ in values),
                "delta": mean(delta),
                "wins": sum(d > ABS_TOLERANCE for d in delta),
                "ties": sum(abs(d) <= ABS_TOLERANCE for d in delta),
                "losses": sum(d < -ABS_TOLERANCE for d in delta),
            }
    return dict(coverage), metrics


def audit_history_bundle(original_path, archive_path):
    from pyarrow import parquet

    baseline = audit_preview(original_path)
    enriched, serps, history, json_groups, reference = _read_bundle(archive_path)
    original = parquet.read_table(original_path)
    if not original.equals(enriched.select(original.column_names)):
        raise ValueError("The enriched table changes candidate rows, values or physical order")
    candidate_columns = [
        "serp_x", "source_row_index", "user_id", "item_id", "dwh_item_id",
        "query_text", "features_timestamp", "attributes_found", "attributes_from",
        "attributes_to", "price_at", "price_loaded_at", "baseline_score",
    ] + [f"attr_{key}" for key in ATTRIBUTES] + list(baseline["targets"])
    candidates = enriched.select(candidate_columns).to_pylist()
    grouped_candidates, grouped_history = _check_alignment(serps, candidates, history, json_groups)
    coverage, methods = _diagnostics(grouped_candidates, grouped_history)
    positions = defaultdict(list)
    for index, row in enumerate(candidates):
        positions[row["serp_x"]].append(index)
    groups = list(positions.values())
    scores = [r["baseline_score"] for r in candidates]
    max_error = 0.0
    if set(reference) != set(baseline["targets"]):
        raise ValueError("The enclosed baseline reference uses different targets")
    for label, original_stats in baseline["targets"].items():
        calculated = audit_label(groups, scores, [r[label] for r in candidates])
        expected = reference[label]
        if (
            calculated["eligible_groups"] != original_stats["eligible_groups"]
            or calculated["eligible_groups"] != expected["eligible_groups"]
        ):
            raise ValueError("Baseline eligible groups changed after enrichment")
        max_error = max(
            max_error, abs(calculated["ndcg"] - original_stats["ndcg"]),
            abs(calculated["ndcg"] - expected["ndcg"]),
        )
    if max_error > ABS_TOLERANCE:
        raise ValueError("Baseline metric changed after enrichment")
    missing = sum(row["history_status"] == "missing_user_id" for row in serps)
    return {
        "protocol": "BX UsedCars seven-day history audit and fixed adjacent-pair diagnostics",
        "archive_sha256": ARCHIVE_SHA256,
        "original_sha256": baseline["input_sha256"],
        "member_checksums_verified": 5,
        "original_columns_verified": len(original.column_names),
        "groups": len(serps),
        "candidates": len(candidates),
        "history_events": len(history),
        "history_event_types": dict(sorted(Counter(r["interaction_type"] for r in history).items())),
        "groups_with_history": len(grouped_history),
        "groups_missing_user_id": missing,
        "known_user_groups_without_history": len(serps) - len(grouped_history) - missing,
        "history_window_violations": 0,
        "candidate_attributes_known": {
            k: sum(_nullable(r[f"attr_{k}"]) is not None for r in candidates)
            for k in ATTRIBUTES
        },
        "candidate_price_load_times_known": sum(r["price_loaded_at"] is not None for r in candidates),
        "history_price_load_times_known": sum(r["price_loaded_at"] is not None for r in history),
        "baseline_labels_reproduced": len(reference),
        "baseline_reference_max_absolute_error": max_error,
        "coverage": coverage,
        "methods": methods,
    }
