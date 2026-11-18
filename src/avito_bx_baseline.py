"""Reproduce the saved BX UsedCars preview baseline on the original row order."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from statistics import median


SHA256 = "eb2add0ce2c114d8088918b85a8730b9b05f21defe81461bd1c96720da055ea3"
PRIMARY_LABEL = "target_contact_s_d7"
DISCOUNT = 0.8
EPSILON = 1e-5
ABS_TOLERANCE = 1e-12


def audit_label(groups, scores, labels):
    """Evaluate one label on the full pool, masking missing gain before ranking."""
    observed = {}
    for index, gain in enumerate(labels):
        if gain is None:
            continue
        if not isinstance(gain, Real):
            raise ValueError("Expected a numeric gain")
        if math.isnan(gain):
            continue
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("Expected finite, nonnegative gains")
        observed[index] = gain

    averages = []
    empty = homogeneous = positive_groups = ties = 0
    for group in groups:
        kept = [index for index in group if index in observed]
        if not kept:
            empty += 1
            continue
        gains = [observed[index] for index in kept]
        positive_groups += max(gains) > 0
        if min(gains) == max(gains):
            homogeneous += 1
            continue

        ties += len({scores[index] for index in kept}) < len(kept)
        ranking = sorted(kept, key=scores.__getitem__, reverse=True)
        actual = (observed[index] for index in ranking)
        ideal = sorted(gains, reverse=True)
        weights = [DISCOUNT**rank for rank in range(1, len(kept) + 1)]
        dcg = math.fsum(gain * weight for gain, weight in zip(actual, weights))
        idcg = math.fsum(gain * weight for gain, weight in zip(ideal, weights))
        averages.append(dcg / (idcg + EPSILON))

    distinct = set(observed.values())
    return {
        "ndcg": math.fsum(averages) / len(averages) if averages else None,
        "eligible_groups": len(averages),
        "homogeneous_groups": homogeneous,
        "no_observed_labels_groups": empty,
        "groups_with_positive_gain": positive_groups,
        "eligible_groups_with_score_ties": ties,
        "missing_rows": len(labels) - len(observed),
        "positive_rows": sum(gain > 0 for gain in observed.values()),
        "distinct_observed_gains": len(distinct),
        "observed_gains": sorted(distinct) if len(distinct) <= 20 else None,
    }


def check_reference(audits, means, counts):
    keys = {label: f"NDCG_b{DISCOUNT}({label})" for label in audits}
    expected_keys = set(keys.values())
    if expected_keys != set(means) or expected_keys != set(counts):
        raise ValueError("Reference targets differ from evaluated targets")
    max_error = 0.0
    for label, audit in audits.items():
        metric = keys[label]
        count = counts[metric]
        if type(count) is not int or count != audit["eligible_groups"]:
            raise ValueError(f"Eligible group count mismatch for {label}")
        actual, expected = audit["ndcg"], means[metric]
        if actual is None and expected is None:
            continue
        if (
            actual is None
            or isinstance(expected, bool)
            or not isinstance(expected, Real)
            or not math.isclose(actual, expected, rel_tol=0, abs_tol=ABS_TOLERANCE)
        ):
            raise ValueError(f"Reference NDCG mismatch for {label}")
        max_error = max(max_error, abs(actual - expected))
    return max_error


def audit_target_components(target, ux, tc):
    triples = list(zip(target, ux, tc, strict=True))
    if any(
        not isinstance(gain, Real) or not math.isfinite(gain) or gain < 0
        for row in triples
        for gain in row
    ):
        raise ValueError("Invalid final target or component gain")
    return {
        "rows_checked": len(triples),
        "equals_max": sum(y == max(u, t) for y, u, t in triples),
        "equals_sum": sum(y == u + t for y, u, t in triples),
        "ux_target_values": sorted(set(ux)),
        "tc_target_values": sorted(set(tc)),
    }


def audit_preview(path):
    """Audit the published preview with the exact README fingerprint."""
    from pyarrow import parquet

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256 or path.stat().st_size != 13_245_668:
        raise ValueError("Parquet does not match the preview fingerprint")

    source = parquet.ParquetFile(path)
    schema = source.schema_arrow
    metadata = schema.metadata or {}
    if b"ndcg_sample_info" not in metadata:
        raise ValueError("Missing ndcg_sample_info")
    info = json.loads(metadata[b"ndcg_sample_info"])
    targets = info["targets"]
    features = info["features"]
    if (
        len(schema.names) != 265
        or len(set(schema.names)) != 265
        or len(targets) != 10
        or len(set(targets)) != 10
        or PRIMARY_LABEL not in targets
        or len(features) != 225
        or len(set(features)) != 225
        or not set(targets + features) <= set(schema.names)
        or set(targets) & set(features)
    ):
        raise ValueError("Unexpected preview schema or metadata")

    structural = [
        "source_row_index",
        "serp_x",
        "baseline_score",
        "query_infm_logical_category",
        "query_text",
        "ux_target",
        "tc_target",
    ]
    data = source.read(columns=structural + targets).to_pydict()
    row_ids = data["source_row_index"]
    if (
        len(row_ids) != 31_784
        or any(type(row_id) is not int or row_id < 0 for row_id in row_ids)
        or len(set(row_ids)) != len(row_ids)
    ):
        raise ValueError("Invalid source_row_index")
    scores = data["baseline_score"]
    if any(not isinstance(score, Real) or not math.isfinite(score) for score in scores):
        raise ValueError("Invalid baseline score")

    group_keys = data["serp_x"]
    if any(key is None or key == "" for key in group_keys):
        raise ValueError("Missing search group key")
    by_group = {}
    for index, key in enumerate(group_keys):
        by_group.setdefault(key, []).append(index)
    if len(by_group) != 1_000:
        raise ValueError("Unexpected search group count")
    if set(data["query_infm_logical_category"]) != {"Transport.UsedCars"}:
        raise ValueError("Unexpected category in UsedCars preview")
    groups = list(by_group.values())
    sizes = [len(group) for group in groups]
    audits = {label: audit_label(groups, scores, data[label]) for label in targets}
    max_error = check_reference(audits, info["baseline_means"], info["eligible_groups"])
    event_columns = [name for name in schema.names if "events" in name.lower()]

    return {
        "protocol": "BX exponential NDCG on full groups",
        "input_sha256": digest,
        "input_bytes": path.stat().st_size,
        "rows": len(row_ids),
        "groups": len(groups),
        "columns": len(schema.names),
        "model_features": len(features),
        "category": "Transport.UsedCars",
        "rows_with_missing_or_blank_query": sum(
            text is None or not text.strip() for text in data["query_text"]
        ),
        "event_columns": event_columns,
        "primary_label": PRIMARY_LABEL,
        "discount_base": DISCOUNT,
        "first_rank": 1,
        "idcg_epsilon": EPSILON,
        "minimum_group_size": min(sizes),
        "median_group_size": median(sizes),
        "maximum_group_size": max(sizes),
        "groups_with_baseline_score_ties": sum(
            len({scores[index] for index in group}) < len(group) for group in groups
        ),
        "baseline_score_distinct": len(set(scores)),
        "targets": audits,
        "target_components": audit_target_components(
            data["target"], data["ux_target"], data["tc_target"]
        ),
        "metadata_abs_tolerance": ABS_TOLERANCE,
        "metadata_max_absolute_error": max_error,
    }
