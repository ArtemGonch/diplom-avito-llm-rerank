"""Reproduce the BX UsedCars baseline on original and normalized candidates."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter
from contextlib import closing
from itertools import chain, groupby
from numbers import Real
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory


SHA256 = "eb2add0ce2c114d8088918b85a8730b9b05f21defe81461bd1c96720da055ea3"
PRIMARY_LABEL = "target_contact_s_d7"
TARGETS = (
    "target", "btc_s", "fav_s", "btc_wo_fav_s", "btc_s_d7", "fav_s_d7",
    "btc_wo_fav_s_d7", "contact_s", "contact_s_d7", PRIMARY_LABEL,
)
DISCOUNT = 0.8
EPSILON = 1e-5
ABS_TOLERANCE = 1e-12
PILOT_SPLIT = "test"
CANDIDATE_COLUMNS = ("serp_x", "source_row_index", "baseline_score", *TARGETS)
GROUP_COUNT_FIELDS = (
    "eligible_groups", "homogeneous_groups", "no_observed_labels_groups",
    "groups_with_positive_gain", "eligible_groups_with_score_ties",
)
PROTOCOL = "BX exponential NDCG on full groups"


def _observed_gain(gain):
    if gain is None:
        return None
    if not isinstance(gain, Real):
        raise ValueError("Expected a numeric gain")
    if math.isnan(gain):
        return None
    if not math.isfinite(gain) or gain < 0:
        raise ValueError("Expected finite, nonnegative gains")
    return gain


def _ranked_group(observed):
    """Return the score and group counts for observations in tie-break order."""
    counts = dict.fromkeys(GROUP_COUNT_FIELDS, 0)
    if not observed:
        counts["no_observed_labels_groups"] = 1
        return None, counts

    scores, gains = zip(*observed)
    ideal = sorted(gains, reverse=True)
    counts["groups_with_positive_gain"] = int(ideal[0] > 0)
    if ideal[0] == ideal[-1]:
        counts["homogeneous_groups"] = 1
        return None, counts

    counts["eligible_groups"] = 1
    counts["eligible_groups_with_score_ties"] = int(len(set(scores)) < len(scores))
    actual = (gain for _, gain in sorted(observed, key=lambda row: row[0], reverse=True))
    weights = [DISCOUNT**rank for rank in range(1, len(ideal) + 1)]
    dcg = math.fsum(gain * weight for gain, weight in zip(actual, weights))
    idcg = math.fsum(gain * weight for gain, weight in zip(ideal, weights))
    return dcg / (idcg + EPSILON), counts


def audit_label(groups, scores, labels):
    """Evaluate one label on the full pool, masking missing gain before ranking."""
    observed = {}
    for index, gain in enumerate(labels):
        value = _observed_gain(gain)
        if value is not None:
            observed[index] = value

    averages = []
    counts = Counter(dict.fromkeys(GROUP_COUNT_FIELDS, 0))
    for group in groups:
        ndcg, counted = _ranked_group([
            (scores[index], observed[index]) for index in group if index in observed
        ])
        counts.update(counted)
        if ndcg is not None:
            averages.append(ndcg)

    distinct = set(observed.values())
    return {
        "ndcg": math.fsum(averages) / len(averages) if averages else None,
        **counts,
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
        "protocol": PROTOCOL,
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


def _candidate_rows(path, *, assigned_split=None):
    """Project the ranking inputs, retaining exact values and masking missing gains."""
    from pyarrow import parquet, types

    def is_string(dtype):
        return types.is_string(dtype) or types.is_large_string(dtype)

    def is_numeric(dtype):
        return types.is_integer(dtype) or types.is_floating(dtype)

    with parquet.ParquetFile(path) as source:
        schema = source.schema_arrow
        names = schema.names
        if len(set(names)) != len(names) or not set(CANDIDATE_COLUMNS) <= set(names):
            raise ValueError("Missing ranking columns or duplicate column names")
        embedded_split = "split" in names
        if not embedded_split and assigned_split is None:
            raise ValueError("Expected a split column or an explicit candidate split")
        if assigned_split is not None and (
            not isinstance(assigned_split, str) or not assigned_split.strip()
        ):
            raise ValueError("Invalid split")

        if (
            not is_string(schema.field("serp_x").type)
            or (embedded_split and not is_string(schema.field("split").type))
            or not types.is_integer(schema.field("source_row_index").type)
            or not is_numeric(schema.field("baseline_score").type)
        ):
            raise ValueError("Invalid ranking key or baseline_score column type")
        for label in TARGETS:
            dtype = schema.field(label).type
            if not (is_numeric(dtype) or types.is_null(dtype)):
                raise ValueError(f"Invalid numeric gain column type for {label}")

        columns = ("split", *CANDIDATE_COLUMNS) if embedded_split else CANDIDATE_COLUMNS
        for batch in source.iter_batches(columns=columns, batch_size=16_384):
            values = [column.to_pylist() for column in batch.columns]
            for row in zip(*values, strict=True):
                split, serp, index, score, *gains = row if embedded_split else (assigned_split, *row)
                if (
                    split is None or not split.strip()
                    or (assigned_split is not None and split != assigned_split)
                ):
                    raise ValueError("Missing or conflicting candidate split")
                if serp is None or not serp.strip():
                    raise ValueError("Missing serp_x")
                if index is None or index < 0:
                    raise ValueError("Invalid source_row_index")
                if score is None or not math.isfinite(score):
                    raise ValueError("Invalid baseline_score")
                # Big-endian bytes preserve numeric order for signed and unsigned Arrow IDs.
                yield split, serp, index.to_bytes(8, "big"), (score, *map(_observed_gain, gains))


class _CandidateAudit:
    def __init__(self):
        self.rows = self.groups = self.tied_groups = 0
        self.counts = {label: Counter() for label in TARGETS}
        self.gains = {label: set() for label in TARGETS}
        self.sums = dict.fromkeys(TARGETS, 0.0)
        self.corrections = dict.fromkeys(TARGETS, 0.0)

    def add(self, size, tied, group_stats):
        self.rows += size
        self.groups += 1
        self.tied_groups += tied
        for label, (ndcg, counts, gains) in zip(TARGETS, group_stats, strict=True):
            self.counts[label].update(counts)
            self.gains[label].update(gains)
            if ndcg is not None:
                # Compensated summation retains precision without storing one value per group.
                increment = ndcg - self.corrections[label]
                total = self.sums[label] + increment
                self.corrections[label] = (total - self.sums[label]) - increment
                self.sums[label] = total

    def result(self):
        audits = {}
        for label, counts in self.counts.items():
            eligible = counts["eligible_groups"]
            distinct = self.gains[label]
            audits[label] = {
                "ndcg": self.sums[label] / eligible if eligible else None,
                **counts,
                "distinct_observed_gains": len(distinct),
                "observed_gains": sorted(distinct) if len(distinct) <= 20 else None,
            }
        return {
            "rows": self.rows,
            "groups": self.groups,
            "groups_with_baseline_score_ties": self.tied_groups,
            "targets": audits,
        }


def _candidate_metrics(db):
    overall = _CandidateAudit()
    splits = {}
    rows = db.execute("""
        SELECT split, serp, payload FROM candidates
        ORDER BY split, serp, source_row_index
    """)
    for split, split_rows in groupby(rows, key=lambda row: row[0]):
        split_audit = _CandidateAudit()
        for _, group_rows in groupby(split_rows, key=lambda row: row[1]):
            group = [json.loads(payload) for _, _, payload in group_rows]
            scores, *labels = zip(*group)
            size = len(scores)
            tied = len(set(scores)) < size
            group_stats = []
            for values in labels:
                observed = [
                    (score, gain) for score, gain in zip(scores, values) if gain is not None
                ]
                ndcg, counts = _ranked_group(observed)
                counts.update(
                    missing_rows=size - len(observed),
                    positive_rows=sum(gain > 0 for _, gain in observed),
                )
                group_stats.append((ndcg, counts, {gain for _, gain in observed}))
            split_audit.add(size, tied, group_stats)
            overall.add(size, tied, group_stats)
        splits[split] = split_audit.result()
    return {**overall.result(), "splits": splits}


def _check_candidate_preview(db, result, original):
    reference = audit_preview(original)
    if set(result["splits"]) != {PILOT_SPLIT}:
        raise ValueError("The normalized pilot must contain exactly the original test split")
    if result["rows"] != reference["rows"] or result["groups"] != reference["groups"]:
        raise ValueError("Normalized pilot candidate or group count differs from the original")

    for split, serp, index, values in _candidate_rows(original, assigned_split=PILOT_SPLIT):
        row = db.execute(
            "SELECT serp, payload FROM candidates WHERE split = ? AND source_row_index = ?",
            (split, index),
        ).fetchone()
        if row is None or row[0] != serp:
            raise ValueError("Normalized pilot candidate or group key differs from the original")
        normalized = json.loads(row[1])
        if normalized[0] != values[0]:
            raise ValueError("Normalized pilot baseline_score differs from the original")
        for label, observed, expected in zip(TARGETS, normalized[1:], values[1:], strict=True):
            if observed != expected:
                raise ValueError(f"Normalized pilot gain for {label} differs from the original")

    means = {}
    counts = {}
    for label, audit in reference["targets"].items():
        calculated = result["targets"][label]
        if any(calculated[key] != value for key, value in audit.items() if key != "ndcg"):
            raise ValueError(f"Normalized pilot counts for {label} differ from the original")
        metric = f"NDCG_b{DISCOUNT}({label})"
        means[metric] = audit["ndcg"]
        counts[metric] = audit["eligible_groups"]
    return {
        "original_sha256": reference["input_sha256"],
        "rows_matched": reference["rows"],
        "groups_matched": reference["groups"],
        "targets_matched": len(counts),
        "absolute_tolerance": ABS_TOLERANCE,
        "max_absolute_ndcg_difference": check_reference(result["targets"], means, counts),
    }


def audit_candidates(paths=(), *, split_paths=(), original=None):
    """Audit normalized Parquet shards, optionally checking an original test preview."""
    with TemporaryDirectory(prefix="bx-candidates-") as directory:
        with closing(sqlite3.connect(Path(directory) / "ranking.sqlite3")) as db, db:
            db.execute("PRAGMA temp_store = FILE")
            db.execute("""
                CREATE TABLE candidates (
                    split TEXT, serp TEXT NOT NULL, source_row_index BLOB, payload TEXT NOT NULL,
                    PRIMARY KEY (split, source_row_index)
                ) WITHOUT ROWID
            """)
            for split, path in chain(((None, path) for path in paths), split_paths):
                rows = _candidate_rows(path, assigned_split=split)
                try:
                    db.executemany("INSERT INTO candidates VALUES (?, ?, ?, ?)", (
                        (row_split, serp, index, json.dumps(values, allow_nan=False))
                        for row_split, serp, index, values in rows
                    ))
                except sqlite3.IntegrityError as error:
                    raise ValueError("Duplicate (split, source_row_index) candidate key") from error
            db.execute("CREATE INDEX groups_in_order ON candidates (split, serp, source_row_index)")
            result = _candidate_metrics(db)
            if not result["rows"]:
                raise ValueError("No candidate rows")
            if original is not None:
                result["reconciliation"] = _check_candidate_preview(db, result, original)
    return {
        "protocol": PROTOCOL,
        "primary_label": PRIMARY_LABEL,
        "discount_base": DISCOUNT,
        "first_rank": 1,
        "idcg_epsilon": EPSILON,
        "score_tie_break": "source_row_index ascending",
        **result,
    }
