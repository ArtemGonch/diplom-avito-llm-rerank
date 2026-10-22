"""Allowlisted technical attributes from the private UsedCars source export.

The returned object remains an internal dataset, not a publication projection.
Only power, displacement, doors, and their provenance are copied from sources;
existing dataset fields are preserved and must still pass the UI privacy filter.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


def _read_rows(root, name):
    path = root / name
    if not path.is_file():
        return [], False
    if path.suffix == ".jsonl":
        rows = []
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid source JSON at {name}:{line_no}") from None
            if not isinstance(row, dict):
                raise ValueError(f"Expected source object at {name}:{line_no}")
            rows.append(row)
        return rows, True
    try:
        rows = json.loads(path.read_text())
    except (ValueError, TypeError):
        raise ValueError(f"Invalid source JSON: {name}") from None
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f"Expected source object list: {name}")
    return rows, True


def _params(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return None
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else None


def _timestamp(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _bounded(value, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    try:
        number = Decimal(str(value).replace("\u00a0", "").replace(" ", ""))
        if not number.is_finite() or not Decimal(str(low)) <= number <= Decimal(
            str(high)
        ):
            return None
        if integer and number != number.to_integral_value():
            return None
        return int(number) if integer else float(number)
    except (InvalidOperation, ValueError):
        return None


def _covers(row, last):
    start = _timestamp(row.get("from_ts"))
    end_value = row.get("to_ts")
    end = _timestamp(end_value)
    if last is None or start is None or (end_value not in (None, "") and end is None):
        return False
    try:
        return start <= last and (end is None or last < end)
    except TypeError:
        # Mixed timezone conventions cannot establish an as-of interval.
        return False


def enrich_technical_attributes(dataset, archive_root):
    """Return a deep copy plus aggregate audit; never overwrite existing attrs.

    A single valid interval at ``last`` takes priority. Missing or ambiguous
    intervals use only a unique persisted viewer snapshot, retaining its actual
    timestamp. Missing technical values stay missing rather than borrowing a
    different date or silently mixing the two sources.
    """
    root = Path(archive_root)
    history_name = "source/item_params_history.jsonl"
    snapshot_name = "source/viewer-ads.json"
    dictionary_name = "source/item_attribute_value_dictionary.jsonl"
    history, have_history = _read_rows(root, history_name)
    snapshots, have_snapshots = _read_rows(root, snapshot_name)
    dictionary, have_dictionary = _read_rows(root, dictionary_name)

    intervals = defaultdict(list)
    for row in history:
        parsed = _params(row.get("json_params"))
        if parsed is not None:
            intervals[str(row.get("item_id"))].append((row, parsed))
    by_item = defaultdict(list)
    for row in snapshots:
        by_item[str(row.get("item_id"))].append(row)

    labels = defaultdict(set)
    for row in dictionary:
        # IDs from different parameters must never share a decoding namespace.
        labels[(str(row.get("param_id")), str(row.get("id")))].add(
            str(row.get("defaultlabel", ""))
        )

    result = deepcopy(dataset)
    counts = Counter()
    missing = Counter()
    rejected = Counter()
    for user in result.get("users", []):
        for card in user.get("cards", []):
            counts["cards"] += 1
            last_raw = card.get("last", "")
            last = _timestamp(last_raw)
            counts["invalid_last_cards"] += last is None
            matches = [
                (row, params)
                for row, params in intervals.get(str(card.get("item")), [])
                if _covers(row, last)
            ]
            source = ""
            stamp = ""
            status = "not_available"
            interval = None
            params = {}
            if len(matches) == 1:
                row, params = matches[0]
                source, stamp, status = history_name, str(last_raw), "as_of_last"
                interval = {"from": str(row["from_ts"]), "to": row.get("to_ts")}
                counts["as_of_last_cards"] += 1
            else:
                counts["ambiguous_intervals"] += len(matches) > 1
                rows = by_item.get(str(card.get("item")), [])
                counts["ambiguous_snapshots"] += len(rows) > 1
                if len(rows) == 1:
                    row = rows[0]
                    parsed = _params(row.get("params"))
                    if parsed is not None:
                        params = parsed
                        source = snapshot_name
                        snapshot_raw = row.get("snapshot_ts")
                        snapshot_time = _timestamp(snapshot_raw)
                        # Retain only a timestamp, never arbitrary source text.
                        stamp = snapshot_raw if snapshot_time is not None else ""
                        status = "snapshot_time_unknown"
                        if snapshot_time is not None and last is not None:
                            try:
                                status = (
                                    "snapshot_after_last"
                                    if snapshot_time > last
                                    else "snapshot_before_or_at_last"
                                )
                            except TypeError:
                                pass
                        counts["fallback_snapshot_cards"] += 1
                if not source:
                    counts["unavailable_cards"] += 1

            fields = {
                "power_hp": _bounded(params.get("110011"), 10, 3000),
                "engine_liters": _bounded(params.get("110012"), 0, 20),
                "doors": None,
            }
            raw_doors = params.get("110004")
            if not isinstance(raw_doors, bool):
                door_labels = labels.get(("110004", str(raw_doors)), set())
                if len(door_labels) == 1:
                    fields["doors"] = _bounded(
                        next(iter(door_labels)), 2, 5, integer=True
                    )
            for field, code in (
                ("power_hp", "110011"),
                ("engine_liters", "110012"),
                ("doors", "110004"),
            ):
                missing[field] += fields[field] is None
                rejected[field] += (
                    params.get(code) is not None and fields[field] is None
                )
            card.update(fields)
            card["technicalSource"] = source
            card["technicalSnapshot"] = stamp
            card["technicalTemporalStatus"] = status
            card["technicalInterval"] = interval
            counts["snapshot_after_last_cards"] += status == "snapshot_after_last"

    audit = {
        key: counts[key]
        for key in (
            "cards",
            "as_of_last_cards",
            "fallback_snapshot_cards",
            "unavailable_cards",
            "ambiguous_intervals",
            "ambiguous_snapshots",
            "invalid_last_cards",
            "snapshot_after_last_cards",
        )
    }
    audit.update(
        source_rows={"params_history": len(history), "viewer_ads": len(snapshots)},
        missing_fields={
            field: missing[field] for field in ("power_hp", "engine_liters", "doors")
        },
        rejected_values={
            field: rejected[field] for field in ("power_hp", "engine_liters", "doors")
        },
        missing_source_files=[
            name
            for name, present in (
                (history_name, have_history),
                (snapshot_name, have_snapshots),
                (dictionary_name, have_dictionary),
            )
            if not present
        ],
    )
    return result, audit
