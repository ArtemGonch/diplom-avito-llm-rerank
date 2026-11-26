#!/usr/bin/env python3
"""Measure Text2Filter output coverage over every text and its frequency in the UsedCars export."""

import argparse
from collections import Counter
import csv
from hashlib import file_digest
import io
import json
from pathlib import Path
from zipfile import ZipFile

from text2filter import proposals_v2, proposals_v3
from avito_text2filter import Catalog
from avito_text2filter.student import student_runner
from avito_text2filter.study import assemble, filter_key


def read_frequencies(archive):
    counts = {}
    with ZipFile(archive) as bundle, io.TextIOWrapper(
        bundle.open("query_frequencies.csv"), encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            query, count = row["query_text"], int(row["search_count"])
            if not query.strip() or count < 1 or query in counts:
                raise ValueError("Expected each nonempty text once with a positive search count")
            counts[query] = count
    if not counts:
        raise ValueError("The frequency table is empty")
    return counts


def audit(counts, methods):
    names = ("with_filter", "with_unresolved", "filter_without_unresolved", "no_filter_or_unresolved")
    modes = {method: {kind: Counter(dict.fromkeys(names, 0)) for kind in ("texts", "search_events")}
             for method in methods}
    transitions = {kind: Counter(dict.fromkeys((
        "filter_set_changed", "v3_gained_a_filter_when_v2_empty", "v3_empty_when_v2_had_filter",
    ), 0)) for kind in ("texts", "search_events")}
    unresolved_reasons = {method: Counter() for method in methods}
    for index, (query, count) in enumerate(counts.items(), 1):
        signatures = {}
        weights = {"texts": 1, "search_events": count}
        for method, run in methods.items():
            result = run(query)
            filters, unresolved = result["filters"], result["unresolved"]
            if result.get("status") == "model_error":
                raise ValueError(f"Inference failed in {method} at input {index}")
            if any(query[row["start"]:row["end"]] != row["evidence"] for row in filters + unresolved):
                raise ValueError(f"Output evidence failed validation in {method} at input {index}")
            signatures[method] = {filter_key(row) for row in filters}
            states = (bool(filters), bool(unresolved), bool(filters and not unresolved),
                      not (filters or unresolved))
            for kind, weight in weights.items():
                modes[method][kind].update({name: value * weight for name, value in zip(names, states, strict=True)})
            unresolved_reasons[method].update({reason: count for reason in {row["reason"] for row in unresolved}})
        v2, v3 = signatures["v2_control"], signatures["v3_verified_qwen_calibrated"]
        for kind, weight in weights.items():
            transitions[kind].update({
                "filter_set_changed": weight * (v2 != v3),
                "v3_gained_a_filter_when_v2_empty": weight * (not v2 and bool(v3)),
                "v3_empty_when_v2_had_filter": weight * (bool(v2) and not v3),
            })
        if index % 5_000 == 0:
            print(f"{index:,}/{len(counts):,} unique texts", flush=True)
    return {"coverage": modes, "v2_to_v3": transitions, "unresolved_reasons_by_search_events": unresolved_reasons}


def fingerprint(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--v2-control", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    counts = read_frequencies(args.archive)
    catalog = Catalog.from_zip(args.archive)
    methods = {
        "rules": catalog.parse,
        "v2_control": student_runner(catalog, args.v2_control, proposals_v2, assemble, 0.5, allow_linked=True),
        "v3_verified_qwen_calibrated": student_runner(
            catalog, args.v3, proposals_v3, assemble, 0.995, allow_linked=True,
        ),
    }
    print(f"Auditing {len(counts):,} unique texts and {sum(counts.values()):,} search events", flush=True)
    result = {
        "scope": "Output behavior across the complete supplied text export, without reference filters",
        "population": {"texts": len(counts), "search_events": sum(counts.values())},
        "input_sha256": fingerprint(args.archive),
        "configuration": {
            "v2_control": {"threshold": 0.5, "model_sha256": fingerprint(args.v2_control / "model.safetensors")},
            "v3_verified_qwen_calibrated": {
                "threshold": 0.995, "model_sha256": fingerprint(args.v3 / "model.safetensors"),
            },
        },
        **audit(counts, methods),
    }
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
