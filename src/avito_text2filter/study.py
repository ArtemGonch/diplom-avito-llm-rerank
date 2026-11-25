"""Deterministic blind sampling and paired evaluation of filter extraction."""

import csv
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
import random
import unicodedata


SEED = 261119


def family_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("ё", "е").split())


def build_packet(archive):
    """Return the private sampling manifest and blank reviewer inputs."""
    families = defaultdict(Counter)
    with io.TextIOWrapper(archive.open("query_frequencies.csv"), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            query = row["query_text"]
            count = int(row["search_count"])
            if not family_key(query) or count < 1:
                raise ValueError("Invalid query frequency")
            families[family_key(query)][query] += count
    ranked = sorted(families, key=lambda key: (-families[key].total(), key))
    strata = (
        ("top10", ranked[:10], 10),
        ("rank11_100", ranked[10:100], 20),
        ("rank101_1000", ranked[100:1000], 30),
        ("other_repeat", [key for key in ranked[1000:] if families[key].total() > 1], 30),
        ("other_singleton", [key for key in ranked[1000:] if families[key].total() == 1], 30),
    )
    rng = random.Random(SEED)
    retained = set()
    manifest = []
    blind = {split: [] for split in ("train", "dev", "test")}

    def add(split, key, stratum=None, probability=None):
        retained.add(key)
        query_id = hashlib.sha256(("text2filter-v1\0" + key).encode()).hexdigest()[:24]
        variants = sorted(families[key].items())
        queries, weights = zip(*variants, strict=True)
        query = rng.choices(queries, weights=weights)[0]
        manifest.append({
            "id": query_id, "split": split, "family_key": key,
            "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "study": {"stratum": stratum, "probability": probability,
                      "family_count": sum(weights)},
        })
        blind[split].append({"id": query_id, "query": query, "filters": None,
                             "unresolved": None, "status": "pending", "notes": ""})

    for stratum, population, n in strata:
        for key in rng.sample(population, n):
            add("test", key, stratum, n / len(population))

    remaining = [key for key in ranked if key not in retained]
    frequent = rng.sample(remaining[:1000], 60)
    tail = rng.sample(remaining[1000:], 60)
    for split, a, b in (("train", 0, 40), ("dev", 40, 60)):
        for key in frequent[a:b] + tail[a:b]:
            add(split, key)

    for rows in blind.values():
        rng.shuffle(rows)
    return manifest, blind


def selected_options(response, options):
    data = json.loads(response)
    if not isinstance(data, dict) or set(data) != {"accept"}:
        raise ValueError("Expected one accept array")
    indexes = data["accept"]
    if not isinstance(indexes, list) or any(type(i) is not int or not 0 <= i < len(options) for i in indexes):
        raise ValueError("Invalid option selection")
    if len(set(indexes)) != len(indexes):
        raise ValueError("Duplicate option")
    return [options[i] for i in sorted(indexes)]


def filter_key(row):
    attr = str(row["attr_id"])
    if row.get("value_id") is not None:
        return attr, "value", str(row["value_id"])
    lo, hi = row.get("min"), row.get("max")
    if (lo is None and hi is None) or any(
        value is not None and (type(value) not in (int, float) or not math.isfinite(value))
        for value in (lo, hi)
    ) or (lo is not None and hi is not None and lo > hi):
        raise ValueError("Invalid numeric filter")
    return (attr, "range", lo, hi,
            row.get("min_inclusive", True) if lo is not None else None,
            row.get("max_inclusive", True) if hi is not None else None)


def evaluate(gold, predicted, manifest):
    def index(rows):
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError("Duplicate query ID")
        return by_id

    gold, predicted = index(gold), index(predicted)
    test = index([row for row in manifest if row["split"] == "test"])
    if not test or gold.keys() != test.keys() or predicted.keys() != test.keys():
        raise ValueError("Expected predictions and annotations for the entire frozen test")
    totals = {"families": Counter(), "traffic": Counter()}
    timings = []
    fallback_queries = 0
    for key, record in test.items():
        a, b = gold[key], predicted[key]
        if a.get("status") == "pending" or a.get("status") == "disputed" or not all(
            isinstance(a.get(field), list) for field in ("filters", "unresolved")
        ):
            raise ValueError("Reference annotations are incomplete")
        fallback_queries += b.get("status") == "model_error"
        query = a["query"]
        if "query_sha256" in record and hashlib.sha256(query.encode()).hexdigest() != record["query_sha256"]:
            raise ValueError("Reference annotation differs from the frozen query")
        if b.get("query", query) != query:
            raise ValueError("Prediction text differs from the reference annotation")
        for row in a["filters"] + a["unresolved"]:
            start, end = row["start"], row["end"]
            if not (type(start) is type(end) is int and 0 <= start < end <= len(query)
                    and query[start:end] == row["evidence"]):
                raise ValueError("Reference evidence must refer to the original query")
        expected = {filter_key(row) for row in a["filters"]}
        actual = {filter_key(row) for row in b["filters"]}
        def residual(row):
            return row["start"], row["end"], row["reason"]
        ea = {residual(row) for row in a["unresolved"]}
        eb = {residual(row) for row in b["unresolved"]}
        weights = record["study"]
        inverse = 1 / weights["probability"]
        for name, weight in (("families", inverse), ("traffic", inverse * weights["family_count"])):
            count = totals[name]
            count["weight"] += weight
            count["exact"] += weight * (actual == expected)
            count["exact_with_unresolved"] += weight * (actual == expected and ea == eb)
            count["applied"] += weight * bool(actual)
            count["empty_gold"] += weight * (not expected)
            count["false_apply"] += weight * (not expected and bool(actual))
            count["tp"] += weight * len(expected & actual)
            count["fp"] += weight * len(actual - expected)
            count["fn"] += weight * len(expected - actual)
        if "latency_ms" in b:
            timings.append(b["latency_ms"])

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    report = {"queries": len(test), "model_fallback_queries": fallback_queries}
    for name, c in totals.items():
        report[name] = {
            "exact_filters": ratio(c["exact"], c["weight"]),
            "exact_with_unresolved": ratio(c["exact_with_unresolved"], c["weight"]),
            "filter_precision": ratio(c["tp"], c["tp"] + c["fp"]),
            "filter_recall": ratio(c["tp"], c["tp"] + c["fn"]),
            "coverage": ratio(c["applied"], c["weight"]),
            "false_apply_on_empty": ratio(c["false_apply"], c["empty_gold"]),
        }
    if timings:
        report["latency_ms"] = latency_summary(timings)
    return report


def latency_summary(timings):
    ordered = sorted(timings)
    return {name: ordered[math.ceil(level * len(ordered)) - 1]
            for name, level in (("p50", .5), ("p95", .95))}


def assemble(baseline, choices):
    spans = {(row["start"], row["end"]) for row in choices}
    if len(spans) != len(choices):
        raise ValueError("Several proposed filters for one fragment")
    values = {}
    for row in choices:
        attr = row["attr_id"]
        if attr in values and values[attr] != row:
            raise ValueError("Several proposed filters for one attribute")
        values[attr] = row
    chosen = {filter_key(row) for row in choices}
    withheld = [
        {key: row[key] for key in ("start", "end", "evidence")} | {"reason": "model_abstained"}
        for row in baseline["filters"] if filter_key(row) not in chosen
    ]
    supported_spans = [
        (part["start"], part["end"])
        for row in choices for part in row.get("evidence_parts", [row])
    ]
    unresolved = []
    for row in baseline["unresolved"]:
        if row["reason"] not in {"unrecognized", "unqualified"}:
            unresolved.append(row)
            continue
        origin, limit = row["start"], row["end"]
        cursor = origin
        gaps = []
        for start, end in sorted(supported_spans):
            if end <= cursor or start >= limit:
                continue
            if start > cursor:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < limit:
            gaps.append((cursor, limit))
        if gaps == [(origin, limit)]:
            unresolved.append(row)
            continue
        for start, end in gaps:
            original = row["evidence"][start - origin:end - origin]
            remaining = original.strip(" ,.;")
            if remaining:
                start += original.index(remaining)
                unresolved.append({"start": start, "end": start + len(remaining),
                                   "evidence": remaining, "reason": row["reason"]})
    return {
        "filters": sorted(choices, key=lambda row: (row["start"], row["end"])),
        "unresolved": sorted(unresolved + withheld, key=lambda row: (row["start"], row["end"])),
    }


def paired_exact_filters(gold, control, augmented, manifest):
    """Compare exact filter sets on a fixed sample and bootstrap within its strata."""
    predictions = [{row["id"]: row for row in rows} for rows in (control, augmented)]
    test = {row["id"]: row["study"] for row in manifest if row["split"] == "test"}
    by_stratum = defaultdict(list)
    for row in gold:
        key = row["id"]
        sample = test[key]
        target = {filter_key(item) for item in row["filters"]}
        a, b = (target == {filter_key(item) for item in result[key]["filters"]}
                for result in predictions)
        inverse = 1 / sample["probability"]
        by_stratum[sample["stratum"]].append((int(b) - int(a), inverse, inverse * sample["family_count"]))

    def mean(rows, weight):
        return math.fsum(row[0] * row[weight] for row in rows) / math.fsum(
            row[weight] for row in rows)

    original = [row for stratum in by_stratum.values() for row in stratum]
    # The high-frequency census contributes unchanged to every resample.
    full = [row for rows in by_stratum.values() for row in rows if row[1] == 1]
    sampled = [[row for row in rows if row[1] != 1] for rows in by_stratum.values()]
    rng = random.Random(SEED)
    draws = [[], []]
    for _ in range(2_000):
        rows = full + [row for stratum in sampled for row in rng.choices(stratum, k=len(stratum))]
        for index in range(2):
            draws[index].append(mean(rows, index + 1))
    result = {label: sum(row[0] == meaning for row in original)
              for label, meaning in (("wins", 1), ("ties", 0), ("losses", -1))}
    for index, name in enumerate(("families", "traffic")):
        values = sorted(draws[index])
        result[name] = {"difference": mean(original, index + 1),
                        "bootstrap_95": [values[49], values[1949]]}
    return result
