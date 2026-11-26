"""Compare exact cosine selection with BX inside the same supplied candidate pool."""

import math
from collections import Counter, defaultdict
from itertools import groupby

import numpy as np

from avito_bx_baseline import _observed_gain, _ranked_group


def _record_group(cohort, metrics):
    for method, measurements in metrics.items():
        for metric, value in measurements.items():
            if value is not None:
                cohort[method][metric].append(value)


def _averages(cohort, names):
    baseline = cohort["bx"]
    return {
        "recall_groups": len(baseline[names[0]]),
        "bx_ndcg_groups": len(baseline["bx_ndcg"]),
        **{
            method: {
                name: math.fsum(values[name]) / len(values[name]) if values[name] else None
                for name in (*names, "bx_ndcg")
            }
            for method, values in cohort.items()
        },
    }


def _group_metrics(scores, gains, cutoffs):
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    positive = [gain is not None and gain > 0 for gain in gains]
    count = sum(positive)
    return {
        **{f"recall@{k}": sum(positive[i] for i in order[:k]) / count if count else None for k in cutoffs},
        "bx_ndcg": _ranked_group([(score, gain) for score, gain in zip(scores, gains) if gain is not None])[0],
    }


def compare_embedding_cg(rows, queries, items, cutoffs):
    """Consume one split with contiguous groups and normalized query/item vector lookups."""
    if not cutoffs or len(set(cutoffs)) != len(cutoffs) or any(type(k) is not int or k <= 0 for k in cutoffs):
        raise ValueError("Cutoffs must be distinct positive integers")
    methods = ("bx", "embedding_cg")
    all_groups = {name: defaultdict(list) for name in methods}
    complete_groups = {name: defaultdict(list) for name in methods}
    counts = Counter(dict.fromkeys((
        "rows", "groups", "groups_with_all_embeddings",
        "groups_missing_query_embedding", "groups_missing_item_embedding",
    ), 0))
    seen = set()
    for group, source in groupby(rows, key=lambda row: row[0]):
        if group in seen:
            raise ValueError("Expected contiguous rows per search group")
        seen.add(group)
        _, row_ids, item_ids, baseline, labels = zip(*sorted(source, key=lambda row: row[1]))
        if len(set(row_ids)) != len(row_ids) or len(set(item_ids)) != len(item_ids):
            raise ValueError("Found a duplicate candidate in a search group")
        if any(not math.isfinite(score) for score in baseline):
            raise ValueError("Baseline scores must be finite")
        gains = tuple(map(_observed_gain, labels))
        query = queries.get(group)
        vectors = [items.get(key) for key in item_ids]
        missing_items = any(vector is None for vector in vectors)
        complete = query is not None and not missing_items
        counts.update({
            "rows": len(baseline),
            "groups": 1,
            "groups_with_all_embeddings": int(complete),
            "groups_missing_query_embedding": int(query is None),
            "groups_missing_item_embedding": int(missing_items),
        })
        baseline_metrics = _group_metrics(baseline, gains, cutoffs)
        metrics = {"bx": baseline_metrics, "embedding_cg": baseline_metrics}
        if complete:
            scores = np.stack(vectors) @ query
            metrics["embedding_cg"] = _group_metrics(scores, gains, cutoffs)
            _record_group(complete_groups, metrics)
        _record_group(all_groups, metrics)

    if not counts["rows"]:
        raise ValueError("No candidate rows")
    names = tuple(f"recall@{k}" for k in sorted(cutoffs))
    return {
        "protocol": "Exact cosine inside the supplied BX pool, with BX fallback for incomplete groups",
        "score_tie_break": "source_row_index ascending",
        **counts,
        "all_groups": _averages(all_groups, names),
        "complete_groups": _averages(complete_groups, names),
    }
