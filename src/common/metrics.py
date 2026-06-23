"""MAP@K and NDCG@K for reranking."""

from __future__ import annotations

import numpy as np


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = relevances[:k]
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2))
    return float(np.sum((2**rel - 1) / discounts))


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(-y_score)
    rel_bin = (np.asarray(y_true) > 0).astype(np.float64)
    rel = rel_bin[order]
    dcg = dcg_at_k(rel, k)
    ideal = dcg_at_k(np.sort(rel_bin)[::-1], k)
    return dcg / ideal if ideal > 0 else 0.0


def map_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    order = np.argsort(-y_score)[:k]
    hits, ap, n_rel = 0, 0.0, int((y_true > 0).sum())
    if n_rel == 0:
        return 0.0
    for i, idx in enumerate(order, start=1):
        if y_true[idx] > 0:
            hits += 1
            ap += hits / i
    return ap / min(n_rel, k)


def evaluate_batch(
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for k in ks:
        out[f"ndcg@{k}"] = []
        out[f"map@{k}"] = []
    for scores, labels in zip(scores_list, labels_list):
        for k in ks:
            out[f"ndcg@{k}"].append(ndcg_at_k(labels, scores, k))
            out[f"map@{k}"].append(map_at_k(labels, scores, k))
    return {k: float(np.mean(v)) for k, v in out.items()}
