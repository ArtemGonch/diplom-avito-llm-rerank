"""MAP@K and NDCG@K for reranking."""

from __future__ import annotations

import numpy as np


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = relevances[:k]
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2))
    return float(np.sum((2**rel - 1) / discounts))


def _score_order(y_score: np.ndarray, item_ids: np.ndarray | None = None) -> np.ndarray:
    """Return a deterministic descending-score order.

    Candidate identifiers are required whenever an evaluation contract declares
    ``item_id`` as its tie-break. Callers without such a contract retain a
    stable input-order tie-break for backwards compatibility.
    """
    scores = np.asarray(y_score, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("Ranking scores must be one-dimensional")
    if item_ids is None:
        return np.argsort(-scores, kind="stable")
    identifiers = np.asarray(item_ids)
    if identifiers.ndim != 1 or identifiers.shape != scores.shape:
        raise ValueError("item_ids and scores must be one-dimensional and have equal length")
    if np.unique(identifiers).size != identifiers.size:
        raise ValueError("item_ids must be unique within each ranking group")
    return np.lexsort((identifiers, -scores))


def ndcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
    item_ids: np.ndarray | None = None,
) -> float:
    """NDCG@k with graded relevance.

    Binary inputs retain the previous behaviour.  Continuous relevance signals,
    such as normalised Avito contacts, are no longer collapsed to all-positive.
    """
    order = _score_order(y_score, item_ids)
    rel_all = np.asarray(y_true, dtype=np.float64)
    if np.any(rel_all < 0):
        raise ValueError("NDCG relevance labels must be non-negative")
    rel = rel_all[order]
    dcg = dcg_at_k(rel, k)
    ideal = dcg_at_k(np.sort(rel_all)[::-1], k)
    return dcg / ideal if ideal > 0 else 0.0


def map_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
    item_ids: np.ndarray | None = None,
) -> float:
    order = _score_order(y_score, item_ids)[:k]
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
    item_ids_list: list[np.ndarray] | None = None,
) -> dict[str, float]:
    if len(scores_list) != len(labels_list):
        raise ValueError("scores_list and labels_list must have equal length")
    if item_ids_list is None:
        item_ids: list[np.ndarray | None] = [None] * len(scores_list)
    else:
        if len(item_ids_list) != len(scores_list):
            raise ValueError("item_ids_list and scores_list must have equal length")
        item_ids = list(item_ids_list)
    out: dict[str, list[float]] = {}
    for k in ks:
        out[f"ndcg@{k}"] = []
        out[f"map@{k}"] = []
    for scores, labels, identifiers in zip(scores_list, labels_list, item_ids):
        if np.asarray(scores).shape != np.asarray(labels).shape:
            raise ValueError("Scores and labels must have equal shape within each group")
        for k in ks:
            out[f"ndcg@{k}"].append(ndcg_at_k(labels, scores, k, identifiers))
            out[f"map@{k}"].append(map_at_k(labels, scores, k, identifiers))
    return {k: float(np.mean(v)) for k, v in out.items()}
