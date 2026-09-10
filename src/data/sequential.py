"""Shared reranking sample builder for sequential interaction datasets."""

from __future__ import annotations

import numpy as np

from .ml1m import RerankSample


def build_rerank_samples_from_sequences(
    user_items: dict[int, list[int]],
    user_ids: list[int],
    all_item_ids: np.ndarray,
    *,
    history_len: int,
    num_candidates: int,
    seed: int = 42,
) -> list[RerankSample]:
    rng = np.random.default_rng(seed)
    all_items = np.asarray(all_item_ids, dtype=np.int64)
    samples: list[RerankSample] = []

    for uid in user_ids:
        seq = user_items.get(uid, [])
        if len(seq) < 2:
            continue
        target = seq[-1]
        history = seq[-(history_len + 1) : -1]
        seen = set(seq)
        negs = [int(i) for i in all_items if int(i) not in seen]
        rng.shuffle(negs)
        negs = negs[: num_candidates - 1]
        cands = [target] + negs
        rng.shuffle(cands)
        labels = [1 if c == target else 0 for c in cands]
        samples.append(
            RerankSample(
                user_id=uid,
                history_item_ids=history,
                candidate_item_ids=cands,
                labels=labels,
                target_item_id=target,
            )
        )
    return samples


def split_users(
    user_ids: list[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    users = list(user_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n = len(users)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return users[:n_train], users[n_train : n_train + n_val], users[n_train + n_val :]
