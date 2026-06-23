"""Batched offline LLM calls for knowledge stage."""

from __future__ import annotations

import hashlib

from tqdm import tqdm


def knowledge_shard_index(key: str, num_shards: int) -> int:
    """Stable shard assignment (same key -> same shard across processes)."""
    if num_shards <= 1:
        return 0
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards


def job_belongs_to_shard(key: str, shard_id: int, num_shards: int) -> bool:
    if num_shards <= 1:
        return True
    return knowledge_shard_index(key, num_shards) == shard_id


def filter_jobs_for_shard(
    jobs: list[tuple[str, str]], shard_id: int, num_shards: int
) -> list[tuple[str, str]]:
    if num_shards <= 1:
        return jobs
    return [(k, p) for k, p in jobs if job_belongs_to_shard(k, shard_id, num_shards)]


def _prioritize_then_cap(ids: list, required: set | None, cap: int | None, label: str) -> list:
    ordered = list(ids)
    if required:
        req_sorted = sorted(required)
        rest = [x for x in ordered if x not in required]
        ordered = req_sorted + rest
    if cap is None or cap <= 0 or len(ordered) <= cap:
        return ordered
    n_req = len(required) if required else 0
    print(
        f"Knowledge cap: {len(ordered)} -> {cap} {label} via LLM "
        f"({n_req} required by samples prioritized; missing keys use template fallback in train)"
    )
    return ordered[:cap]


def cap_knowledge_ids(
    item_ids: list,
    user_ids: list,
    llm_cfg: dict,
    *,
    sample_item_ids: set | None = None,
    sample_user_ids: set | None = None,
) -> tuple[list, list]:
    cap_i = llm_cfg.get("knowledge_max_items")
    cap_u = llm_cfg.get("knowledge_max_users")
    return (
        _prioritize_then_cap(item_ids, sample_item_ids, cap_i, "items"),
        _prioritize_then_cap(user_ids, sample_user_ids, cap_u, "users"),
    )


def estimate_knowledge_hours(
    n_items: int,
    n_users: int,
    *,
    sec_per_call: float = 55.0,
    batch_size: int = 1,
) -> float:
    n = n_items + n_users
    if n == 0:
        return 0.0
    bs = max(1, batch_size)
    return (n / bs) * sec_per_call / 3600.0


def llm_generate_loop(
    gen,
    jobs: list[tuple[str, str]],
    target: dict[str, str],
    *,
    batch_method: str,
    llm_cfg: dict,
    users: dict[str, str],
    items: dict[str, str],
    store,
    desc: str,
) -> None:
    """Run batched LLM for (key, prompt) pairs missing from target."""
    batch_size = max(1, int(llm_cfg.get("batch_size", getattr(gen, "batch_size", 1))))
    ckpt_every = int(llm_cfg.get("checkpoint_every", 50))
    batch_fn = getattr(gen, batch_method)
    todo = [(k, p) for k, p in jobs if k not in target]
    if not todo:
        return

    pending_k: list[str] = []
    pending_p: list[str] = []
    n_new = 0
    with tqdm(total=len(todo), desc=desc) as pbar:
        for key, prompt in todo:
            pending_k.append(key)
            pending_p.append(prompt)
            if len(pending_p) < batch_size:
                continue
            texts = batch_fn(pending_p)
            for k, text in zip(pending_k, texts):
                target[k] = text
            n_new += len(pending_k)
            pbar.update(len(pending_k))
            if ckpt_every > 0 and n_new % ckpt_every == 0:
                store.save_partial(users, items)
            pending_k, pending_p = [], []

        if pending_p:
            texts = batch_fn(pending_p)
            for k, text in zip(pending_k, texts):
                target[k] = text
            n_new += len(pending_k)
            pbar.update(len(pending_k))
            if ckpt_every > 0 and n_new % ckpt_every == 0:
                store.save_partial(users, items)
