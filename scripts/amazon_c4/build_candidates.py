#!/usr/bin/env python3
"""Materialise fixed BM25 or BLaIR top-k candidates for Amazon-C4 Automotive."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.amazon_c4 import (  # noqa: E402
    BM25Index,
    aggregate_metric_rows,
    load_items,
    load_queries,
    load_query_texts,
    ranking_metrics,
    stable_top_k,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _normalise_embeddings(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def _sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _encode_blair(
    texts: Sequence[str],
    *,
    model_name: str,
    model_revision: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
    model = AutoModel.from_pretrained(model_name, revision=model_revision).to(device).eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded, return_dict=True).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden.float(), dim=1)
        chunks.append(hidden.cpu().numpy())
        done = min(start + batch_size, len(texts))
        print(f"BLaIR encoded {done}/{len(texts)}", flush=True)
    return np.concatenate(chunks, axis=0)


def _load_or_encode_blair_items(
    texts: Sequence[str],
    item_ids: Sequence[str],
    *,
    cache_dir: Path,
    model_name: str,
    model_revision: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = cache_dir / "automotive_item_embeddings.npy"
    metadata_path = cache_dir / "automotive_item_embeddings.meta.json"
    expected = {
        "model": model_name,
        "model_revision": model_revision,
        "max_length": max_length,
        "n_items": len(item_ids),
        "item_ids_sha256": _sequence_sha256(item_ids),
        "item_texts_sha256": _sequence_sha256(texts),
    }
    if embeddings_path.exists() and metadata_path.exists():
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual == expected:
            cached = np.load(embeddings_path)
            if cached.shape[0] == len(item_ids):
                print(f"Reusing BLaIR item cache: {embeddings_path}")
                return _normalise_embeddings(cached.astype(np.float32, copy=False))
    embeddings = _encode_blair(
        texts,
        model_name=model_name,
        model_revision=model_revision,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    np.save(embeddings_path, embeddings.astype(np.float16))
    _write_json(metadata_path, expected)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("bm25", "blair"), required=True)
    parser.add_argument("--category", default="Automotive")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--item-pool", type=Path, default=ROOT / "data/amazon-c4/sampled_item_metadata_1M.jsonl")
    parser.add_argument("--query-csv", type=Path, default=ROOT / "data/amazon-c4/test.csv")
    parser.add_argument("--history-root", type=Path, default=ROOT / "data/amazon-c4-user-purchase-history")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/amazon-c4-automotive/candidates")
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--model", default="hyp1231/blair-roberta-base")
    parser.add_argument(
        "--model-revision",
        default="88adfe2b621cb202dee9aabb19b59de8f622844c8",
        help="Pinned Hugging Face commit for reproducible BLaIR embeddings.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/amazon-c4-automotive/cache/blair")
    args = parser.parse_args()

    started = time.time()
    query_texts = load_query_texts(args.query_csv)
    items = load_items(args.item_pool, args.category)
    queries = load_queries(args.history_root, args.category, args.splits, query_texts)
    item_ids = [item.item_id for item in items]
    item_id_set = set(item_ids)
    missing_positives = sorted({query.positive_item_id for query in queries} - item_id_set)
    if missing_positives:
        raise ValueError(f"{len(missing_positives)} positives absent from item pool")
    print(f"Loaded {len(items)} {args.category} items and {len(queries)} queries")

    score_rows: list[np.ndarray]
    if args.method == "bm25":
        index = BM25Index([item.metadata for item in items])
        score_rows = [index.scores(query.query) for query in queries]
    else:
        item_embeddings = _load_or_encode_blair_items(
            [item.metadata for item in items],
            item_ids,
            cache_dir=args.cache_dir,
            model_name=args.model,
            model_revision=args.model_revision,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        query_embeddings = _encode_blair(
            [query.query for query in queries],
            model_name=args.model,
            model_revision=args.model_revision,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
        score_rows = [row for row in query_embeddings @ item_embeddings.T]

    output_by_split: dict[str, list[dict]] = defaultdict(list)
    metric_rows_by_split: dict[str, list[dict[str, float]]] = defaultdict(list)
    for query, scores in zip(queries, score_rows):
        top_indices = stable_top_k(scores, args.top_k)
        ranked_ids = [item_ids[int(index)] for index in top_indices]
        candidates = [
            {
                "item_id": item_ids[int(index)],
                "rank": rank,
                "retrieval_score": float(scores[int(index)]),
            }
            for rank, index in enumerate(top_indices, start=1)
        ]
        output_by_split[query.split].append(
            {
                "protocol_version": 1,
                "method": args.method,
                "category": args.category,
                "split": query.split,
                "qid": query.qid,
                "user_id": query.user_id,
                "query": query.query,
                "positive_item_id": query.positive_item_id,
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        )
        metric_rows_by_split[query.split].append(
            ranking_metrics(ranked_ids, query.positive_item_id, k=10)
        )

    metrics = {
        "protocol": {
            "version": 1,
            "method": args.method,
            "category": args.category,
            "top_k": args.top_k,
            "positive_forced_into_candidates": False,
            "query_fields": ["query"],
            "forbidden_reference_fields": ["ori_rating", "ori_review"],
            "item_fields": ["metadata"],
            "model": args.model if args.method == "blair" else None,
            "model_revision": args.model_revision if args.method == "blair" else None,
        },
        "counts": {
            "items": len(items),
            "queries": len(queries),
            "by_split": {split: len(output_by_split[split]) for split in args.splits},
        },
        "metrics": {
            split: aggregate_metric_rows(metric_rows_by_split[split]) for split in args.splits
        },
        "elapsed_seconds": time.time() - started,
    }

    for split in args.splits:
        _write_jsonl(args.output_dir / f"{args.method}_{split}_top{args.top_k}.jsonl", output_by_split[split])
    metrics_output = args.metrics_output or args.output_dir / f"{args.method}_metrics_top{args.top_k}.json"
    _write_json(metrics_output, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote candidates under {args.output_dir}")


if __name__ == "__main__":
    main()
