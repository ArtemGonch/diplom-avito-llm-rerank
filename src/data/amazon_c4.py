"""Leakage-free utilities for the personalized Amazon-C4 benchmark.

The public Amazon-C4 item pool contains roughly one million products.  The
MemRerank release adds temporal user histories and fixed train/dev/test splits.
This module deliberately keeps retrieval separate from reranking: retrieval
materialises a top-k list without forcing the positive item into it, so
``Recall@k`` remains meaningful and every downstream method sees the same
candidate set.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """A deterministic, dependency-free tokenizer for the BM25 baseline."""

    return TOKEN_RE.findall(str(text).lower())


@dataclass(frozen=True)
class AmazonC4Item:
    item_id: str
    category: str
    metadata: str


@dataclass(frozen=True)
class AmazonC4Query:
    split: str
    qid: int
    query: str
    user_id: str
    positive_item_id: str
    positive_category: str
    grouped_purchase_history: Mapping[str, Sequence[Sequence[object]]]


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def load_query_texts(test_csv: Path) -> dict[int, str]:
    """Load only legal rank-time query text from Amazon-C4.

    ``ori_rating`` and ``ori_review`` are intentionally ignored because the
    dataset documentation marks them as reference-only fields derived from the
    target event.
    """

    out: dict[int, str] = {}
    with test_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            qid = int(row["qid"])
            if qid in out:
                raise ValueError(f"Duplicate qid={qid} in {test_csv}")
            out[qid] = row["query"]
    return out


def load_items(item_pool: Path, category: str) -> list[AmazonC4Item]:
    expected = category.casefold()
    items: list[AmazonC4Item] = []
    seen: set[str] = set()
    for row in iter_jsonl(item_pool):
        if str(row.get("category", "")).casefold() != expected:
            continue
        item_id = str(row["item_id"])
        if item_id in seen:
            raise ValueError(f"Duplicate item_id={item_id} in {item_pool}")
        seen.add(item_id)
        items.append(
            AmazonC4Item(
                item_id=item_id,
                category=str(row["category"]),
                metadata=str(row.get("metadata", "")),
            )
        )
    if not items:
        raise ValueError(f"No category={category!r} items found in {item_pool}")
    return items


def load_queries(
    history_root: Path,
    category: str,
    splits: Sequence[str],
    query_texts: Mapping[int, str],
) -> list[AmazonC4Query]:
    queries: list[AmazonC4Query] = []
    seen_keys: set[tuple[str, int, str]] = set()
    for split in splits:
        path = history_root / category / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            qid = int(row["query"])
            if qid not in query_texts:
                raise ValueError(f"qid={qid} from {path} is absent in query CSV")
            key = (split, qid, str(row["user_id"]))
            if key in seen_keys:
                raise ValueError(f"Duplicate query key={key}")
            seen_keys.add(key)
            queries.append(
                AmazonC4Query(
                    split=split,
                    qid=qid,
                    query=query_texts[qid],
                    user_id=str(row["user_id"]),
                    positive_item_id=str(row["pos_product"]),
                    positive_category=str(row["pos_product_category"]),
                    grouped_purchase_history=row.get("grouped_purchase_history", {}),
                )
            )
    return queries


class BM25Index:
    """Small inverted BM25 index suitable for a category-sized item pool."""

    def __init__(
        self,
        documents: Sequence[str],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25 requires at least one document")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 expects k1>0 and 0<=b<=1")
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_lengths = np.empty(len(documents), dtype=np.float32)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, document in enumerate(documents):
            counts = Counter(tokenize(document))
            self.doc_lengths[doc_id] = sum(counts.values())
            for token, frequency in counts.items():
                postings[token].append((doc_id, frequency))
        self.avg_doc_length = float(np.mean(self.doc_lengths)) or 1.0
        n_documents = len(documents)
        self.postings = {
            token: (
                np.fromiter((doc_id for doc_id, _ in rows), dtype=np.int32),
                np.fromiter((frequency for _, frequency in rows), dtype=np.float32),
            )
            for token, rows in postings.items()
        }
        self.idf = {
            token: math.log(
                1.0
                + (n_documents - len(doc_ids) + 0.5)
                / (len(doc_ids) + 0.5)
            )
            for token, (doc_ids, _) in self.postings.items()
        }
        self.term_weights: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for token, (doc_ids, frequencies) in self.postings.items():
            length_norm = 1.0 - self.b + self.b * (
                self.doc_lengths[doc_ids] / self.avg_doc_length
            )
            weights = (
                self.idf[token]
                * frequencies
                * (self.k1 + 1.0)
                / (frequencies + self.k1 * length_norm)
            )
            self.term_weights[token] = (doc_ids, weights.astype(np.float32, copy=False))

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.doc_lengths), dtype=np.float32)
        query_counts = Counter(tokenize(query))
        for token, query_frequency in query_counts.items():
            row = self.term_weights.get(token)
            if row is None:
                continue
            doc_ids, weights = row
            scores[doc_ids] += query_frequency * weights
        return scores


def stable_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Return score-descending indices with document-id tie breaking."""

    values = np.asarray(scores)
    if values.ndim != 1:
        raise ValueError("stable_top_k expects a one-dimensional score array")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, len(values))
    if k == len(values):
        candidates = np.arange(len(values))
    else:
        candidates = np.argpartition(-values, k - 1)[:k]
    return candidates[np.lexsort((candidates, -values[candidates]))]


def ranking_metrics(
    ranked_item_ids: Sequence[str],
    positive_item_id: str,
    *,
    k: int = 10,
) -> dict[str, float]:
    """Single-positive retrieval/reranking metrics."""

    try:
        rank = ranked_item_ids.index(positive_item_id) + 1
    except ValueError:
        rank = None
    return {
        f"recall@{len(ranked_item_ids)}": float(rank is not None),
        f"hit@{k}": float(rank is not None and rank <= k),
        f"mrr@{k}": 1.0 / rank if rank is not None and rank <= k else 0.0,
        f"ndcg@{k}": 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0,
    }


def aggregate_metric_rows(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    collected: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            collected[key].append(float(value))
    return {key: float(np.mean(values)) for key, values in sorted(collected.items())}


def select_main_image_url(images: object) -> str | None:
    """Select one MAIN image using the agreed large→hi_res→thumb policy."""

    if not isinstance(images, list):
        return None
    main = next(
        (
            image
            for image in images
            if isinstance(image, dict) and str(image.get("variant", "")).upper() == "MAIN"
        ),
        None,
    )
    if main is None:
        return None
    for field in ("large", "hi_res", "thumb"):
        value = main.get(field)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


def image_filename(item_id: str) -> str:
    """Return a traversal-safe deterministic JPEG filename for an item id."""

    value = str(item_id)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", value).strip("._")
    if not safe:
        safe = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{safe}.jpg"
