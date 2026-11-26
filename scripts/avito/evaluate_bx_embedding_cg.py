#!/usr/bin/env python3
"""Evaluate precomputed query and item embeddings inside the BX candidate pool."""

import argparse
import json
import math
import sqlite3
import sys
from contextlib import closing, contextmanager
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pyarrow import parquet

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from avito_bx_baseline import PRIMARY_LABEL, TARGETS, _observed_gain
from avito_bx_embedding_cg import compare_embedding_cg


def valid_id(value):
    return type(value) is int and value >= 0 or isinstance(value, str) and bool(value.strip())


class EmbeddingIndex:
    def __init__(self, directory, key, split, candidate_digest):
        with parquet.ParquetFile(directory / "keys.parquet") as source:
            metadata = source.schema_arrow.metadata or {}
            self.model = metadata.get(b"embedding_model", b"").decode().strip()
            if not self.model or source.schema_arrow.names.count(key) != 1:
                raise ValueError("Expected a unique key column and embedding_model metadata")
            if metadata.get(b"bx_candidate_sha256") != candidate_digest.encode() or metadata.get(
                b"bx_split"
            ) != split.encode():
                raise ValueError("Embeddings do not match the candidate split and SHA-256")
            keys = (value for batch in source.iter_batches(columns=[key]) for value in batch[0].to_pylist())
            self.rows = {}
            for index, value in enumerate(keys):
                if not valid_id(value) or value in self.rows:
                    raise ValueError("Invalid or duplicate embedding key")
                self.rows[value] = index

        self.vectors = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
        if (
            self.vectors.dtype.kind != "f" or self.vectors.ndim != 2
            or self.vectors.shape[0] != len(self.rows) or not self.rows or not self.vectors.shape[1]
        ):
            raise ValueError("vectors.npy must be a nonempty floating-point matrix aligned to keys.parquet")
        self.dimensions = self.vectors.shape[1]
        self.norms = np.empty(len(self.rows), dtype=np.float64)
        for start in range(0, len(self.rows), 4096):
            block = self.vectors[start:start + 4096]
            norms = np.sqrt(np.einsum("ij,ij->i", block, block, dtype=np.float64))
            if not np.isfinite(norms).all() or (norms == 0).any():
                raise ValueError("Embedding matrix contains invalid or zero vectors")
            self.norms[start:start + len(block)] = norms

    def get(self, key):
        index = self.rows.get(key)
        if index is not None:
            return self.vectors[index] / self.norms[index]
        return None


def candidate_rows(path, item_key, label, split):
    columns = ["serp_x", "source_row_index", item_key, "baseline_score", label]
    with parquet.ParquetFile(path) as source:
        available = source.schema_arrow.names
        if len(set(available)) != len(available) or not set(columns) <= set(available):
            raise ValueError("Candidate Parquet is missing ranking columns or has duplicate fields")
        has_split = "split" in available
        selected = [*columns, "split"] if has_split else columns
        for batch in source.iter_batches(columns=selected):
            for group, index, item, score, gain, *embedded in zip(
                *(array.to_pylist() for array in batch.columns), strict=True
            ):
                if embedded and embedded != [split]:
                    raise ValueError("Candidate split differs from the declared split")
                if not valid_id(group) or not valid_id(item) or type(index) is not int or index < 0:
                    raise ValueError("Invalid search, candidate or source order key")
                if type(score) not in (int, float) or not math.isfinite(score):
                    raise ValueError("Missing or non-finite baseline score")
                if isinstance(gain, bool):
                    raise ValueError("Expected numeric gain, not a boolean column")
                yield group, index, item, score, _observed_gain(gain)


def candidate_fingerprint(paths):
    fingerprints = []
    for path in paths:
        with path.open("rb") as stream:
            fingerprints.append(file_digest(stream, "sha256").hexdigest())
    if len(fingerprints) == 1:
        return fingerprints[0]
    return sha256("\n".join(sorted(fingerprints)).encode()).hexdigest()


@contextmanager
def ordered_candidates(rows):
    """Group across Parquet row groups and files without loading all candidates into RAM."""
    with TemporaryDirectory(prefix="bx-embedding-candidates-") as scratch:
        with closing(sqlite3.connect(Path(scratch) / "rows.sqlite3")) as db, db:
            db.execute("PRAGMA temp_store = FILE")
            db.execute("""
                CREATE TABLE candidates (
                    source_index BLOB PRIMARY KEY, serp TEXT NOT NULL, payload TEXT NOT NULL
                ) WITHOUT ROWID
            """)
            try:
                db.executemany("INSERT INTO candidates VALUES (?, ?, ?)", (
                    (index.to_bytes(8, "big"), json.dumps(serp), json.dumps((item, score, gain), allow_nan=False))
                    for serp, index, item, score, gain in rows
                ))
            except sqlite3.IntegrityError as error:
                raise ValueError("Duplicate source_row_index in candidate split") from error
            db.execute("CREATE INDEX rows_by_serp ON candidates (serp, source_index)")
            yield (
                (json.loads(serp), int.from_bytes(index, "big"), *json.loads(payload))
                for serp, index, payload in db.execute(
                    "SELECT serp, source_index, payload FROM candidates ORDER BY serp, source_index"
                )
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, action="extend", nargs="+", required=True, help="Parquet shards of one split",
    )
    parser.add_argument("--split", required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--item-embeddings", type=Path, required=True)
    parser.add_argument("--item-key", choices=("snapshot_id", "item_id"), default="snapshot_id")
    parser.add_argument("--label", choices=TARGETS, default=PRIMARY_LABEL)
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20, 50])
    args = parser.parse_args()
    if not args.split.strip():
        parser.error("Expected a nonblank candidate split")
    digest = candidate_fingerprint(args.candidates)
    queries = EmbeddingIndex(args.query_embeddings, "serp_x", args.split, digest)
    items = EmbeddingIndex(args.item_embeddings, args.item_key, args.split, digest)
    if queries.model != items.model or queries.dimensions != items.dimensions:
        parser.error("Query and item embeddings must use the same model and dimension")
    rows = (
        row for path in args.candidates for row in candidate_rows(path, args.item_key, args.label, args.split)
    )
    with ordered_candidates(rows) as ordered:
        result = compare_embedding_cg(ordered, queries, items, args.k)
    print(json.dumps({
        "split": args.split,
        "label": args.label,
        "candidate_sha256": digest,
        "embedding_model": queries.model,
        "dimensions": queries.dimensions,
        **result,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
