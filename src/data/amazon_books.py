"""Amazon-Books 5-core (UR4Rec paper Table 1, Appendix C)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .download_utils import download_file, parse_json_line
from .sequential import build_rerank_samples_from_sequences, split_users

# UCSD McAuley Lab — Amazon 2018, Books 5-core (direct download)
# Index: https://nijianmo.github.io/amazon/index.html
_AMAZON_FILES = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles"
)
AMAZON_BOOKS_REVIEWS_URL = f"{_AMAZON_FILES}/reviews_Books_5.json.gz"
AMAZON_BOOKS_META_URL = f"{_AMAZON_FILES}/meta_Books.json.gz"
REVIEWS_FILENAME = "reviews_Books_5.json.gz"
META_FILENAME = "meta_Books.json.gz"


def download_amazon_books(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    reviews = data_dir / REVIEWS_FILENAME
    meta = data_dir / META_FILENAME
    if reviews.exists() and meta.exists():
        return data_dir
    if not reviews.exists():
        print(f"Amazon-Books 5-core reviews (~3 GB) -> {reviews}")
        download_file(AMAZON_BOOKS_REVIEWS_URL, reviews)
    if not meta.exists():
        print(f"Amazon-Books metadata (~780 MB) -> {meta}")
        download_file(AMAZON_BOOKS_META_URL, meta)
    return data_dir


class AmazonBooks:
    """Sequential rerank dataset from Amazon Books 5-core reviews + meta."""

    def __init__(
        self,
        root: Path,
        min_rating: float = 4.0,
        max_reviews: int | None = None,
    ):
        self.root = Path(root)
        self.items = self._load_meta()
        self.ratings = self._load_ratings(min_rating, max_reviews)
        self.user_items = self._build_user_items()
        self.num_users = max(self.user_items) if self.user_items else 0
        self.num_items = int(self.items.index.max()) if len(self.items) else 0

    def _load_meta(self) -> pd.DataFrame:
        path = self.root / META_FILENAME
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                d = parse_json_line(line)
                if not d:
                    continue
                asin = d.get("asin")
                if not asin:
                    continue
                title = d.get("title") or asin
                cats = d.get("category") or d.get("categories") or []
                if isinstance(cats, list) and cats and isinstance(cats[0], list):
                    cat = " > ".join(str(c) for c in cats[0] if c)
                elif isinstance(cats, list):
                    cat = " > ".join(str(c) for c in cats if c)
                else:
                    cat = str(cats)
                brand = d.get("brand") or d.get("manufacturer") or ""
                rows.append(
                    {
                        "asin": asin,
                        "title": str(title),
                        "category": cat or "Books",
                        "brand": str(brand) if brand else "unknown",
                    }
                )
        df = pd.DataFrame(rows).drop_duplicates("asin")
        df["item_id"] = np.arange(1, len(df) + 1, dtype=np.int64)
        self._asin_to_id = dict(zip(df["asin"], df["item_id"]))
        self._id_to_asin = dict(zip(df["item_id"], df["asin"]))
        return df.set_index("item_id")

    def _load_ratings(self, min_rating: float, max_reviews: int | None) -> pd.DataFrame:
        path = self.root / REVIEWS_FILENAME
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_reviews is not None and i >= max_reviews:
                    break
                d = parse_json_line(line)
                if not d:
                    continue
                rating = float(d.get("overall", 0))
                if rating < min_rating:
                    continue
                asin = d.get("asin")
                uid = d.get("reviewerID")
                ts = int(d.get("unixReviewTime", 0))
                if not asin or not uid or asin not in self._asin_to_id:
                    continue
                rows.append(
                    {
                        "user_key": str(uid),
                        "item_id": int(self._asin_to_id[asin]),
                        "rating": rating,
                        "timestamp": ts,
                    }
                )
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No Amazon-Books ratings loaded; check data files.")
        user_keys = sorted(df["user_key"].unique())
        self._user_key_to_id = {k: i + 1 for i, k in enumerate(user_keys)}
        df["user_id"] = df["user_key"].map(self._user_key_to_id).astype(np.int64)
        return df.sort_values("timestamp")

    def _build_user_items(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for uid, grp in self.ratings.groupby("user_id"):
            out[int(uid)] = grp["item_id"].astype(int).tolist()
        return out

    def item_meta(self, item_id: int) -> tuple[str, str]:
        row = self.items.loc[item_id]
        title = str(row["title"])
        cat = str(row["category"])
        brand = str(row["brand"])
        return title, f"{cat}; brand: {brand}"

    def train_val_test_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[int], list[int], list[int]]:
        return split_users(sorted(self.user_items.keys()), train_ratio, val_ratio, seed)

    def build_rerank_samples(
        self,
        user_ids: list[int],
        history_len: int,
        num_candidates: int,
        seed: int = 42,
    ):
        return build_rerank_samples_from_sequences(
            self.user_items,
            user_ids,
            self.items.index.values,
            history_len=history_len,
            num_candidates=num_candidates,
            seed=seed,
        )
