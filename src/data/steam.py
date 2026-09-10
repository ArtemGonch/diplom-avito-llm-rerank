"""Steam reviews dataset (UR4Rec paper; SASRec / Kang et al.)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .download_utils import download_file, parse_json_line
from .sequential import build_rerank_samples_from_sequences, split_users

STEAM_REVIEWS_URL = "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz"
STEAM_GAMES_URL = "http://cseweb.ucsd.edu/~wckang/steam_games.json.gz"


def download_steam(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    reviews = data_dir / "steam_reviews.json.gz"
    games = data_dir / "steam_games.json.gz"
    if reviews.exists() and games.exists():
        return data_dir
    if not games.exists():
        print(f"Steam games metadata -> {games}")
        download_file(STEAM_GAMES_URL, games)
    if not reviews.exists():
        print(f"Steam reviews (~1.3 GB) -> {reviews}")
        download_file(STEAM_REVIEWS_URL, reviews)
    return data_dir


class SteamReviews:
    """Sequential rerank from Steam play/review logs + game metadata."""

    def __init__(
        self,
        root: Path,
        min_hours: float = 1.0,
        max_reviews: int | None = None,
    ):
        self.root = Path(root)
        self.items = self._load_games()
        self.ratings = self._load_reviews(min_hours, max_reviews)
        self.user_items = self._build_user_items()
        self.num_users = max(self.user_items) if self.user_items else 0
        self.num_items = int(self.items.index.max()) if len(self.items) else 0

    def _load_games(self) -> pd.DataFrame:
        path = self.root / "steam_games.json.gz"
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                d = parse_json_line(line)
                if not d:
                    continue
                gid = str(d.get("id") or d.get("app_id") or "")
                if not gid:
                    continue
                title = d.get("app_name") or d.get("title") or gid
                genres = d.get("genres") or []
                if isinstance(genres, list):
                    gstr = ", ".join(str(g) for g in genres)
                else:
                    gstr = str(genres)
                dev = d.get("developer") or d.get("publisher") or "unknown"
                rows.append(
                    {
                        "game_id": gid,
                        "title": str(title),
                        "genres": gstr or "Game",
                        "developer": str(dev),
                    }
                )
        df = pd.DataFrame(rows).drop_duplicates("game_id")
        df["item_id"] = np.arange(1, len(df) + 1, dtype=np.int64)
        self._game_to_id = dict(zip(df["game_id"], df["item_id"]))
        return df.set_index("item_id")

    def _load_reviews(self, min_hours: float, max_reviews: int | None) -> pd.DataFrame:
        path = self.root / "steam_reviews.json.gz"
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_reviews is not None and i >= max_reviews:
                    break
                d = parse_json_line(line)
                if not d:
                    continue
                gid = str(d.get("product_id") or d.get("app_id") or d.get("id") or "")
                user = d.get("username") or d.get("user_id")
                if not gid or not user or gid not in self._game_to_id:
                    continue
                hours = float(d.get("hours", d.get("play_hours", 0)) or 0)
                if hours < min_hours:
                    continue
                raw_ts = d.get("timestamp", d.get("date", 0)) or 0
                try:
                    ts = int(raw_ts)
                except (TypeError, ValueError):
                    ts = 0
                if ts <= 0:
                    ts = i
                rows.append(
                    {
                        "user_key": str(user),
                        "item_id": int(self._game_to_id[gid]),
                        "hours": hours,
                        "timestamp": ts if ts > 0 else i,
                    }
                )
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No Steam reviews loaded; check data files.")
        user_keys = sorted(df["user_key"].unique())
        self._user_key_to_id = {k: i + 1 for i, k in enumerate(user_keys)}
        df["user_id"] = df["user_key"].map(self._user_key_to_id).astype(np.int64)
        return df.sort_values(["user_id", "timestamp"])

    def _build_user_items(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for uid, grp in self.ratings.groupby("user_id"):
            out[int(uid)] = grp["item_id"].astype(int).tolist()
        return out

    def item_meta(self, item_id: int) -> tuple[str, str]:
        row = self.items.loc[item_id]
        title = str(row["title"])
        genres = str(row["genres"])
        dev = str(row["developer"])
        return title, f"{genres}; developer: {dev}"

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
