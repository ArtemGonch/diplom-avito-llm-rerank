"""MovieLens-1M loader and reranking sample builder (paper §4.1, Appendix C)."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from .kcore import k_core_user_items
from .mf_candidates import MFBPR


ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


@dataclass
class RerankSample:
    user_id: int
    history_item_ids: list[int]
    candidate_item_ids: list[int]
    labels: list[int]  # 1 for relevant, 0 else
    target_item_id: int


def download_movielens_1m(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ready = data_dir / "ml-1m" / "ratings.dat"
    if ready.exists():
        return data_dir / "ml-1m"
    zpath = data_dir / "ml-1m.zip"
    print(f"Downloading MovieLens-1M -> {zpath}")
    urlretrieve(ML1M_URL, zpath)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(data_dir)
    return data_dir / "ml-1m"


class MovieLens1M:
    def __init__(
        self,
        root: Path,
        min_rating: float = 4.0,
        k_core: int | None = None,
    ):
        self.root = Path(root)
        self.k_core = k_core
        self.movies = self._load_movies()
        self.ratings = self._load_ratings(min_rating)
        self.user_items = self._build_user_items()
        if k_core is not None and k_core > 1:
            before_u, before_i = len(self.user_items), self._count_items()
            self.user_items = k_core_user_items(self.user_items, k_core)
            after_u, after_i = len(self.user_items), self._count_items()
            print(
                f"k-core={k_core}: users {before_u}->{after_u}, "
                f"items {before_i}->{after_i}"
            )
        self.num_users = int(max(self.user_items.keys(), default=0))
        self.num_items = int(self.movies["movieId"].max())
        self._mf: MFBPR | None = None

    def _count_items(self) -> int:
        return len({i for seq in self.user_items.values() for i in seq})

    def _load_movies(self) -> pd.DataFrame:
        path = self.root / "movies.dat"
        df = pd.read_csv(
            path,
            sep="::",
            engine="python",
            header=None,
            names=["movieId", "title", "genres"],
            encoding="latin-1",
        )
        df["genres"] = df["genres"].str.replace("|", ", ")
        return df

    def _load_ratings(self, min_rating: float) -> pd.DataFrame:
        path = self.root / "ratings.dat"
        df = pd.read_csv(
            path,
            sep="::",
            engine="python",
            header=None,
            names=["userId", "movieId", "rating", "timestamp"],
        )
        return df[df["rating"] >= min_rating].sort_values("timestamp")

    def _build_user_items(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for uid, grp in self.ratings.groupby("userId"):
            out[int(uid)] = grp["movieId"].astype(int).tolist()
        return out

    def fit_mf_candidates(
        self,
        train_user_ids: list[int],
        dim: int = 64,
        epochs: int = 30,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        """Train BPR-MF on train users for top-100 rerank candidates (KAR/rank2rec)."""
        train_ui = {u: self.user_items[u] for u in train_user_ids if u in self.user_items}
        self._mf = MFBPR(self.num_users, self.num_items, dim=dim, seed=seed, device=device)
        self._mf.fit(train_ui, epochs=epochs, seed=seed)
        print(f"MF candidate generator trained on {len(train_ui)} users")

    def movie_meta(self, item_id: int) -> tuple[str, str]:
        row = self.movies[self.movies["movieId"] == item_id].iloc[0]
        return str(row["title"]), str(row["genres"])

    def train_val_test_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[int], list[int], list[int]]:
        users = sorted(self.user_items.keys())
        rng = np.random.default_rng(seed)
        rng.shuffle(users)
        n = len(users)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_u = users[:n_train]
        val_u = users[n_train : n_train + n_val]
        test_u = users[n_train + n_val :]
        return train_u, val_u, test_u

    def build_rerank_samples(
        self,
        user_ids: list[int],
        history_len: int,
        num_candidates: int,
        seed: int = 42,
        candidate_mode: str = "mf_topk",
    ) -> list[RerankSample]:
        rng = np.random.default_rng(seed)
        all_items = self.movies["movieId"].astype(int).values
        use_mf = candidate_mode == "mf_topk" and self._mf is not None
        samples: list[RerankSample] = []

        for uid in user_ids:
            seq = self.user_items.get(uid, [])
            if len(seq) < history_len + 2:
                continue
            target = seq[-1]
            history = seq[-(history_len + 1) : -1]
            seen = set(seq)
            if use_mf:
                cands = self._mf.topk_candidates(uid, target, seen, num_candidates, rng)
            else:
                negs = [i for i in all_items if i not in seen]
                if len(negs) < num_candidates - 1:
                    raise ValueError(
                        f"User {uid} has only {len(negs)} valid negatives for "
                        f"{num_candidates} candidates"
                    )
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

    def build_temporal_rerank_splits(
        self,
        history_len: int,
        num_candidates: int,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
        candidate_mode: str = "random",
    ) -> tuple[list[RerankSample], list[RerankSample], list[RerankSample]]:
        """One chronological train/val/test target per user.

        Unlike a user-wise split, this keeps every evaluated user represented in
        backbone training while ensuring each target follows its own history.
        The initial corrected protocol deliberately uses random candidates; an
        MF candidate generator needs a separate temporal training protocol.
        """
        if candidate_mode != "random":
            raise ValueError(
                "temporal_per_user currently supports candidate_mode=random only"
            )
        rng = np.random.default_rng(seed)
        all_items = self.movies["movieId"].astype(int).values
        splits: tuple[list[RerankSample], list[RerankSample], list[RerankSample]] = (
            [],
            [],
            [],
        )

        for uid in sorted(self.user_items):
            seq = self.user_items[uid]
            if len(seq) < history_len + 3:
                continue
            train_idx = max(history_len, int(len(seq) * train_ratio) - 1)
            train_idx = min(train_idx, len(seq) - 3)
            val_idx = max(train_idx + 1, int(len(seq) * (train_ratio + val_ratio)) - 1)
            val_idx = min(val_idx, len(seq) - 2)
            test_idx = len(seq) - 1
            known_positives = set(seq)
            negative_pool = [i for i in all_items if i not in known_positives]
            if len(negative_pool) < num_candidates - 1:
                raise ValueError(
                    f"User {uid} has only {len(negative_pool)} valid negatives for "
                    f"{num_candidates} candidates"
                )

            for target_idx, target_split in zip(
                (train_idx, val_idx, test_idx),
                splits,
            ):
                target = seq[target_idx]
                history = seq[target_idx - history_len : target_idx]
                negs = list(negative_pool)
                rng.shuffle(negs)
                cands = [target] + negs[: num_candidates - 1]
                rng.shuffle(cands)
                target_split.append(
                    RerankSample(
                        user_id=uid,
                        history_item_ids=history,
                        candidate_item_ids=cands,
                        labels=[1 if c == target else 0 for c in cands],
                        target_item_id=target,
                    )
                )
        return splits
