"""Avito SERP parquet -> reranking samples (items_with_attrs.parquet)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .ml1m import RerankSample


@dataclass
class AvitoSERP:
    """
    One rerank sample = one search result page (serp_x).
    Labels from contacts_daily or clicks_daily within the SERP.
    user_id in samples = internal serp index (not buyer/seller id).
    """

    items: pd.DataFrame
    users: pd.DataFrame | None
    item2idx: dict[int, int]
    idx2item: dict[int, int]
    serp2idx: dict[str, int]
    idx2serp: dict[int, str]
    num_items: int
    num_users: int
    label_field: str
    min_serp_size: int

    @classmethod
    def from_parquet(
        cls,
        items_path: Path,
        users_path: Path | None = None,
        label_field: str = "contacts",
        min_serp_size: int = 10,
    ) -> AvitoSERP:
        items = pd.read_parquet(items_path)
        users = pd.read_parquet(users_path) if users_path and Path(users_path).exists() else None
        item2idx: dict[int, int] = {0: 0}
        for i, iid in enumerate(sorted(items["item_id"].astype(np.int64).unique()), start=1):
            item2idx[int(iid)] = i
        idx2item = {v: k for k, v in item2idx.items() if k != 0}
        serps = sorted(items["serp_x"].astype(str).unique())
        serp2idx = {s: i + 1 for i, s in enumerate(serps)}
        idx2serp = {i: s for s, i in serp2idx.items()}
        return cls(
            items=items,
            users=users,
            item2idx=item2idx,
            idx2item=idx2item,
            serp2idx=serp2idx,
            idx2serp=idx2serp,
            num_items=max(item2idx.values()),
            num_users=len(serp2idx),
            label_field=label_field,
            min_serp_size=min_serp_size,
        )

    def all_item_indices(self) -> np.ndarray:
        return np.array([self.item2idx[i] for i in self.item2idx if i != 0], dtype=np.int64)

    def item_meta(self, item_idx: int) -> tuple[str, str]:
        raw = self.idx2item[int(item_idx)]
        row = self.items.loc[self.items["item_id"] == raw].iloc[0]
        title = str(row.get("title", "") or "listing")
        attrs = " ".join(
            str(row.get(c, "") or "")
            for c in ("brand", "model_name", "fuel_text", "gearbox_text", "body_type")
        ).strip()
        return title, attrs or "auto"

    def serp_query_text(self, serp_x: str) -> str:
        row = self.items.loc[self.items["serp_x"] == serp_x].iloc[0]
        cat = row.get("query_infm_logical_category", "Transport")
        loc = row.get("query_loc", "")
        return f"Avito search: {cat}, region or location id {loc}."

    def train_val_test_split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[int], list[int], list[int]]:
        serp_ids = sorted(self.serp2idx.values())
        rng = np.random.default_rng(seed)
        rng.shuffle(serp_ids)
        n = len(serp_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            serp_ids[:n_train],
            serp_ids[n_train : n_train + n_val],
            serp_ids[n_train + n_val :],
        )

    def _labels_for_group(self, grp: pd.DataFrame) -> np.ndarray:
        if self.label_field == "clicks":
            v = grp["clicks_daily"].astype(np.float32).values
        else:
            v = grp["contacts_daily"].astype(np.float32).values
        mx = float(v.max()) if len(v) else 0.0
        if mx <= 0:
            return np.zeros_like(v)
        return v / mx

    def build_rerank_samples(
        self,
        serp_indices: list[int],
        history_len: int,
        num_candidates: int,
        seed: int = 42,
    ) -> list[RerankSample]:
        rng = np.random.default_rng(seed)
        samples: list[RerankSample] = []

        for sid in serp_indices:
            serp_x = self.idx2serp[sid]
            grp = self.items[self.items["serp_x"] == serp_x]
            if len(grp) < self.min_serp_size:
                continue

            grp = grp.copy()
            grp["_idx"] = grp["item_id"].map(lambda x: self.item2idx[int(x)])
            labels_full = self._labels_for_group(grp)
            grp["_lab"] = labels_full
            target_row = grp.loc[grp["_lab"].idxmax()]
            target_idx = int(target_row["_idx"])

            if len(grp) <= num_candidates:
                sub = grp
            else:
                pos = grp[grp["_lab"] > 0]
                n_pos = min(len(pos), max(1, num_candidates // 2))
                pos_pick = pos.nlargest(n_pos, "_lab")
                rest = grp[~grp.index.isin(pos_pick.index)]
                n_rest = num_candidates - len(pos_pick)
                rest_pick = rest.sample(n=min(n_rest, len(rest)), random_state=int(rng.integers(1e9)))
                sub = pd.concat([pos_pick, rest_pick]).drop_duplicates("_idx")
                if len(sub) < num_candidates:
                    extra = grp[~grp.index.isin(sub.index)].sample(
                        n=min(num_candidates - len(sub), len(grp) - len(sub)),
                        random_state=int(rng.integers(1e9)),
                    )
                    sub = pd.concat([sub, extra])

            cands = sub["_idx"].astype(int).tolist()
            labs = (sub["_lab"] > 0).astype(float).tolist()
            order = rng.permutation(len(cands))
            cands = [cands[i] for i in order]
            labs = [labs[i] for i in order]

            samples.append(
                RerankSample(
                    user_id=sid,
                    history_item_ids=[],
                    candidate_item_ids=cands,
                    labels=labs,
                    target_item_id=target_idx,
                )
            )
        return samples
