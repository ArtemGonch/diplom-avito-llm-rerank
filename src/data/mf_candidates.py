"""Matrix-factorization candidate generator (rank2rec / KAR style top-K rerank)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _MFBPRModule(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users + 1, dim)
        self.item_emb = nn.Embedding(num_items + 1, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(users) * self.item_emb(items)).sum(dim=-1)


class MFBPR:
    """BPR-MF retriever for top-100 rerank candidates (paper / KAR protocol)."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        dim: int = 64,
        seed: int = 42,
        device: str = "cpu",
    ):
        torch.manual_seed(seed)
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.device = torch.device(device)
        self.model = _MFBPRModule(num_users, num_items, dim).to(self.device)
        self._all_items: np.ndarray | None = None

    def fit(
        self,
        user_items: dict[int, list[int]],
        epochs: int = 30,
        lr: float = 0.05,
        batch_size: int = 8192,
        negatives_per_pos: int = 1,
        seed: int = 42,
    ) -> None:
        pairs: list[tuple[int, int]] = []
        user_seen: dict[int, set[int]] = {}
        for uid, seq in user_items.items():
            seen = set(seq)
            user_seen[uid] = seen
            for iid in seq:
                pairs.append((uid, iid))
        if not pairs:
            return

        all_items = np.array(
            sorted({i for seq in user_items.values() for i in seq}),
            dtype=np.int64,
        )
        self._all_items = all_items
        rng = np.random.default_rng(seed)
        opt = torch.optim.Adagrad(self.model.parameters(), lr=lr)

        u_arr = np.array([p[0] for p in pairs], dtype=np.int64)
        i_arr = np.array([p[1] for p in pairs], dtype=np.int64)
        n_pairs = len(pairs)

        for _ in range(epochs):
            idx = rng.permutation(n_pairs)
            for start in range(0, n_pairs, batch_size):
                batch_idx = idx[start : start + batch_size]
                users = torch.tensor(u_arr[batch_idx], device=self.device)
                pos = torch.tensor(i_arr[batch_idx], device=self.device)
                negs = []
                for u in u_arr[batch_idx]:
                    seen = user_seen[int(u)]
                    for _ in range(negatives_per_pos):
                        nid = int(all_items[rng.integers(len(all_items))])
                        tries = 0
                        while nid in seen and tries < 16:
                            nid = int(all_items[rng.integers(len(all_items))])
                            tries += 1
                        negs.append(nid)
                neg = torch.tensor(negs, device=self.device)
                pos_score = self.model(users, pos)
                neg_score = self.model(users, neg)
                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.model.eval()

    @torch.no_grad()
    def topk_candidates(
        self,
        user_id: int,
        target_id: int,
        seen: set[int],
        num_candidates: int,
        rng: np.random.Generator,
    ) -> list[int]:
        scores = (
            self.model.user_emb.weight @ self.model.item_emb.weight.T
        )[user_id].cpu().numpy()
        order = np.argsort(-scores)
        picked: list[int] = []
        for iid in order:
            iid = int(iid)
            if iid == 0 or iid in seen:
                continue
            picked.append(iid)
            if len(picked) >= num_candidates - 1:
                break
        cands = picked[: num_candidates - 1] + [target_id]
        rng.shuffle(cands)
        return cands
