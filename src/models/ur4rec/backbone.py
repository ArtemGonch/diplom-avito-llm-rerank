"""DLCM-style listwise reranker with UR4Rec augmentation (paper §3.3)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DLCMReranker(nn.Module):
    """
    Listwise reranker: GRU over candidate representations.
    Input per item: concat(item_emb, user_emb, optional aug_emb).
    """

    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 768,
        aug_dim: int = 768,
        use_aug: bool = True,
    ):
        super().__init__()
        self.use_aug = use_aug
        self.aug_dim = aug_dim
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        # resized in resize_users(); do not allocate a huge table upfront
        self.user_emb = nn.Embedding(1, hidden_dim, padding_idx=0)
        in_dim = hidden_dim * 2 + (aug_dim if use_aug else 0)
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.score = nn.Linear(hidden_dim, 1)

    def resize_users(self, num_users: int) -> None:
        n = num_users + 1
        if n == self.user_emb.num_embeddings:
            return
        old = self.user_emb
        new = nn.Embedding(n, old.embedding_dim, padding_idx=0, device=old.weight.device)
        with torch.no_grad():
            n_copy = min(n, old.num_embeddings)
            new.weight[:n_copy] = old.weight[:n_copy]
        self.user_emb = new

    def item_hidden(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_emb(item_ids)

    def expand_for_augmentation(self) -> None:
        """Grow GRU input from [item; user] to [item; user; aug] for joint training (§3.3)."""
        if self.use_aug:
            return
        hidden = self.gru.hidden_size
        in_old = hidden * 2
        in_new = in_old + self.aug_dim
        device = self.gru.weight_ih_l0.device
        dtype = self.gru.weight_ih_l0.dtype
        new_gru = nn.GRU(in_new, hidden, batch_first=True).to(device=device, dtype=dtype)
        with torch.no_grad():
            new_gru.weight_ih_l0[:, :in_old] = self.gru.weight_ih_l0
            new_gru.weight_ih_l0[:, in_old:] = 0
            new_gru.weight_hh_l0.copy_(self.gru.weight_hh_l0)
            # bias size is 3*hidden, not input dim — copy fully
            new_gru.bias_ih_l0.copy_(self.gru.bias_ih_l0)
            new_gru.bias_hh_l0.copy_(self.gru.bias_hh_l0)
        self.gru = new_gru
        self.use_aug = True

    def forward(
        self,
        item_ids: torch.Tensor,
        user_ids: torch.Tensor,
        aug: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        item_ids: [B, L]
        user_ids: [B]
        aug: [B, L, D] optional
        returns scores [B, L]
        """
        ie = self.item_emb(item_ids)
        ue = self.user_emb(user_ids).unsqueeze(1).expand_as(ie)
        if self.use_aug:
            if aug is None:
                aug = torch.zeros(
                    ie.size(0), ie.size(1), self.aug_dim, device=ie.device, dtype=ie.dtype
                )
            feats = torch.cat([ie, ue, aug], dim=-1)
        else:
            feats = torch.cat([ie, ue], dim=-1)
        out, _ = self.gru(feats)
        return self.score(out).squeeze(-1)
