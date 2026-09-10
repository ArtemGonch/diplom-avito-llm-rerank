"""UR4Rec losses: InfoNCE (§3.2) and DLCM listwise ranking loss (§3.3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def proxy_max_cosine_sim(
    proxy_out: torch.Tensor,
    item_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Paper §3.2: max cosine sim over K proxy vectors.
    proxy_out: [B, K, D]
    item_emb: [B, L, D] or [B, D] -> returns [B, L] or [B]
    """
    if item_emb.dim() == 2:
        item_emb = item_emb.unsqueeze(1)
    a = F.normalize(proxy_out, dim=-1)
    b = F.normalize(item_emb, dim=-1)
    sim = torch.einsum("bkd,bld->bkl", a, b)
    out = sim.max(dim=1).values  # [B, L]
    if out.size(-1) == 1:
        return out.squeeze(-1)
    return out


def info_nce_loss(
    proxy_pref: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    proxy_pref: [B, K, D]
    positive: [B, D]
    negatives: [B, M, D]
    """
    pos_logits = proxy_max_cosine_sim(proxy_pref, positive) / temperature
    neg_logits = proxy_max_cosine_sim(proxy_pref, negatives) / temperature
    logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
    labels = torch.zeros(proxy_pref.size(0), dtype=torch.long, device=proxy_pref.device)
    return F.cross_entropy(logits, labels)


def listwise_ce_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """DLCM listwise softmax loss (one relevant item per candidate list)."""
    labels = labels.float()
    log_probs = F.log_softmax(scores, dim=-1)
    denom = labels.sum(dim=-1).clamp(min=1.0)
    return -((labels * log_probs).sum(dim=-1) / denom).mean()
