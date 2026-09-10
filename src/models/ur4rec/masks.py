"""Attention masks from UR4Rec Figure 3 (a,b,c)."""

from __future__ import annotations

import torch


def bidirectional_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Full bidirectional attention among seq_len tokens."""
    return torch.zeros(seq_len, seq_len, device=device)


def proxy_item_mask(
    num_proxies: int,
    num_items: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Figure 3(c): proxies attend to all; items attend to proxies + global context
    but items cannot attend to each other.
    """
    n = num_proxies + num_items
    mask = torch.zeros(n, n, device=device)
    # Block off-diagonal item -> item attention while retaining each item's
    # self-attention.  Figure 3(c): items are isolated from one another and
    # proxies can still attend to every item.
    if num_items > 1:
        start = num_proxies
        stop = num_proxies + num_items
        item_block = torch.full(
            (num_items, num_items),
            float("-inf"),
            device=device,
        )
        item_block.fill_diagonal_(0.0)
        mask[start:stop, start:stop] = item_block
    return mask


def build_extended_mask(
    base_mask: torch.Tensor | None,
    batch_size: int,
    num_heads: int,
) -> torch.Tensor | None:
    """Return (L, L) additive mask; MHA broadcasts over batch and heads."""
    del batch_size, num_heads
    return base_mask
