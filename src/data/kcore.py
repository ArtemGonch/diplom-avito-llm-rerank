"""k-core filtering for bipartite user-item interaction graphs (paper Appendix C)."""

from __future__ import annotations


def k_core_user_items(user_items: dict[int, list[int]], k: int = 5) -> dict[int, list[int]]:
    """Iteratively remove users/items with fewer than k interactions."""
    ui = {u: list(seq) for u, seq in user_items.items()}
    while True:
        item_counts: dict[int, int] = {}
        for seq in ui.values():
            for iid in seq:
                item_counts[iid] = item_counts.get(iid, 0) + 1
        valid_items = {iid for iid, c in item_counts.items() if c >= k}
        new_ui: dict[int, list[int]] = {}
        for uid, seq in ui.items():
            filt = [i for i in seq if i in valid_items]
            if len(filt) >= k:
                new_ui[uid] = filt
        if len(new_ui) == len(ui) and all(
            len(new_ui[u]) == len(ui[u]) for u in new_ui
        ):
            return new_ui
        ui = new_ui
        if not ui:
            return {}
