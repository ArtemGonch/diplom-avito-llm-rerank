import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_bx_history import _diagnostics, check_history_time, history_preferences, paired_order


def event(item, brand, kind="btc"):
    return {"item_id": item, "attr_brand": brand, "interaction_type": kind}


class BXHistoryTests(unittest.TestCase):
    def test_window_is_closed_on_left_open_on_right(self):
        search = datetime(2026, 10, 6)
        check_history_time(search, search - timedelta(days=7))
        check_history_time(search, search - timedelta(seconds=1))
        for event_at in (search, search + timedelta(seconds=1),
                         search - timedelta(days=7, seconds=1)):
            with self.assertRaises(ValueError):
                check_history_time(search, event_at)

    def test_btc_only_unique_items_and_conflicting_brands(self):
        seen, brand = history_preferences(
            [event(1, "A")] * 4 + [
                event(2, "A"),
                event(2, "B"),  # contradictory item gives no brand vote
                event(3, "A", "dtb"),
                event(4, "A", "contact"),
                event(5, "B"),
                event(6, "B"),
            ]
        )
        self.assertEqual(seen, {1, 2, 5, 6})
        self.assertEqual(brand, "b")

    def test_brand_requires_two_distinct_items_and_unique_mode(self):
        self.assertIsNone(history_preferences([event(1, "a")] * 20)[1])
        self.assertIsNone(history_preferences(
            [event(1, "a"), event(2, "A"), event(3, "b"), event(4, "b")]
        )[1])

    def test_adjacent_pairs_keep_full_pool_and_abstain_on_unknown(self):
        scores = [0.3, 0.9, 0.1, 0.8, 0.2]
        priority = [1, 0, 1, 1, None]
        # BX order 1,3 | 0,4 | 2. Second pair cannot compare a missing brand.
        self.assertEqual(paired_order(scores, priority), ([3, 1, 0, 4, 2], 1))
        self.assertEqual(paired_order([0, 0, 0], [0, 0, 1]), ([0, 1, 2], 0))

    def test_metric_mask_does_not_redraw_adjacent_pairs(self):
        candidates = [
            {
                "item_id": item, "attr_brand": "car", "baseline_score": 10 - item,
                "target_contact_s_d7": label, "target": label,
            }
            for item, label in enumerate((0, None, 1, 0))
        ]
        _, methods = _diagnostics({"search": candidates}, {"search": [event(2, "car")]})
        method = methods["seen_btc_item"]
        self.assertEqual(method["changed_groups"], 0)
        self.assertEqual(method["metrics"]["target_contact_s_d7"]["delta"], 0)
