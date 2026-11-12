from copy import deepcopy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_user_view.serp_contract import inspect_serps


def request():
    return {
        "request_id": "demo-request",
        "user_id": "demo-user",
        "shown_at": "2026-10-01T09:00:00+03:00",
        "history_cutoff": "2026-10-01T06:00:00Z",
        "query_context": {},
        "logged_size": 2,
        "candidates": [
            {"item_id": "a", "position": 1, "clicked": True, "target_contact": False},
            {"item_id": "b", "position": 2, "clicked": False, "target_contact": None},
        ],
    }


class SerpContractTests(unittest.TestCase):
    def test_unknown_contact_is_not_negative_or_eligible(self):
        row = request()
        r = inspect_serps([row])
        self.assertEqual(r["click"]["complete_requests_with_positive"], 1)
        self.assertEqual(r["target_contact"]["complete_requests_with_positive"], 0)
        self.assertEqual(r["target_contact"]["requests_with_unknown_labels"], 1)
        self.assertNotIn("demo-", str(r))
        self.assertFalse(r["ranking_evaluated"])

    def test_no_positives_distinguished_from_missing_labels(self):
        row = request()
        row["candidates"][1]["target_contact"] = False
        self.assertEqual(
            inspect_serps([row])["target_contact"]["complete_requests_without_positive"], 1
        )

    def test_reject_incomplete_pool_duplicates_and_temporal_leakage(self):
        row = request()
        mutations = (
            lambda x: x["candidates"].pop(),
            lambda x: x["candidates"][1].update(item_id="a"),
            lambda x: x["candidates"][1].update(position=3),
            lambda x: x["candidates"][1].update(clicked=0),
            lambda x: x["candidates"][1].pop("clicked"),
            lambda x: x.update(history_cutoff="2026-10-01T06:00:01Z"),
            lambda x: x.update(shown_at="2026-10-01T09:00:00"),
        )
        for mutate in mutations:
            invalid = deepcopy(row)
            mutate(invalid)
            with self.assertRaises(ValueError):
                inspect_serps([invalid])
        with self.assertRaises(ValueError):
            inspect_serps([row, row])
