"""Contracts for the complete 16 October user-summary assignment."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from avito_user_view import build_payload
from avito_user_view.analysis import analyze_searches
from tests.test_avito_user_view import card, dataset


def search(t="2026-09-10 12:00:00", params=None, filters=None, **other):
    return dict(
        t=t,
        payloadStatus="matched",
        params=params or {},
        filters=filters or [],
        filtersReadable=True,
        **other,
    )


class CompleteSummaryTests(unittest.TestCase):
    def test_common_fuel_is_not_called_distinctive_and_loo_reference_excludes_self(
        self,
    ):
        others = [
            dict(
                id=str(k),
                cards=[card(k * 100 + i, drive="Передний") for i in range(12)],
                searches=[],
            )
            for k in range(1, 4)
        ]
        u = build_payload(
            dataset([card(i, drive="Полный") for i in range(10)], others)
        )["users"][0]
        imp = {x["field"]: x for x in u["analysis"]["importance"]}
        self.assertEqual(imp["engine_type"]["status"], "common_in_sample")
        self.assertEqual(imp["drive"]["status"], "distinctive")
        self.assertEqual(imp["drive"]["reference_known"], 36)
        self.assertEqual(imp["drive"]["reference_share"], 0)

    def test_search_pagination_and_list_order_do_not_inflate_states(self):
        raw = [
            search(params={"110008": ["Автомат", "Вариатор"]}, page=0),
            search(params={"110008": ["Вариатор", "Автомат"]}, page=2),
        ]
        s = analyze_searches(raw, {"body_type": {"Седан"}})
        self.assertEqual(s["unique_states"], 1)
        self.assertEqual(s["fields"]["transmission"]["states"], 1)
        self.assertEqual(s["fields"]["transmission"]["events"], 2)
        self.assertEqual(s["fields"]["transmission"]["days"], 1)

    def test_search_ranges_zero_sentinel_and_invalid_year(self):
        s = analyze_searches(
            [
                search(
                    params={
                        "164669": {"from": "2020", "to": 0},
                        "2687": {"to": 150000},
                    },
                    filters=["Цена от 900 000 ₽", "Цена до 1 500 000 ₽"],
                )
            ],
            {},
        )
        r = s["latest"]["ranges"]
        self.assertEqual(r["year"], {"min": 2020, "max": None})
        self.assertEqual(r["price"], {"min": 900000, "max": 1500000})
        bad = analyze_searches(
            [search(params={"164669": {"from": "201", "to": 2020}})], {}
        )
        self.assertTrue(bad["latest"]["warnings"])
        self.assertNotIn("year", bad["latest"]["ranges"])

    def test_scenarios_are_not_cross_joined_and_ambiguous_search_has_no_constraints(
        self,
    ):
        raw = [
            search(params={"147411": "Седан", "110008": "Автомат"}),
            search(
                t="2026-09-11",
                params={"147411": "Внедорожник 5-дверный", "110008": "Механика"},
            ),
            dict(
                payloadStatus="ambiguous",
                t="2026-09-12",
                params={"110008": "Робот"},
                filters=[],
            ),
        ]
        s = analyze_searches(raw, {"body_type": {"Седан", "Внедорожник"}})
        self.assertEqual(len(s["states"]), 2)
        self.assertEqual(s["latest"]["categorical"]["body_type"], ["Внедорожник"])
        self.assertEqual(s["latest"]["categorical"]["transmission"], ["Механика"])
        self.assertEqual(s["ambiguous_events"], 1)
        self.assertTrue(s["newer_ambiguous"])

    def test_later_unrestricted_search_clears_previous_hard_filters(self):
        s = analyze_searches(
            [
                search(params={"110008": "Механика"}),
                search(t="2026-09-12", filters=["Тип автомобиля: С пробегом"]),
            ],
            {},
        )
        self.assertFalse(s["latest"]["categorical"])
        self.assertEqual(s["fields"]["transmission"]["states"], 1)

    def test_equal_filter_counts_have_stable_lexical_ties(self):
        labels = [f"Поле {i:02}" for i in range(20)]
        d = dataset([])
        d["users"][0]["searches"] = [search(filters=labels)]
        u = build_payload(d)["users"][0]
        self.assertEqual([r["text"] for r in u["repeated_filters"]], labels[:12])

    def test_brand_region_is_explicit_not_listing_country(self):
        d = dataset([card(i) for i in range(6)])
        d["users"][0]["searches"] = [search(params={"152642": ["Япония"]})]
        u = build_payload(d)["users"][0]
        self.assertEqual(
            u["analysis"]["search"]["fields"]["brand_region"]["values"], ["Япония"]
        )
        self.assertIsNone(u["country"])

    def test_shortest_eighty_percent_range_excludes_a_single_price_extreme(self):
        u = build_payload(
            dataset(
                [card(i, price=1_000_000) for i in range(5)]
                + [card(5, price=50_000_000)]
            )
        )["users"][0]
        r = next(x for x in u["analysis"]["numeric"] if x["field"] == "price")
        self.assertEqual(r["core_max"], 1_000_000)
        self.assertTrue(r["narrow"])
        self.assertEqual(r["count"], 6)
        self.assertEqual(r["core_count"], 5)

    def test_model_summary_does_not_merge_equal_names_from_different_brands(self):
        u = build_payload(
            dataset([card(i, model="3", brand="A" if i < 3 else "B") for i in range(6)])
        )["users"][0]
        self.assertEqual(len(u["analysis"]["models"]), 2)
        self.assertEqual([x["count"] for x in u["analysis"]["models"]], [3, 3])
        self.assertTrue(u["analysis"]["model_common"])

    def test_tied_clusters_have_no_unique_primary(self):
        u = build_payload(
            dataset(
                [
                    card(i, body_type="Седан" if i < 4 else "Внедорожник")
                    for i in range(8)
                ]
            )
        )["users"][0]
        self.assertEqual(u["analysis"]["cluster_status"], "tied")
        self.assertIsNone(u["analysis"]["primary_cluster"])

    def test_alternative_obeys_latest_search_without_mixing_old_filter(self):
        d = dataset(
            [card(i) for i in range(6)],
            [
                dict(
                    id="pool",
                    cards=[
                        card(30, model="Diesel", engine_type="Дизель"),
                        card(31, model="Petrol"),
                    ],
                    searches=[],
                )
            ],
        )
        d["users"][0]["searches"] = [
            search(params={"110006": "Бензин"}),
            search(t="2026-09-12", params={"110006": "Дизель"}),
        ]
        u = build_payload(d)["users"][0]
        self.assertEqual([r["model"] for r in u["recommendations"]], ["Diesel"])
        self.assertEqual(u["analysis"]["recommendation_status"], "candidates")

    def test_region_constraint_unknown_is_not_claimed_verified(self):
        d = dataset(
            [card(i) for i in range(6)],
            [dict(id="pool", cards=[card(30, model="New")], searches=[])],
        )
        d["users"][0]["searches"] = [search(params={"152642": "Япония"})]
        u = build_payload(d)["users"][0]
        self.assertEqual(u["analysis"]["recommendation_status"], "conditional")
        self.assertIn("brand_region", u["recommendations"][0]["unverified"])

    def test_seen_listing_is_excluded_even_if_other_snapshot_renames_model(self):
        d = dataset(
            [card(i) for i in range(6)],
            [dict(id="pool", cards=[card(0, model="Renamed")], searches=[])],
        )
        u = build_payload(d)["users"][0]
        self.assertFalse(u["recommendations"])

    def test_no_match_has_explanation_and_no_fabricated_model(self):
        d = dataset(
            [card(i) for i in range(6)],
            [dict(id="pool", cards=[card(30, model="New")], searches=[])],
        )
        d["users"][0]["searches"] = [search(params={"110008": "Механика"})]
        u = build_payload(d)["users"][0]
        self.assertFalse(u["recommendations"])
        self.assertEqual(
            u["analysis"]["recommendation_status"], "no_matching_candidates"
        )

    def test_every_readable_fact_has_supported_evidence_and_no_communication_topics(
        self,
    ):
        d = dataset([card(i) for i in range(6)])
        u = build_payload(d)["users"][0]
        self.assertTrue(u["facts"])
        ids = {c["item"] for c in u["items"]}
        for f in u["facts"]:
            self.assertTrue(set(f.get("item_ids", [])) <= ids)
            self.assertIn("rule_version", f)
        self.assertEqual(u["communications"]["topics"], [])
        self.assertTrue(u["summary"]["sections"])


if __name__ == "__main__":
    unittest.main()
