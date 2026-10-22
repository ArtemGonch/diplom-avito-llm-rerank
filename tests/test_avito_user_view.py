"""User-view contracts: independent evidence, deterministic cohorts and safe HTML."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from avito_user_view import build_payload, render_html, verify_archive, reconcile_btc


def card(i, **changes):
    result = dict(
        item=str(i),
        title=f"Car {i}",
        brand="Brand",
        model="Known",
        price=1_000_000,
        year="2020",
        mileage_km=70_000,
        body_type="Седан",
        transmission="Автомат",
        engine_type="Бензин",
        drive="Передний",
        events=[dict(t="2026-09-10T12:00:00", kind="view")],
        displayEvents=[dict(t="2026-09-10T12:00:00", kind="view")],
        contacts=[],
    )
    result.update(changes)
    return result


def dataset(cards, others=()):
    return dict(
        users=[
            dict(
                id="9007199254740993123",
                cards=cards,
                searches=[],
                reportEvents=[],
                unmappedContacts=[],
            ),
            *others,
        ],
        period=["2026-09-06", "2026-09-19"],
    )


class UserViewTests(unittest.TestCase):
    def test_deduplicates_items_and_does_not_inflate_confidence(self):
        c = card(1)
        c["events"] *= 30
        u = build_payload(dataset([c, c]))["users"][0]
        self.assertEqual(u["id"], "9007199254740993123")
        self.assertEqual(u["counts"]["items"], 1)
        self.assertFalse(u["preferences"])
        self.assertEqual(u["status"], "sparse")

    def test_missing_is_not_negative_and_has_evidence_denominators(self):
        cards = [card(i, transmission="Автомат" if i < 5 else None) for i in range(10)]
        u = build_payload(dataset(cards))["users"][0]
        self.assertFalse(any(p["field"] == "transmission" for p in u["preferences"]))
        p = next(p for p in u["preferences"] if p["field"] == "body_type")
        self.assertEqual((p["count"], p["known"], p["total"]), (10, 10, 10))
        self.assertEqual(len(p["evidence"]), 10)
        self.assertIsNone(u["country"])

    def test_clusters_keep_distinct_budgets_and_isolate_rare_item(self):
        cs = (
            [card(i) for i in range(6)]
            + [card(i, price=4_000_000) for i in range(6, 10)]
            + [card(10, price=30_000_000, body_type="Купе")]
        )
        u = build_payload(dataset(cs))["users"][0]
        self.assertEqual([c["count"] for c in u["clusters"]], [6, 4])
        self.assertEqual(u["clusters"][0]["price"]["median"], 1_000_000)
        self.assertEqual(u["outliers"][0]["item"], "10")
        self.assertEqual(
            build_payload(dataset(list(reversed(cs))))["users"][0]["clusters"],
            u["clusters"],
        )

    def test_recommendations_exclude_seen_models_and_explain_evidence(self):
        pool = dict(
            id="other",
            cards=[card(30, model="New"), card(31, model="Known")],
            searches=[],
        )
        u = build_payload(dataset([card(i) for i in range(6)], [pool]))["users"][0]
        self.assertEqual([r["model"] for r in u["recommendations"]], ["New"])
        self.assertTrue(u["recommendations"][0]["reasons"])
        self.assertEqual(u["communications"]["status"], "not_available")
        self.assertEqual(u["communications"]["topics"], [])

    def test_no_history_cannot_generate_profile_or_recommendations(self):
        u = build_payload(dataset([]))["users"][0]
        self.assertEqual(u["status"], "empty")
        self.assertFalse(u["preferences"] or u["recommendations"] or u["clusters"])

    def test_script_injection_and_private_fields_are_not_exported(self):
        c = card(
            1,
            title="</script><img src=x onerror=alert(1)>",
            vin="SECRET",
            description="PHONE SECRET",
        )
        p = build_payload(dataset([c]))
        html = render_html(p)
        self.assertNotIn("</script><img", html)
        self.assertNotIn("PHONE SECRET", html)
        self.assertNotIn("vin", p["users"][0]["items"][0])
        self.assertIn("connect-src", html)
        self.assertIn("textContent", html)

    def test_decimal_price_provenance_and_client_only_exclusion(self):
        cs = [
            card(
                i,
                price="1200000.000000000000000",
                priceSource="Снимок до 20 сентября",
                snapshot="2026-09-19",
            )
            for i in range(6)
        ]
        cs.append(
            card(
                9,
                displayEvents=[],
                contacts=[],
                events=[{"t": "2026-09-10", "kind": "view_client"}],
            )
        )
        u = build_payload(dataset(cs))["users"][0]
        self.assertEqual(u["counts"]["analyzed_items"], 6)
        self.assertEqual(u["clusters"][0]["price"]["median"], 1200000)
        self.assertEqual(u["items"][0]["priceSource"], "Снимок до 20 сентября")
        self.assertFalse(u["items"][-1]["analyzed"])

    def test_extreme_numbers_and_low_price_cannot_form_budget(self):
        from avito_user_view.core import number

        self.assertIsNone(number("1e1000"))
        u = build_payload(dataset([card(i, price=1) for i in range(6)]))["users"][0]
        self.assertFalse(u["clusters"])
        self.assertTrue(all(c["price_flag"] for c in u["items"]))

    def test_client_only_seen_model_is_excluded_from_new_models(self):
        own = [card(i) for i in range(6)] + [
            card(50, model="SeenClient", displayEvents=[])
        ]
        pool = dict(id="other", cards=[card(40, model="SeenClient")], searches=[])
        u = build_payload(dataset(own, [pool]))["users"][0]
        self.assertFalse(u["recommendations"])

    def test_source_event_multiplicity_and_contact_layers_are_preserved(self):
        c = card(1)
        c["displayEvents"] *= 2
        c["contacts"] = [
            dict(
                t="2026-09-10", channel="phone", is_common_funnel=False, is_target=True
            )
        ]
        u = build_payload(dataset([c]))["users"][0]
        self.assertEqual(u["counts"]["display_events"], 2)
        self.assertEqual(u["counts"]["contacts"], 1)
        self.assertEqual(u["counts"]["funnel_contacts"], 0)
        self.assertEqual(u["counts"]["target_funnel_contacts"], 0)

    def test_btc_reconciliation_preserves_multiplicity_and_does_not_mutate_input(self):
        c = card(1)
        c["events"] = [dict(t="2026-09-10 12:00:00.000000", kind="favorite", eid=42)]
        d = dataset([c])
        base = dict(
            user_id=d["users"][0]["id"],
            item_id="1",
            action_ts="2026-09-10 12:00:00.000",
            action_eid=42,
            action_type="favorite",
        )
        rows = [
            dict(base, event_hash=1),
            dict(base, event_hash=2),
            dict(base, action_eid=308, action_type="social_share", event_hash=3),
        ]
        enriched, report = reconcile_btc(d, rows)
        self.assertNotIn("supplementaryEvents", c)
        self.assertEqual(report["added_events"], 2)
        self.assertEqual(
            len(enriched["users"][0]["cards"][0]["supplementaryEvents"]), 2
        )
        self.assertEqual(
            build_payload(enriched)["users"][0]["counts"]["btc_supplement"], 2
        )

    def test_behavior_insights_use_distinct_days_and_contact_items(self):
        cs = [
            card(
                i,
                displayEvents=[
                    dict(t="2026-09-10 12:00:00", kind="view"),
                    dict(t="2026-09-12 12:00:00", kind="view"),
                ],
            )
            for i in range(6)
        ]
        for c in cs[:3]:
            c["contacts"] = [
                dict(t="2026-09-13", channel="phone", is_common_funnel=True)
            ]
        u = build_payload(dataset(cs))["users"][0]
        returns = next(x for x in u["behavior"] if x["kind"] == "return_model")
        self.assertEqual(returns["days"], 2)
        contacts = next(x for x in u["behavior"] if x["kind"] == "contact_focus")
        self.assertEqual(contacts["total"], 3)
        self.assertEqual(len(contacts["evidence"]), 3)

    def test_zero_price_is_flagged_not_discarded_and_recommendation_has_provenance(
        self,
    ):
        pool = dict(
            id="other",
            cards=[
                card(30, model="New", priceSource="snapshot", snapshot="2026-09-19")
            ],
            searches=[],
        )
        u = build_payload(
            dataset([card(i) for i in range(6)] + [card(10, price=0)], [pool])
        )["users"][0]
        zero = next(c for c in u["items"] if c["item"] == "10")
        self.assertEqual(zero["price"], 0)
        self.assertTrue(zero["price_flag"])
        self.assertEqual(u["recommendations"][0]["example"]["priceSource"], "snapshot")

    def test_alternatives_respect_main_group_age_range(self):
        pool = dict(
            id="other",
            cards=[
                card(30, model="TooOld", year="1993"),
                card(31, model="Recent", year="2019"),
            ],
            searches=[],
        )
        u = build_payload(dataset([card(i) for i in range(6)], [pool]))["users"][0]
        self.assertEqual([r["model"] for r in u["recommendations"]], ["Recent"])

    def test_return_model_does_not_count_favorite_only_as_viewed(self):
        first = card(
            1,
            displayEvents=[
                dict(t="2026-09-10", kind="view"),
                dict(t="2026-09-12", kind="view"),
            ],
        )
        others = [
            card(i, displayEvents=[dict(t="2026-09-10", kind="favorite")])
            for i in (2, 3)
        ]
        u = build_payload(dataset([first, *others]))["users"][0]
        self.assertFalse(any(x["kind"] == "return_model" for x in u["behavior"]))

    def test_reports_and_imv_are_display_only_and_keep_distinct_meanings(self):
        c = card(
            1,
            imv=dict(
                point=900000, low=800000, high=1000000, day="2026-09-10", label="daily"
            ),
        )
        d = dataset([c])
        d["users"][0]["reportEvents"] = [
            dict(kind="report_purchase", t="2026-09-10", item=None, id="PRIVATE"),
            dict(kind="report_use", t="2026-09-11", item="1", id="PRIVATE"),
        ]
        u = build_payload(d)["users"][0]
        self.assertEqual(u["counts"]["report_purchases"], 1)
        self.assertEqual(u["counts"]["report_uses"], 1)
        self.assertNotIn("id", u["report_events"][0])
        self.assertEqual(u["items"][0]["imv"]["day"], "2026-09-10")
        self.assertFalse(u["preferences"])

    def test_archive_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            (b / "x").write_text("bad")
            (b / "manifest.json").write_text(
                json.dumps({"files": [{"path": "x", "bytes": 3, "sha256": "wrong"}]})
            )
            with self.assertRaises(ValueError):
                verify_archive(b)

    def test_manifest_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            (b / "manifest.json").write_text(
                json.dumps({"files": [{"path": "../x", "bytes": 3, "sha256": "wrong"}]})
            )
            with self.assertRaises(ValueError):
                verify_archive(b)


if __name__ == "__main__":
    unittest.main()
