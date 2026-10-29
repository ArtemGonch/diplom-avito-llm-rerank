"""Statistical and research contracts for the 22 October follow-up."""

import copy
import json
import math
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from avito_user_view import build_payload
from avito_user_view.research import (
    median_confidence_interval,
    numeric_intervals,
    top_interest,
    compare_overviews,
    blind_pair,
    enrich_research,
    write_research_reports,
    render_blind_html,
    explicit_stability,
)
from avito_user_view.narrative import attach_editor_selections
from tests.test_avito_user_view import card, dataset
from tests.test_avito_user_view_analysis import search


def signal(field="engine_type", **changes):
    row = dict(
        field=field,
        label=field,
        status="distinctive",
        score=0.3,
        values=["Бензин"],
        count=9,
        known=10,
        total=10,
        share=0.9,
        reference_share=0.2,
        reference_known=50,
        evidence=["one"],
        explicit=None,
    )
    row.update(changes)
    return row


def query(field, **changes):
    row = dict(
        field=field,
        label=field,
        values=["Дизель"],
        states=3,
        total_states=4,
        days=2,
        events=9,
        search_refs=["search-1"],
    )
    row.update(changes)
    return row


def interest_user(rows=(), fields=None, latest=None, newer=False):
    return dict(
        analysis=dict(
            importance=list(rows),
            search=dict(fields=fields or {}, latest=latest, newer_ambiguous=newer),
        )
    )


class MedianIntervalTests(unittest.TestCase):
    def test_exact_binomial_ranks_not_percentile_coverage(self):
        for n, ranks, coverage in [
            (6, (1, 6), 0.96875),
            (10, (2, 9), 0.978515625),
            (20, (6, 15), 0.9586105346679688),
            (100, (40, 61), 0.9647997997822952),
        ]:
            row = median_confidence_interval(range(1, n + 1))
            self.assertEqual(row["status"], "available")
            self.assertEqual((row["rank_low"], row["rank_high"]), ranks)
            self.assertEqual((row["low"], row["high"]), ranks)
            self.assertAlmostEqual(row["iid_coverage_bound"], coverage)
            self.assertGreaterEqual(row["iid_coverage_bound"], 0.95)

    def test_cannot_claim_finite_95_percent_interval_for_five_observations(self):
        r = median_confidence_interval([1, 2, 3, 4, 5])
        self.assertEqual(r["status"], "insufficient")
        self.assertIsNone(r["low"])
        self.assertIsNone(r["high"])
        self.assertEqual(r["median"], 3)

    def test_finite_guards_and_permutation_affine_invariance(self):
        base = list(range(1, 21))
        a = median_confidence_interval(base)
        b = median_confidence_interval(
            list(reversed(base)) + [True, None, math.nan, math.inf, "bad"]
        )
        self.assertEqual(a, b)
        c = median_confidence_interval([x * 7 + 100 for x in base])
        self.assertEqual(
            (c["low"], c["high"]), (7 * a["low"] + 100, 7 * a["high"] + 100)
        )

    def test_large_samples_do_not_underflow_to_zero_width(self):
        for n in (2000, 10000):
            r = median_confidence_interval(range(n))
            self.assertLess(r["rank_low"], n // 2 - 10)
            self.assertGreater(r["rank_high"], n // 2 + 10)
            self.assertGreaterEqual(r["iid_coverage_bound"], 0.95)
            self.assertLess(r["iid_coverage_bound"], 0.97)

    def test_ties_are_explicit_and_invalid_confidence_rejected(self):
        r = median_confidence_interval([2020] * 20)
        self.assertEqual((r["low"], r["high"]), (2020, 2020))
        self.assertTrue(r["ties"])
        for q in (0, 1, -1, math.nan, True):
            with self.assertRaises(ValueError):
                median_confidence_interval([1, 2], confidence=q)

    def test_quality_gate_and_coverage_boundary(self):
        for n, known, expected in [
            (9, 9, "insufficient"),
            (10, 10, "available"),
            (20, 12, "available"),
            (20, 11, "low_coverage"),
        ]:
            u = build_payload(
                dataset(
                    [card(i, power_hp=100 + i if i < known else None) for i in range(n)]
                )
            )["users"][0]
            r = next(x for x in numeric_intervals(u) if x["field"] == "power_hp")
            self.assertEqual(r["status"], expected)
            self.assertEqual((r["n"], r["total"]), (known, n))
            if expected != "available":
                self.assertIsNone(r["low"])

    def test_unique_items_missing_domain_and_zero_mileage(self):
        u = build_payload(
            dataset([card(i, power_hp=100, mileage_km=0) for i in range(12)])
        )["users"][0]
        u["items"] *= 3
        a = {r["field"]: r for r in numeric_intervals(u)}
        self.assertEqual(a["price"]["n"], 12)
        self.assertEqual(a["mileage_km"]["low"], 0)
        self.assertEqual(a["mileage_km"]["total"], 12)
        for c in u["items"]:
            c.update(price=0, price_flag="bad", power_hp=math.inf, year=2020.5)
        a = {r["field"]: r for r in numeric_intervals(u)}
        for f in ("price", "power_hp", "year"):
            self.assertEqual(a[f]["n"], 0)

    def test_conflicting_duplicate_values_do_not_silently_pick_a_snapshot(self):
        u = build_payload(dataset([card(i) for i in range(12)]))["users"][0]
        other = dict(u["items"][0], price=2_000_000)
        u["items"].append(other)
        a = next(x for x in numeric_intervals(u) if x["field"] == "price")
        self.assertEqual((a["n"], a["total"], a["conflicting_items"]), (11, 12, 1))
        self.assertNotIn(other["item"], a["evidence"])


class TopInterestTests(unittest.TestCase):
    def test_common_or_insufficient_attributes_cannot_be_forced_into_top(self):
        for status in ("common_in_sample", "insufficient", "variable", "concentrated"):
            r = top_interest(interest_user([signal(status=status, score=99)]))
            self.assertEqual(r["status"], "not_available")
            self.assertEqual(r["leaders"], [])

    def test_explicit_values_do_not_get_replaced_by_observed_values(self):
        u = interest_user(
            [signal(values=["Бензин"])], {"engine_type": query("engine_type")}
        )
        r = top_interest(u)
        self.assertEqual(r["leaders"][0]["values"], ["Дизель"])
        self.assertEqual(r["leaders"][0]["source"], "historical_explicit")
        self.assertEqual(r["leaders"][0]["search_refs"], ["search-1"])

    def test_tied_leaders_are_not_claimed_as_one_significant_attribute(self):
        fields = {f: query(f) for f in ("model", "price")}
        r = top_interest(interest_user(fields=fields))
        self.assertEqual(r["status"], "tied")
        self.assertEqual({x["field"] for x in r["leaders"]}, {"model", "price"})
        self.assertIsNone(r["unique_top_field"])

    def test_numeric_explicit_filters_participate_and_history_is_not_current(self):
        u = interest_user(
            fields={"price": query("price", values=["до 2 000 000 ₽"])},
            latest={"categorical": {}, "ranges": {}},
        )
        r = top_interest(u)
        self.assertEqual(r["unique_top_field"], "price")
        self.assertEqual(r["context_status"], "latest_without_constraints")
        self.assertFalse(r["is_current_constraint"])

    def test_real_numeric_search_ranges_are_kept_as_separate_configurations(self):
        raw = dataset([card(i) for i in range(12)])
        raw["users"][0]["searches"] = [
            search(
                params={"164669": {"from": 2010, "to": 2015}},
                filters=["Цена от 1 000 000 ₽", "Цена до 2 000 000 ₽"],
            ),
            search(
                t="2026-09-11",
                params={"164669": {"from": 2020, "to": 2025}},
                filters=["Цена от 3 000 000 ₽", "Цена до 4 000 000 ₽"],
            ),
        ]
        u = build_payload(raw)["users"][0]
        r = top_interest(u)
        self.assertEqual({x["field"] for x in r["leaders"]}, {"price", "year"})
        price = next(x for x in r["leaders"] if x["field"] == "price")
        self.assertEqual(
            [(x["min"], x["max"]) for x in price["range_options"]],
            [(1_000_000, 2_000_000), (3_000_000, 4_000_000)],
        )
        self.assertEqual(price["range_options"][0]["search_refs"], ["search-1"])
        self.assertTrue(price["values"])

    def test_distinctive_history_is_fallback_and_tie_order_deterministic(self):
        rows = [signal("engine_type"), signal("drive")]
        a = top_interest(interest_user(rows))
        b = top_interest(interest_user(list(reversed(rows))))
        self.assertEqual(a, b)
        self.assertEqual(a["status"], "tied")
        self.assertTrue(all(x["source"] == "history_contrast" for x in a["leaders"]))


class ExplicitStabilityTests(unittest.TestCase):
    def user(self, searches):
        raw = dataset([card(i) for i in range(12)])
        raw["users"][0]["searches"] = searches
        return build_payload(raw)["users"][0]

    def test_same_state_on_three_days_survives_removing_any_one_day(self):
        u = self.user(
            [
                search(t=f"2026-09-{d:02d}", params={"110008": "Автомат"})
                for d in (10, 11, 12)
            ]
        )
        r = explicit_stability(u)
        self.assertEqual(
            (r["status"], r["days"], r["same_top_set_removals"]), ("available", 3, 3)
        )
        self.assertEqual(r["leaders"][0]["unique_leader_removals"], 3)

    def test_removing_a_day_can_turn_unique_leader_into_tie(self):
        u = self.user(
            [
                search(t="2026-09-10", params={"110000": "Mazda"}),
                search(t="2026-09-11", params={"110000": "BMW"}),
                search(t="2026-09-12", filters=["Цена до 2 000 000 ₽"]),
            ]
        )
        r = explicit_stability(u)
        self.assertTrue(r["original_unique"])
        self.assertEqual(r["leaders"][0]["field"], "brand")
        self.assertEqual(r["leaders"][0]["among_leaders_removals"], 3)
        self.assertEqual(r["leaders"][0]["unique_leader_removals"], 1)
        self.assertEqual(r["same_top_set_removals"], 1)
        without_mazda = next(
            x for x in r["removals"] if x["removed_day"] == "2026-09-10"
        )
        brand = next(x for x in without_mazda["leaders"] if x["field"] == "brand")
        self.assertEqual(brand["values"], ["BMW"])

    def test_short_history_and_invalid_dates_do_not_claim_stability(self):
        u = self.user([search(params={"110008": "Автомат"})])
        self.assertEqual(explicit_stability(u)["status"], "insufficient_days")
        u = self.user([search(t="not-a-date", params={"110008": "Автомат"})])
        self.assertEqual(explicit_stability(u)["status"], "invalid_timestamps")


class OverviewComparisonTests(unittest.TestCase):
    def setUp(self):
        self.payload = build_payload(dataset([card(i) for i in range(12)]))
        u = self.payload["users"][0]
        self.payload = attach_editor_selections(
            self.payload,
            [
                dict(
                    user_index=1,
                    fact_hash=u["fact_hash"],
                    overview_fact_ids=["overview", "topology", "communications"],
                )
            ],
            dict(provider="test", model="test", model_revision="test"),
        )
        self.user = self.payload["users"][0]

    def test_frozen_editor_and_rules_have_equal_budget_and_same_context(self):
        r = compare_overviews(self.user)
        self.assertEqual(r["status"], "available")
        self.assertEqual(len(r["llm_ids"]), len(r["rule_ids"]))
        self.assertEqual(r["budget"], 2)
        self.assertEqual(r["original_llm_count"], 3)
        self.assertNotIn("communications", r["llm_ids"])
        self.assertNotIn("communications", r["rule_ids"])
        self.assertEqual(
            set(r["shared_context_ids"]), {"communications", "country", "temporal"}
        )
        self.assertGreaterEqual(r["jaccard"], 0)
        self.assertLessEqual(r["jaccard"], 1)
        self.assertFalse(r["usefulness_evaluated"])

    def test_research_is_present_in_both_rules_and_editor_builds(self):
        raw = build_payload(dataset([card(i) for i in range(12)]))
        self.assertEqual(raw["version"], "user-summary-v3")
        self.assertIn("research", raw["users"][0])
        self.assertEqual(raw["research_audit"]["compared_users"], 0)
        self.assertEqual(self.payload["research_audit"]["compared_users"], 1)

    def test_without_llm_cache_no_comparison_claim(self):
        u = build_payload(dataset([card(i) for i in range(12)]))["users"][0]
        self.assertEqual(compare_overviews(u)["status"], "not_available")

    def test_context_only_overview_is_not_a_substantive_comparison(self):
        p = build_payload(dataset([card(i) for i in range(12)]))
        u = p["users"][0]
        p = attach_editor_selections(
            p,
            [
                dict(
                    user_index=1,
                    fact_hash=u["fact_hash"],
                    overview_fact_ids=["communications", "country", "temporal"],
                )
            ],
            dict(provider="test", model="test", model_revision="test"),
        )
        r = compare_overviews(p["users"][0])
        self.assertEqual(r["status"], "context_only")
        self.assertEqual(r["budget"], 0)
        self.assertEqual(p["research_audit"]["compared_users"], 0)
        with self.assertRaises(ValueError):
            blind_pair(p["users"][0], 1)

    def test_stale_or_unknown_facts_cannot_enter_comparison(self):
        for patch in (dict(overview=["fake"]), dict(fact_hash="stale")):
            u = copy.deepcopy(self.user)
            if "overview" in patch:
                u["summary"].update(patch)
            else:
                u.update(patch)
            with self.assertRaises(ValueError):
                compare_overviews(u)
        u = copy.deepcopy(self.user)
        u["summary"]["fact_hash"] = "stale"
        with self.assertRaises(ValueError):
            compare_overviews(u)

    def test_blind_pair_does_not_leak_labels_or_listing_ids(self):
        pair, key = blind_pair(self.user, 1)
        self.assertEqual(pair, blind_pair(self.user, 1)[0])
        self.assertEqual(set(key["sides"].values()), {"llm", "rules"})
        self.assertEqual(pair["user_index"], 1)
        self.assertNotIn(self.user["id"], json.dumps(pair))
        self.assertNotIn("llm", pair)
        self.assertNotIn("rules", pair)
        self.assertTrue(pair["A"] and pair["B"] and pair["shared_context"])
        self.assertEqual(len(pair["reference_facts"]), len(self.user["facts"]))
        self.assertTrue(
            all(
                "priority" not in f and "item_ids" not in f
                for f in pair["reference_facts"]
            )
        )

    def test_diverse_baseline_has_equal_budget_and_is_labelled_as_a_heuristic(self):
        r = compare_overviews(self.user)
        self.assertEqual(len(r["diverse_rule_ids"]), r["budget"])
        self.assertGreaterEqual(len(r["diverse_sections"]), len(r["rule_sections"]))
        pair, key = blind_pair(self.user, 1, "diverse")
        self.assertEqual(key["baseline"], "diverse")
        self.assertNotEqual(
            pair["comparison_id"], blind_pair(self.user, 1)[0]["comparison_id"]
        )

    def test_research_does_not_change_existing_fact_hashes_or_overview(self):
        old = copy.deepcopy(self.user)
        enrich_research(self.payload)
        u = self.payload["users"][0]
        self.assertEqual(u["facts"], old["facts"])
        self.assertEqual(u["fact_hash"], old["fact_hash"])
        self.assertEqual(u["summary"], old["summary"])
        self.assertEqual(len(u["research"]["numeric_ci"]), 4)
        self.assertEqual(u["research"]["cluster_ci"], [])
        self.assertEqual(
            u["research"]["cluster_ci_status"], "not_computed_adaptive_groups"
        )

    def test_expert_packages_are_separate_and_do_not_include_the_answer_key(self):
        with tempfile.TemporaryDirectory() as t:
            write_research_reports(self.payload, t)
            for baseline in ("priority", "diverse"):
                with zipfile.ZipFile(Path(t) / f"annotation_{baseline}.zip") as z:
                    self.assertIsNone(z.testzip())
                    self.assertNotIn("blind_summary_key.json", z.namelist())
                    pairs = json.loads(z.read("pairs.json"))
                    self.assertEqual(len(pairs), 1)
                    self.assertNotIn("sides", pairs[0])
                    self.assertNotIn("__BLIND_PAIRS__", z.read("index.html").decode())
                    self.assertIn("reference_facts", z.read("index.html").decode())
                    self.assertIn("разным экспертам", z.read("README.txt").decode())

    def test_blind_html_escapes_script_breakout_without_changing_data(self):
        html = render_blind_html([{"A": ["</script><img src=x onerror=alert(1)>"]}])
        self.assertNotIn("</script><img", html)
        data = html.split('<script id="pairs" type="application/json">', 1)[1].split(
            "</script>", 1
        )[0]
        self.assertEqual(
            json.loads(data)[0]["A"][0], "</script><img src=x onerror=alert(1)>"
        )


if __name__ == "__main__":
    unittest.main()
