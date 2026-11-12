import csv
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_user_view.evaluation import annotation_report, temporal_search_report


def search(day, brand=None, gearbox=None, hour=9, matched=True):
    params = {}
    if brand is not None:
        params["110000"] = brand
    if gearbox is not None:
        params["110008"] = gearbox
    return {
        "t": f"2026-09-{day:02d} {hour:02d}:00:00",
        "params": params,
        "payloadStatus": "matched" if matched else "ambiguous",
    }


def temporal(*histories):
    return temporal_search_report(
        {"users": [{"id": str(i), "searches": h} for i, h in enumerate(histories)]}
    )


class TemporalTests(unittest.TestCase):
    def test_fixed_boundary_recency_and_future_not_used(self):
        a = search(10, "A")
        r = temporal([a] * 20 + [search(11, "B"), search(12, "B"),
                                  search(13, "B", hour=0), search(14, "A")])
        self.assertEqual(r["target_slots"], 1)
        self.assertEqual(r["methods"]["last_explicit"]["hits"], 1)
        self.assertEqual(r["methods"]["mode_by_day"]["hits"], 1)
        self.assertEqual(r["users_with_repeated_projection"], 1)
        r = temporal([search(12, "A"), search(13, "B"), search(14, "B")])
        self.assertEqual(r["methods"]["last_explicit"]["hits"], 0)
        self.assertEqual(r["users_with_repeated_projection"], 0)

    def test_no_later_target_substitution_or_ambiguous_prediction(self):
        r = temporal([search(12, ["A", "B"]), search(13, None), search(14, "A")],
                     [search(12, "A"), search(13, "B", matched=False), search(14, "A")],
                     [search(12, "A"), search(12, "B"), search(13, "A")])
        self.assertEqual(r["target_slots"], 1)
        self.assertEqual(r["users_without_singleton_target"], 1)
        self.assertEqual(r["users_ambiguous_first_future"], 1)
        self.assertEqual(r["methods"]["last_explicit"]["predicted"], 0)
        self.assertEqual(r["methods"]["mode_by_day"]["predicted"], 0)

    def test_user_balanced_common_comparison_and_missing_history(self):
        r = temporal([search(12, "A", "auto"), search(13, "B", "auto")],
                     [search(12, "A"), search(13, "A")],
                     [search(13, "B")])
        self.assertEqual(r["target_slots"], 4)
        self.assertEqual(r["methods"]["last_explicit"]["predicted"], 3)
        self.assertEqual(r["common"]["users"], 2)
        self.assertEqual(r["common"]["last_explicit_user_mean"], 0.75)

    def test_cohort_baseline_excludes_subject_and_future_counts(self):
        r = temporal([search(12, "A"), search(13, "A")],
                     [search(12, "B"), search(14, "A")],
                     [search(12, "B")])
        self.assertEqual(r["methods"]["cohort_mode"]["predicted"], 1)
        self.assertEqual(r["methods"]["cohort_mode"]["hits"], 0)
        self.assertEqual(r["common_with_cohort"]["users"], 1)
        self.assertEqual(r["common_with_cohort"]["last_explicit_user_mean"], 1)

    def test_duplicate_users_fail_before_counting(self):
        with self.assertRaises(ValueError):
            temporal_search_report({"users": [{"id": "same"}, {"id": "same"}]})


KEY = [
    {"comparison_id": "001-P", "baseline": "priority", "sides": {"A": "rules", "B": "llm"}},
    {"comparison_id": "002-P", "baseline": "priority", "sides": {"A": "llm", "B": "rules"}},
]


def answers(*rows):
    f = io.StringIO()
    writer = csv.writer(f)
    fields = ["comparison_id", "user_index", "preference_A_B_tie"]
    for d in ("accuracy", "relevance", "actionability", "clarity"):
        fields += [f"{d}_A_1_5", f"{d}_B_1_5"]
    writer.writerow(fields + ["notes"])
    for row in rows:
        writer.writerow([*row, *([""] * (12 - len(row)))])
    f.seek(0)
    return f


def experts(*files):
    return [(f"expert-{i}", file) for i, file in enumerate(files, 1)]


class AnnotationTests(unittest.TestCase):
    def test_decode_swapped_sides_and_balance_users(self):
        r = annotation_report(
            KEY,
            experts(
                answers(("001-P", 1, "B", 2, 5), ("002-P", 2, "B")),
                answers(("001-P", 1, "tie")),
            ),
        )
        p = r["baselines"]["priority"]
        self.assertEqual(p["votes"], {"llm": 1, "rules": 1, "tie": 1})
        self.assertEqual(p["user_mean_net_preference"], -0.25)
        self.assertEqual(p["rubric"]["accuracy"]["user_mean_llm_minus_rules"], 3.0)
        self.assertEqual(p["multi_rater_users"], 1)
        self.assertEqual(p["agreeing_multi_rater_users"], 0)

    def test_blank_and_insufficient_are_not_ties(self):
        r = annotation_report(
            KEY, experts(answers(("001-P", 1, ""), ("002-P", 2, "insufficient")))
        )
        p = r["baselines"]["priority"]
        self.assertEqual(r["status"], "no_decisive_or_tied_votes")
        self.assertEqual(p["insufficient_votes"], 1)
        self.assertEqual(p["votes"]["tie"], 0)
        self.assertIsNone(p["user_mean_net_preference"])

    def test_reject_bad_keys_and_rows(self):
        for f in (answers(("003-P", 3, "A")), answers(("001-P", 2, "A")),
                  answers(("001-P", 1, "X")), answers(("001-P", 1, "", "9")),
                  answers(("001-P", 1, "A"), ("001-P", 1, "B"))):
            with self.subTest(row=f.getvalue()), self.assertRaises(ValueError):
                annotation_report(KEY, experts(f))
        with self.assertRaises(ValueError):
            annotation_report(KEY + KEY, [])

    def test_reject_multiple_exports_from_same_expert(self):
        with self.assertRaises(ValueError):
            annotation_report(
                KEY,
                [("judge", answers(("001-P", 1, "A"))),
                 ("judge", answers(("001-P", 1, "B")))],
            )
