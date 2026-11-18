import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_bx_baseline import audit_label, audit_target_components, check_reference


class BXBaselineTests(unittest.TestCase):
    def test_linear_gain_one_based_discount_stable_ties_and_null_mask(self):
        # The first two observed rows share a score, so their original order matters.
        groups = [[0, 1, 2, 3]]
        scores = [8.0, 99.0, 8.0, 1.0]
        labels = [1.0, None, 3.0, 0.0]
        result = audit_label(groups, scores, labels)
        expected = (0.8 + 3 * 0.8**2) / (3 * 0.8 + 0.8**2 + 1e-5)
        self.assertAlmostEqual(result["ndcg"], expected, places=14)
        self.assertEqual(result["missing_rows"], 1)
        self.assertEqual(result["eligible_groups_with_score_ties"], 1)

    def test_skips_uniform_positive_uniform_zero_and_missing(self):
        result = audit_label(
            [[0, 1], [2, 3], [4], [5, 6], [7, 8]],
            [1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 4.0],
            [2.0, 2.0, 0.0, 0.0, None, 1.0, 0.0, 1.0, None],
        )
        expected = 0.8**2 / (0.8 + 1e-5)
        self.assertEqual(result["eligible_groups"], 1)
        self.assertEqual(result["homogeneous_groups"], 3)
        self.assertEqual(result["no_observed_labels_groups"], 1)
        self.assertAlmostEqual(result["ndcg"], expected, places=14)
        self.assertIsNone(audit_label([[0]], [1.0], [0.0])["ndcg"])

    def test_missing_nan_is_masked_but_invalid_gain_fails(self):
        result = audit_label([[0, 1, 2]], [10, 3, 1], [float("nan"), 1, 0])
        self.assertEqual(result["missing_rows"], 1)
        for value in (-1, float("inf"), "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit_label([[0, 1]], [2, 1], [value, 0])

    def test_reconciliation_checks_keys_counts_and_absolute_error(self):
        result = {"label": {"ndcg": 0.5, "eligible_groups": 2}}
        metric = "NDCG_b0.8(label)"
        self.assertLess(
            check_reference(result, {metric: 0.5 + 1e-13}, {metric: 2}), 1e-12
        )
        for means, counts in (
            ({metric: 0.5 + 1e-11}, {metric: 2}),
            ({metric: 0.5}, {metric: 3}),
            ({metric: 0.5}, {metric: True}),
            ({metric: 0.5, "extra": 0}, {metric: 2}),
            ({metric: None}, {metric: 2}),
        ):
            with self.subTest(means=means, counts=counts), self.assertRaises(ValueError):
                check_reference(result, means, counts)

    def test_distinguishes_max_from_addition_and_preserves_denominator(self):
        result = audit_target_components([0.2, 0.2, 0.0], [0.05, 0, 0], [0.2, 0.2, 0])
        self.assertEqual(result["rows_checked"], 3)
        self.assertEqual(result["equals_max"], 3)
        self.assertEqual(result["equals_sum"], 2)
        with self.assertRaises(ValueError):
            audit_target_components([0.2], [0], [])
        with self.assertRaises(ValueError):
            audit_target_components([0.2], [None], [0.2])
