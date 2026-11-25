import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa
from pyarrow import parquet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_bx_baseline import ABS_TOLERANCE, PRIMARY_LABEL, audit_candidates, audit_label


TARGETS = (
    "target", "btc_s", "fav_s", "btc_wo_fav_s", "btc_s_d7", "fav_s_d7",
    "btc_wo_fav_s_d7", "contact_s", "contact_s_d7", PRIMARY_LABEL,
)


class BXCandidateTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def write(self, name, rows, *, original=False, mutate=None):
        split, group, index, score, labels = zip(*rows, strict=True)
        columns = {
            "split": pa.array(split, type=pa.string()),
            "serp_x": pa.array(group, type=pa.string()),
            "source_row_index": pa.array(index, type=pa.int64()),
            "baseline_score": pa.array(score, type=pa.float64()),
        }
        columns.update({
            label: pa.array([gain.get(label) for gain in labels], type=pa.float64())
            for label in TARGETS
        })
        if original:
            del columns["split"]
        if mutate:
            mutate(columns)
        path = self.directory / f"{name}.parquet"
        parquet.write_table(pa.table(columns), path, row_group_size=1)
        return path

    def test_shards_and_splits_keep_full_groups_and_stable_ties(self):
        first = self.write("one", [
            ("test", "g", 2, 5, {PRIMARY_LABEL: 1}),
            ("dev", "g", 1, 1, {PRIMARY_LABEL: 1}),
            ("test", "h", 4, 2, {}),
        ])
        second = self.write("two", [
            ("test", "g", 1, 5, {PRIMARY_LABEL: 0}),
            ("test", "g", 3, 100, {PRIMARY_LABEL: math.nan}),
            ("dev", "g", 2, 2, {PRIMARY_LABEL: 1}),
        ])
        result = audit_candidates([first, second])
        self.assertEqual(result, audit_candidates([second, first]))
        self.assertEqual((result["rows"], result["groups"]), (6, 3))
        test = result["splits"]["test"]
        dev = result["splits"]["dev"]
        self.assertEqual((test["rows"], test["groups"]), (4, 2))
        self.assertEqual((dev["rows"], dev["groups"]), (2, 1))
        self.assertEqual(set(result["targets"]), set(TARGETS))
        target = test["targets"][PRIMARY_LABEL]
        self.assertAlmostEqual(target["ndcg"], 0.8**2 / (0.8 + 1e-5))
        self.assertEqual(target["eligible_groups_with_score_ties"], 1)
        self.assertEqual(target["missing_rows"], 2)
        self.assertEqual(target["no_observed_labels_groups"], 1)
        self.assertIsNone(dev["targets"][PRIMARY_LABEL]["ndcg"])
        self.assertEqual(dev["targets"][PRIMARY_LABEL]["homogeneous_groups"], 1)

    def test_rejects_invalid_rows_and_types(self):
        row = ("test", "g", 1, 0, {})
        for field, value, dtype in (
            ("split", " ", pa.string()), ("serp_x", None, pa.string()),
            ("source_row_index", -1, pa.int64()), ("source_row_index", 1.0, pa.float64()),
            ("baseline_score", None, pa.float64()), ("baseline_score", math.nan, pa.float64()),
            ("baseline_score", math.inf, pa.float64()), ("baseline_score", True, pa.bool_()),
            ("target", -1.0, pa.float64()), ("target", math.inf, pa.float64()),
            ("target", "1", pa.string()),
        ):
            with self.subTest(field=field, value=value):
                path = self.write("invalid", [row], mutate=lambda cols: cols.update({
                    field: pa.array([value], type=dtype)
                }))
                with self.assertRaises(ValueError):
                    audit_candidates([path])

        missing = self.write("missing", [row], mutate=lambda cols: cols.pop("fav_s"))
        with self.assertRaises(ValueError):
            audit_candidates([missing])

        first = self.write("one", [row])
        duplicate = self.write("two", [("test", "other group", 1, 5, {})])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_candidates([first, duplicate])

    def test_split_is_required_and_explicit_split_checks_embedded_value(self):
        rows = [("test", "g", 1, 25, {}), ("test", "g", 2, 10, {})]
        embedded = self.write("embedded", rows)
        implicit = self.write("implicit", rows, original=True)
        with self.assertRaisesRegex(ValueError, "explicit candidate split"):
            audit_candidates([implicit])
        with self.assertRaisesRegex(ValueError, "conflicting candidate split"):
            audit_candidates(split_paths=[("train", embedded)])
        self.assertEqual(
            audit_candidates([embedded]),
            audit_candidates(split_paths=[("test", implicit)]),
        )

    def test_reconciles_all_keys_scores_and_labels(self):
        gains = dict.fromkeys(TARGETS, 1.0)
        original_rows = [("test", "g", 1, 25, gains), ("test", "g", 2, 10, {"target": 0})]
        original = self.write("original", original_rows, original=True)
        audits = {
            label: audit_label([[0, 1]], [25, 10], [r[4].get(label) for r in original_rows])
            for label in TARGETS
        }
        reference = {"input_sha256": "test-fixture", "rows": 2, "groups": 1, "targets": audits}
        normalized_rows = [("test", "g", 2, 10, {"target": 0, "fav_s": math.nan}), original_rows[0]]
        good = self.write("good", normalized_rows)
        with patch("avito_bx_baseline.audit_preview", return_value=reference):
            result = audit_candidates([good], original=original)
            self.assertEqual(result["reconciliation"]["rows_matched"], 2)
            self.assertEqual(result["reconciliation"]["targets_matched"], len(TARGETS))
            self.assertLessEqual(
                result["reconciliation"]["max_absolute_ndcg_difference"], ABS_TOLERANCE
            )
            for bad_row in (
                ("test", "g", 3, 10, {"target": 0}),
                ("test", "h", 2, 10, {"target": 0}),
                ("test", "g", 2, 9, {"target": 0}),
                ("test", "g", 2, 10, {"target": 0, "btc_s_d7": 0}),
                ("dev", "g", 2, 10, {"target": 0}),
            ):
                with self.subTest(bad_row=bad_row):
                    bad = self.write("bad", [bad_row, original_rows[0]])
                    with self.assertRaises(ValueError):
                        audit_candidates([bad], original=original)
