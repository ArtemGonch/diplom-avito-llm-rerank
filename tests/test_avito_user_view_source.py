"""Technical-field enrichment contracts against synthetic source files only."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avito_user_view.source_io import enrich_technical_attributes


class TechnicalSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "source").mkdir()
        self.data = {
            "users": [
                {
                    "id": "9007199254740993123",
                    "cards": [
                        {
                            "item": "9007199254740993999",
                            "last": "2026-09-10T12:00:00",
                            "engine_type": "Бензин",
                            "price": 1_000_000,
                            "brand": "Synthetic Brand",
                        }
                    ],
                }
            ]
        }
        self.write_sources()

    def write_sources(self, intervals=(), snapshots=(), dictionary=None):
        if dictionary is None:
            dictionary = [
                {"param_id": 110004, "id": 77, "defaultlabel": "5"},
                {"param_id": 110008, "id": 77, "defaultlabel": "2"},
            ]
        for name, rows in (
            ("item_params_history.jsonl", intervals),
            ("item_attribute_value_dictionary.jsonl", dictionary),
        ):
            (self.root / "source" / name).write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
        (self.root / "source/viewer-ads.json").write_text(json.dumps(snapshots))

    def interval(self, start="2026-09-10T11:00:00", end="2026-09-10T13:00:00", **kw):
        row = {
            "item_id": 9007199254740993999,
            "from_ts": start,
            "to_ts": end,
            "json_params": json.dumps({"110011": 150, "110012": 1.6, "110004": 77}),
        }
        row.update(kw)
        return row

    def snapshot(self, stamp="2026-09-11T10:00:00", **kw):
        row = {
            "item_id": "9007199254740993999",
            "snapshot_ts": stamp,
            "params": json.dumps({"110011": 100, "110012": 2, "110004": 77}),
            "vin": "SOURCE_PRIVATE_VIN",
        }
        row.update(kw)
        return row

    def card(self, result):
        return result["users"][0]["cards"][0]

    def test_interval_is_left_inclusive_and_right_exclusive(self):
        earlier = self.interval(end="2026-09-10T12:00:00", json_params={"110011": 80})
        current = self.interval(
            start="2026-09-10T12:00:00", json_params={"110011": 170}
        )
        self.write_sources([earlier, current], [self.snapshot()])
        result, audit = enrich_technical_attributes(self.data, self.root)
        c = self.card(result)
        self.assertEqual(c["power_hp"], 170)
        self.assertEqual(c["technicalTemporalStatus"], "as_of_last")
        self.assertEqual(c["technicalSource"], "source/item_params_history.jsonl")
        self.assertEqual(c["technicalSnapshot"], c["last"])
        self.assertEqual(audit["as_of_last_cards"], 1)

    def test_ambiguous_intervals_fall_back_without_asof_claim(self):
        self.write_sources(
            [self.interval(), self.interval(json_params={"110011": 200})],
            [self.snapshot()],
        )
        result, audit = enrich_technical_attributes(self.data, self.root)
        c = self.card(result)
        self.assertEqual(c["power_hp"], 100)
        self.assertEqual(c["technicalTemporalStatus"], "snapshot_after_last")
        self.assertEqual(c["technicalSource"], "source/viewer-ads.json")
        self.assertEqual(c["technicalSnapshot"], "2026-09-11T10:00:00")
        self.assertEqual(audit["ambiguous_intervals"], 1)

    def test_snapshot_fallback_records_original_timestamp_and_no_mutation(self):
        original = copy.deepcopy(self.data)
        self.write_sources([], [self.snapshot("2026-09-09T10:00:00")])
        result, audit = enrich_technical_attributes(self.data, self.root)
        c = self.card(result)
        self.assertEqual(self.data, original)
        self.assertIsNot(result, self.data)
        self.assertEqual(c["technicalTemporalStatus"], "snapshot_before_or_at_last")
        self.assertEqual(c["price"], original["users"][0]["cards"][0]["price"])
        self.assertEqual(c["brand"], "Synthetic Brand")
        self.assertEqual(audit["fallback_snapshot_cards"], 1)

    def test_doors_dictionary_uses_param_and_value_id_together(self):
        self.write_sources([self.interval()])
        result, _ = enrich_technical_attributes(self.data, self.root)
        self.assertEqual(self.card(result)["doors"], 5)

    def test_zero_liters_for_electric_is_valid_and_sources_do_not_leak(self):
        self.data["users"][0]["cards"][0]["engine_type"] = "Электро"
        params = {
            "110011": 150,
            "110012": 0,
            "110004": 77,
            "1167": 13812,
            "vin": "SOURCE_PRIVATE_VIN",
            "description": "SOURCE_PRIVATE_DESCRIPTION",
        }
        self.write_sources([self.interval(json_params=json.dumps(params))])
        result, _ = enrich_technical_attributes(self.data, self.root)
        c = self.card(result)
        self.assertEqual(c["engine_liters"], 0)
        encoded = json.dumps(result)
        self.assertNotIn("SOURCE_PRIVATE", encoded)
        self.assertNotIn("json_params", encoded)
        self.assertNotIn('"1167"', encoded)
        self.assertNotIn('"params"', encoded)

    def test_nonfinite_boolean_and_out_of_domain_values_are_missing(self):
        cases = [
            ({"110011": "NaN", "110012": "Infinity"}, (None, None)),
            ({"110011": True, "110012": False}, (None, None)),
            ({"110011": 9, "110012": 20.1}, (None, None)),
            ({"110011": 3000.1, "110012": -0.1}, (None, None)),
            ({"110011": "10", "110012": "20"}, (10, 20)),
        ]
        for params, expected in cases:
            with self.subTest(params=params):
                self.write_sources([self.interval(json_params=params)])
                result, _ = enrich_technical_attributes(self.data, self.root)
                c = self.card(result)
                self.assertEqual((c["power_hp"], c["engine_liters"]), expected)

    def test_missing_malformed_or_duplicate_snapshot_cannot_claim_asof(self):
        self.write_sources(
            [self.interval(json_params="{bad")], [self.snapshot(), self.snapshot()]
        )
        result, audit = enrich_technical_attributes(self.data, self.root)
        c = self.card(result)
        self.assertIsNone(c["power_hp"])
        self.assertEqual(c["technicalTemporalStatus"], "not_available")
        self.assertEqual(audit["ambiguous_snapshots"], 1)

    def test_missing_last_can_use_snapshot_but_not_temporal_claim(self):
        self.data["users"][0]["cards"][0]["last"] = ""
        self.write_sources([self.interval()], [self.snapshot()])
        result, _ = enrich_technical_attributes(self.data, self.root)
        self.assertEqual(
            self.card(result)["technicalTemporalStatus"], "snapshot_time_unknown"
        )

    def test_unknown_snapshot_time_is_not_copied_as_unvalidated_source_text(self):
        self.write_sources([], [self.snapshot("SOURCE_PRIVATE_INVALID_TIMESTAMP")])
        result, _ = enrich_technical_attributes(self.data, self.root)
        self.assertEqual(self.card(result)["technicalSnapshot"], "")
        self.assertEqual(
            self.card(result)["technicalTemporalStatus"], "snapshot_time_unknown"
        )
        self.assertNotIn("SOURCE_PRIVATE", json.dumps(result))

    def test_missing_optional_sources_leave_explicit_unavailable_values(self):
        for file in (self.root / "source").iterdir():
            file.unlink()
        result, audit = enrich_technical_attributes(self.data, self.root)
        c = self.card(result)
        self.assertEqual(c["technicalTemporalStatus"], "not_available")
        self.assertEqual(
            (c["power_hp"], c["engine_liters"], c["doors"]), (None, None, None)
        )
        self.assertEqual(len(audit["missing_source_files"]), 3)


if __name__ == "__main__":
    unittest.main()
