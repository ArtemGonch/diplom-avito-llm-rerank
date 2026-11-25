import io
from pathlib import Path
import sys
import unittest
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_text2filter.study import assemble, build_packet, evaluate, selected_options


class StudyTests(unittest.TestCase):
    def test_packet_blind_deterministic_disjoint_and_weighted(self):
        raw = io.BytesIO()
        with ZipFile(raw, "w") as archive:
            archive.writestr("query_frequencies.csv", "query_text,search_count\n" +
                             "".join(f"марка {i},{max(2,3000-i) if i < 1250 else 1}\n"
                                     for i in range(1450)) + " МАРКА   0,4\n")
        raw.seek(0)
        with ZipFile(raw) as archive:
            private, blind = build_packet(archive)
            self.assertEqual((private, blind), build_packet(archive))
        self.assertEqual([len(blind[s]) for s in ("train", "dev", "test")], [80, 40, 120])
        all_rows = sum(blind.values(), [])
        self.assertEqual(len({row["id"] for row in all_rows}), 240)
        self.assertTrue(all(row["filters"] is None for row in all_rows))
        self.assertEqual(sum(row["study"]["probability"] == 1 for row in private if row["split"] == "test"), 10)

    def test_score_requires_complete_matching_gold_and_separates_errors(self):
        rows = [{"id": "q", "query": "цена", "filters": None, "unresolved": None}]
        filters = [{"attr_id": "price", "min": None, "max": 9, "start": 0, "end": 4, "evidence": "цена"}]
        predicted = [{"id": "q", "filters": filters, "unresolved": [], "latency_ms": 3}]
        design = [{"id": "q", "split": "test", "study": {"probability": .5, "family_count": 10}}]
        with self.assertRaises(ValueError):
            evaluate(rows, predicted, design)
        rows[0].update(filters=[], unresolved=[])
        scores = evaluate(rows, predicted, design)
        self.assertEqual(scores["families"]["filter_precision"], 0.0)
        self.assertEqual(scores["traffic"]["false_apply_on_empty"], 1.0)
        predicted[0]["status"] = "model_error"
        self.assertEqual(evaluate(rows, predicted, design)["model_fallback_queries"], 1)
        with self.assertRaises(ValueError):
            evaluate(rows, predicted + predicted, design)

    def test_teacher_cannot_invent_options(self):
        options = [{"attr_id": "price"}]
        self.assertEqual(selected_options('{"accept":[0]}', options), options)
        for response in ('{"accept":[1]}', '{"accept":[true]}', '{"accept":[0,0]}', '[]'):
            with self.assertRaises(ValueError):
                selected_options(response, options)


    def test_rejected_rule_keeps_visible_evidence(self):
        row = {"attr_id": "a", "value_id": "1", "start": 0, "end": 7, "evidence": "автомат"}
        baseline = {"filters": [row], "unresolved": []}
        self.assertEqual(assemble(baseline, [])["unresolved"], [
            {"start": 0, "end": 7, "evidence": "автомат", "reason": "model_abstained"}
        ])

    def test_model_keeps_conditions_between_numeric_bounds(self):
        filter_row = {"attr_id": "price", "min": 1, "max": 2, "start": 0, "end": 50,
                      "evidence_parts": [{"start": 0, "end": 10}, {"start": 40, "end": 50}]}
        gap = {"start": 12, "end": 28, "evidence": "панорамная крыша", "reason": "unrecognized"}
        baseline = {"filters": [filter_row], "unresolved": [gap]}
        self.assertEqual(assemble(baseline, [filter_row])["unresolved"], [gap])

    def test_model_only_consumes_its_part_of_unrecognized_span(self):
        query = "Tooyota с люком"
        proposal = {"attr_id": "a", "value_id": "1", "start": 0, "end": 7, "evidence": query[:7]}
        unresolved = {"start": 0, "end": len(query), "evidence": query, "reason": "unrecognized"}
        parsed = assemble({"filters": [], "unresolved": [unresolved]}, [proposal])
        residual, = parsed["unresolved"]
        self.assertEqual(residual["evidence"], "с люком")
        self.assertEqual(query[residual["start"]:residual["end"]], residual["evidence"])
