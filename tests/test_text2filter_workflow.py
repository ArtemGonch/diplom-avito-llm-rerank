import io
import json
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


class PairedStudyTests(unittest.TestCase):
    def test_paired_differences_keep_census_fixed_and_estimate_both_weights(self):
        from avito_text2filter.study import paired_exact_filters

        def rows(accepted):
            return [{"id": i, "filters": filters} for i, filters in enumerate(accepted)]

        brand = {"attr_id": "110000", "value_id": "10"}
        gold = rows([[brand], [brand], []])
        control = rows([[], [], []])
        augmented = rows([[brand], [brand], [brand]])
        manifest = [
            {"id": 0, "split": "test", "study": {"stratum": "top", "probability": 1, "family_count": 20}},
            {"id": 1, "split": "test", "study": {"stratum": "tail", "probability": .1, "family_count": 2}},
            {"id": 2, "split": "test", "study": {"stratum": "tail", "probability": .1, "family_count": 2}},
        ]
        report = paired_exact_filters(gold, control, augmented, manifest)
        self.assertEqual(report["wins"], 2)
        self.assertEqual(report["losses"], 1)
        self.assertAlmostEqual(report["families"]["difference"], 1 / 21)
        self.assertAlmostEqual(report["traffic"]["difference"], 1 / 3)
        self.assertEqual(report, paired_exact_filters(gold, control, augmented, manifest))


class TeacherConsensusTests(unittest.TestCase):
    def test_consensus_requires_a_valid_agreeing_set_in_both_orders(self):
        from avito_text2filter.study import consensus_choices

        options = [{"attr_id": "price", "min": 5}, {"attr_id": "year", "max": 9},
                   {"attr_id": "body", "value_id": "12"}]
        self.assertEqual(consensus_choices('{"accept":[2,0]}', '{"accept":[2,0]}', options),
                         [options[0], options[2]])
        self.assertEqual(consensus_choices('{"accept":[]}', '{"accept":[]}', options), [])
        self.assertIsNone(consensus_choices('{"accept":[0]}', '{"accept":[0]}', options))
        for response in ('garbage', '{"accept":[99]}'):
            with self.assertRaises(ValueError):
                consensus_choices('{"accept":[0]}', response, options)

    def test_v2_can_keep_grounded_make_and_model_on_one_span(self):
        from avito_text2filter.study import assemble

        base = {"filters": [], "unresolved": [
            {"reason": "ambiguous", "start": 0, "end": 4, "evidence": "тест"},
            {"reason": "negated", "start": 5, "end": 6, "evidence": "x"},
        ]}
        model = {"attr_id": "110001", "value_id": "10", "start": 0, "end": 4, "evidence": "тест"}
        make = {**model, "attr_id": "110000", "value_id": "11"}
        with self.assertRaises(ValueError):
            assemble(base, [make, model])
        result = assemble(base, [make, model], allow_linked=True)
        self.assertEqual(len(result["filters"]), 2)
        self.assertEqual([row["reason"] for row in result["unresolved"]], ["negated"])

    def test_teacher_input_may_not_reuse_reference_families(self):
        from avito_text2filter.study import ensure_disjoint_queries

        manifest = [{"family_key": "тойота rav4"}]
        ensure_disjoint_queries([{"id": "a", "query": " Honda  Civic "}], manifest)
        for rows in (
            [{"id": "a", "query": "ТОЙОТА   RAV4"}],
            [{"id": "a", "query": "Honda"}, {"id": "b", "query": "honda"}],
            [{"id": "x", "query": "Honda"}, {"id": "x", "query": "BMW"}],
        ):
            with self.assertRaises(ValueError):
                ensure_disjoint_queries(rows, manifest)


class StudentSamplingTests(unittest.TestCase):
    def test_weak_pairs_are_balanced_and_do_not_replace_reference_pairs(self):
        import random
        from avito_text2filter.student import training_batch

        ref = {0: [("ref-negative", "", 0)], 1: [("ref-positive", "", 1)]}
        weak = {0: [("weak-negative", "", 0)] * 1000, 1: [("weak-positive", "", 1)] * 1000}
        result = training_batch(random.Random(4), ref, weak)
        self.assertEqual(sum(row[0].startswith("ref") for row in result), 8)
        self.assertEqual(sum(row[2] for row in result), 8)
        self.assertEqual(len(training_batch(random.Random(4), ref, {0: [], 1: []})), 16)


class AuditedPairTests(unittest.TestCase):
    def test_unverified_options_are_ignored_instead_of_treated_as_negatives(self):
        from avito_text2filter.student import labeled_examples

        def option(value):
            return {"attr_id":"110000","attr_name":"Марка","value_id":value,
                    "value_name":value,"evidence":"марка"}
        options = [option("yes"), option("no"), option("unverified")]
        rows = [{"query":"марка","status":"audited_pairs",
                 "filters":[options[0]],"verified_rejections":[options[1]]}]
        result,missed = labeled_examples(None, lambda *_: ({}, options), rows)
        self.assertEqual([target for _,_,target in result], [1,0])
        self.assertEqual(missed,0)

    def test_audited_pair_rejects_contradictory_labels(self):
        from avito_text2filter.student import labeled_examples

        row={"attr_id":"110000","attr_name":"Марка","value_id":"5"}
        source={"query":"марка","status":"audited_pairs","filters":[row],"verified_rejections":[row]}
        with self.assertRaises(ValueError):
            labeled_examples(None,lambda *_: ({},[]),[source])


    def test_v3_does_not_overrepeat_one_verified_negative_in_each_batch(self):
        import random
        from avito_text2filter.student import training_batch

        reference = {0: [("ref-negative", "", 0)], 1: [("ref-positive", "", 1)]}
        weak = {0: [("one-weak-negative", "", 0)], 1: [("one-weak-positive", "", 1)]}
        batch = training_batch(random.Random(4), reference, weak, distinct_weak=True)
        self.assertEqual(len(batch), 16)
        self.assertEqual(sum(row[0] == "one-weak-negative" for row in batch), 1)
        self.assertEqual(sum(row[0] == "one-weak-positive" for row in batch), 1)


class V4ProposalTests(unittest.TestCase):
    def test_v4_options_use_the_guarded_baseline(self):
        import runpy

        q = "Toyota Camry 2.5%"
        unreliable = {"attr_id": "110012", "attr_name": "Объём двигателя", "min": 2.5,
                      "max": 2.5, "start": 13, "end": 16, "evidence": "2.5"}

        class Catalog:
            def parse(self, query):
                return {"filters": [unreliable], "unresolved": []}

            def baseline_v4(self, query, baseline):
                return {"filters": [], "unresolved": [
                    {**unreliable, "reason": "unsupported"}
                ]}

            def model_candidates_v4(self, query, baseline):
                if baseline["filters"]:
                    raise AssertionError("The linker received the unguarded baseline")
                return []

        script = Path(__file__).resolve().parents[1] / "scripts/avito/text2filter.py"
        propose = runpy.run_path(str(script))["proposals_v4"]
        baseline, options = propose(Catalog(), q)
        self.assertFalse(options)
        self.assertEqual(baseline["unresolved"][0]["reason"], "unsupported")


class DemoPageTests(unittest.TestCase):
    def test_portable_demo_embeds_results_from_both_supplied_parsers(self):
        import runpy

        class Catalog:
            def parse(self, query):
                return {"filters": [], "unresolved": [{"evidence": query, "reason": "unsupported"}]}

        def student(query):
            return {"filters": [], "unresolved": [{"evidence": query, "reason": "unrecognized"}]}

        script = Path(__file__).resolve().parents[1] / "scripts/avito/text2filter.py"
        build = runpy.run_path(str(script))["demo_page"]
        html = build(Catalog(), student=student, student_label="</script>", offline=True).decode()
        start = '<script id="demo-data" type="application/json">'
        config = json.loads(html.split(start, 1)[1].split("</script>", 1)[0])
        self.assertEqual(config["student_label"], "</script>")
        self.assertTrue(config["offline"])
        for query in config["examples"]:
            self.assertEqual(config["replays"][query]["rules"]["unresolved"][0]["reason"], "unsupported")
            self.assertEqual(config["replays"][query]["bert"]["unresolved"][0]["reason"], "unrecognized")
        self.assertEqual(config["benchmark"]["queries"], 80)
        self.assertEqual(config["benchmark"]["rows"][2]["raw_exact"], 54)
