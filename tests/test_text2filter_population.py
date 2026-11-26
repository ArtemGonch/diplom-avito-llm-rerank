from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "avito"))
from audit_text2filter_population import audit


class PopulationAuditTests(unittest.TestCase):
    def test_equal_text_counts_can_hide_a_change_in_traffic_coverage(self):
        def run(query, applied):
            span = {"start": 0, "end": len(query), "evidence": query}
            if query == applied:
                return {"filters": [{**span, "attr_id": "test", "value_id": "foo"}], "unresolved": []}
            return {"filters": [], "unresolved": [{**span, "reason": "unrecognized"}]}

        v2 = "v2_control"
        v3 = "v3_verified_qwen_calibrated"
        result = audit({"frequent": 99, "rare": 1}, {
            v2: lambda query: run(query, "rare"),
            v3: lambda query: run(query, "frequent"),
        })
        coverage = result["coverage"]
        self.assertEqual(coverage[v2]["texts"]["with_filter"], 1)
        self.assertEqual(coverage[v3]["texts"]["with_filter"], 1)
        self.assertEqual(coverage[v2]["search_events"]["with_filter"], 1)
        self.assertEqual(coverage[v3]["search_events"]["with_filter"], 99)
        self.assertEqual(result["v2_to_v3"]["search_events"], {
            "filter_set_changed": 100,
            "v3_gained_a_filter_when_v2_empty": 99,
            "v3_empty_when_v2_had_filter": 1,
        })
