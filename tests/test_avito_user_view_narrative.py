"""Only grounded fact IDs may enter the LLM editorial selection cache."""

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from avito_user_view import build_payload
from avito_user_view.narrative import (
    validate_selection,
    attach_editor_selections,
    editor_requests,
)
from tests.test_avito_user_view import card, dataset


class NarrativeTests(unittest.TestCase):
    def setUp(self):
        self.payload = build_payload(dataset([card(i) for i in range(6)]))
        self.user = self.payload["users"][0]
        self.selection = dict(
            user_index=1,
            fact_hash=self.user["fact_hash"],
            overview_fact_ids=["overview", "topology"],
        )

    def test_valid_selection_resolves_only_canonical_text(self):
        validate_selection(self.selection, self.user)
        p = attach_editor_selections(
            self.payload,
            [self.selection],
            dict(
                provider="codex_collaboration",
                model="session_inherited",
                model_revision="not_exposed",
            ),
        )
        self.assertEqual(p["users"][0]["summary"]["status"], "llm_fact_selection")
        self.assertEqual(p["users"][0]["summary"]["overview"], ["overview", "topology"])

    def test_stale_hash_unknown_fact_and_free_text_are_rejected(self):
        for patch in [
            dict(fact_hash="stale"),
            dict(overview_fact_ids=["made-up"]),
            dict(text="Пользователь купил машину", n=999),
        ]:
            with self.assertRaises(ValueError):
                validate_selection(dict(self.selection, **patch), self.user)

    def test_tampered_number_or_evidence_rejected_even_with_recomputed_hash(self):
        from avito_user_view.analysis import digest

        for field, value in [("text", "999 покупок"), ("item_ids", ["FOREIGN"])]:
            u = copy.deepcopy(self.user)
            u["facts"][0][field] = value
            u["fact_hash"] = digest(u["facts"])
            with self.assertRaises(ValueError):
                validate_selection(dict(self.selection, fact_hash=u["fact_hash"]), u)

    def test_incomplete_editor_cache_does_not_claim_full_llm_coverage(self):
        p = build_payload(dataset([card(1)], [dict(id="other", cards=[], searches=[])]))
        p = attach_editor_selections(
            p,
            [],
            dict(
                provider="codex_collaboration",
                model="session_inherited",
                model_revision="not_exposed",
            ),
        )
        self.assertEqual(p["editor_audit"]["validated_users"], 0)
        self.assertEqual(p["users"][0]["summary"]["status"], "evidence_rules")

    def test_requests_have_facts_without_user_or_listing_identifiers(self):
        r = editor_requests(self.payload)[0]
        self.assertNotIn("id", r)
        self.assertTrue(
            all("item_ids" not in f and "search_refs" not in f for f in r["facts"])
        )
        self.assertTrue(r["facts"])


if __name__ == "__main__":
    unittest.main()
