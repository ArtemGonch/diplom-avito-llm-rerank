from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "amazon_c4"))

from data.amazon_c4 import (  # noqa: E402
    BM25Index,
    image_filename,
    ranking_metrics,
    select_main_image_url,
    stable_top_k,
)
from evaluate_multimodal import _ordered_item_ids, _zscore  # noqa: E402
from build_candidates import _sequence_sha256  # noqa: E402


class AmazonC4PipelineTests(unittest.TestCase):
    def test_bm25_prefers_lexically_matching_item(self) -> None:
        index = BM25Index(["red car floor mats", "kitchen coffee mug"])
        scores = index.scores("floor mats for my car")
        self.assertGreater(scores[0], scores[1])

    def test_stable_top_k_breaks_ties_by_document_id(self) -> None:
        indices = stable_top_k(np.array([1.0, 2.0, 2.0, 0.0]), 3)
        self.assertEqual(indices.tolist(), [1, 2, 0])

    def test_embedding_cache_hash_detects_middle_row_change(self) -> None:
        self.assertNotEqual(
            _sequence_sha256(["first", "middle-a", "last"]),
            _sequence_sha256(["first", "middle-b", "last"]),
        )

    def test_retrieval_does_not_assume_positive_is_present(self) -> None:
        metrics = ranking_metrics(["a", "b", "c"], "missing", k=2)
        self.assertEqual(metrics["recall@3"], 0.0)
        self.assertEqual(metrics["mrr@2"], 0.0)

    def test_main_image_policy_is_large_then_hi_res_then_thumb(self) -> None:
        images = [
            {"variant": "PT01", "large": "https://example.test/other.jpg"},
            {
                "variant": "MAIN",
                "large": "https://example.test/large.jpg",
                "hi_res": "https://example.test/high.jpg",
                "thumb": "https://example.test/thumb.jpg",
            },
        ]
        self.assertEqual(
            select_main_image_url(images),
            "https://example.test/large.jpg",
        )

    def test_image_filename_cannot_escape_output_directory(self) -> None:
        filename = image_filename("../../unsafe/item")
        self.assertNotIn("/", filename)
        self.assertNotIn("..", filename)
        self.assertTrue(filename.endswith(".jpg"))

    def test_missing_image_is_neutral_in_fusion_normalisation(self) -> None:
        values = np.array([0.2, np.nan, 0.8])
        normalised = _zscore(values, np.isfinite(values))
        self.assertEqual(normalised[1], 0.0)
        self.assertLess(normalised[0], normalised[2])

    def test_ranking_ties_keep_original_candidate_order(self) -> None:
        ranked = _ordered_item_ids(["first", "second", "third"], np.ones(3))
        self.assertEqual(ranked, ["first", "second", "third"])


if __name__ == "__main__":
    unittest.main()
