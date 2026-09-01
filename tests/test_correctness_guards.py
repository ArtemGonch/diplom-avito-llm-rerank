from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "exp3rt"))
sys.path.insert(0, str(ROOT / "scripts" / "ur4rec"))
sys.path.insert(0, str(ROOT / "scripts" / "avito"))

from common.llm.generate import (  # noqa: E402
    KnowledgeStore,
    ShardKnowledgeWriter,
    TemplateKnowledgeGenerator,
    merge_knowledge_shards,
)
from common.llm.hf_chat_generator import _generated_continuations  # noqa: E402
from common.metrics import ndcg_at_k  # noqa: E402
from data.ml1m import MovieLens1M, RerankSample  # noqa: E402
from models.ur4rec.masks import proxy_item_mask  # noqa: E402
from models.ur4rec.retriever import RetrieverBlock, UR4RecRetriever  # noqa: E402
from models.ur4rec.text_encoder import FrozenTextEncoder  # noqa: E402
from exp3rt_avito_attribute_rerank import (  # noqa: E402
    heuristic_scores,
    score_diagnostics,
)
from run_ur4rec import _user_preference_prompt_for_sample  # noqa: E402
from run_local_catboost import (  # noqa: E402
    FORBIDDEN_RANK_TIME_COLUMNS,
    rank_time_feature_columns,
)
from run_llm_pointwise_diagnostic import (  # noqa: E402
    FORBIDDEN_PROMPT_FEATURES,
    build_prompt,
    merge_exact_catboost_scores,
    prompt_feature_columns,
)


class CorrectnessGuardTests(unittest.TestCase):
    def test_local_catboost_excludes_post_exposure_signals(self) -> None:
        self.assertTrue(
            FORBIDDEN_RANK_TIME_COLUMNS.isdisjoint(rank_time_feature_columns())
        )

    def test_local_llm_prompt_excludes_post_exposure_and_position_signals(self) -> None:
        self.assertTrue(FORBIDDEN_PROMPT_FEATURES.isdisjoint(prompt_feature_columns()))
        row = pd.Series({column: "safe-value" for column in prompt_feature_columns()})
        prompt = build_prompt(row)
        for forbidden in FORBIDDEN_PROMPT_FEATURES:
            self.assertNotIn(forbidden, prompt)

    def test_catboost_score_join_rejects_extra_candidates(self) -> None:
        frame = pd.DataFrame(
            [{"split": "test", "serp_x": "s", "item_id": 1}]
        )
        scores = pd.DataFrame(
            [
                {"split": "test", "serp_x": "s", "item_id": 1, "catboost_score": 0.1},
                {"split": "test", "serp_x": "s", "item_id": 2, "catboost_score": 0.2},
            ]
        )
        with self.assertRaisesRegex(ValueError, "extra=1"):
            merge_exact_catboost_scores(frame, scores)

    def test_ndcg_preserves_graded_relevance(self) -> None:
        labels = np.array([1.0, 0.1])
        reversed_scores = np.array([0.0, 1.0])
        self.assertLess(ndcg_at_k(labels, reversed_scores, 2), 1.0)

    def test_proxy_item_mask_isolates_only_distinct_items(self) -> None:
        mask = proxy_item_mask(2, 3, torch.device("cpu"))
        item_block = mask[2:, 2:]
        self.assertTrue(torch.equal(torch.diag(item_block), torch.zeros(3)))
        off_diagonal = item_block[~torch.eye(3, dtype=torch.bool)]
        self.assertTrue(torch.isneginf(off_diagonal).all())
        self.assertTrue(torch.equal(mask[:2], torch.zeros(2, 5)))

    def test_decoder_continuation_starts_after_full_left_padded_prompt(self) -> None:
        output = torch.tensor([[0, 0, 11, 12, 21], [13, 14, 15, 16, 22]])
        continuation = _generated_continuations(output, padded_prompt_width=4)
        self.assertTrue(torch.equal(continuation, torch.tensor([[21], [22]])))

    def test_template_generator_implements_batch_contract(self) -> None:
        generator = TemplateKnowledgeGenerator()
        prompts = ["one", "two"]
        self.assertEqual(
            len(generator.generate_item_knowledge_batch(prompts)),
            len(prompts),
        )
        self.assertEqual(
            len(generator.generate_user_preference_batch(prompts)),
            len(prompts),
        )

    def test_preference_aggregation_keeps_memory_tokens(self) -> None:
        user = torch.tensor([1.0, 2.0])
        items = [torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
        memory = FrozenTextEncoder.aggregate_preference(None, user, items)
        self.assertEqual(tuple(memory.shape), (3, 2))

    def test_hashing_smoke_encoder_is_deterministic(self) -> None:
        encoder = FrozenTextEncoder(
            use_hashing_encoder=True,
            hashing_hidden_size=12,
        )
        first = encoder.encode(["same text", "different text"])
        second = encoder.encode(["same text", "different text"])
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (2, 12))

    def test_retriever_memory_keeps_multiple_keys(self) -> None:
        retriever = object.__new__(UR4RecRetriever)
        retriever.hidden_size = 4
        memory = torch.randn(2, 3, 4)
        self.assertIs(retriever._memory_tokens(memory), memory)

    def test_bert_initialisation_copies_self_attention(self) -> None:
        config = BertConfig(
            hidden_size=12,
            intermediate_size=48,
            num_attention_heads=3,
            num_hidden_layers=1,
        )
        config._attn_implementation = "eager"
        source = BertLayer(config)
        block = RetrieverBlock(hidden_size=12, num_heads=3, dropout=0.0)
        UR4RecRetriever._init_block_from_bert(block, source)
        self.assertTrue(
            torch.equal(
                block.self_attn.in_proj_weight[:12],
                source.attention.self.query.weight,
            )
        )

    def test_avito_heuristic_does_not_read_evaluation_signals(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "title": "BMW X5",
                    "brand": "BMW",
                    "model_name": "X5",
                    "price": 2_000_000,
                    "contacts_daily": 0,
                    "clicks_daily": 0,
                },
                {
                    "title": "BMW X5",
                    "brand": "BMW",
                    "model_name": "X5",
                    "price": 2_000_000,
                    "contacts_daily": 999,
                    "clicks_daily": 999,
                },
            ]
        )
        scores = heuristic_scores(
            "The user prefers BMW X5. Typical contact price around 2000000.",
            "BMW X5",
            rows,
            users=None,
            user_id=None,
        )
        self.assertEqual(scores[0], scores[1])

    def test_avito_heuristic_flags_degenerate_all_tie_rankings(self) -> None:
        diagnostics = score_diagnostics(
            [np.zeros(3, dtype=np.float64), np.ones(2, dtype=np.float64)]
        )
        self.assertEqual(diagnostics["constant_score_serps"], 2)
        self.assertEqual(diagnostics["all_zero_score_serps"], 1)
        self.assertEqual(diagnostics["constant_score_fraction"], 1.0)

    def test_temporal_split_keeps_users_and_targets_after_history(self) -> None:
        data = object.__new__(MovieLens1M)
        data.movies = pd.DataFrame(
            {
                "movieId": list(range(1, 41)),
                "title": [f"movie-{i}" for i in range(1, 41)],
                "genres": ["Drama"] * 40,
            }
        )
        data.user_items = {
            1: list(range(1, 16)),
            2: list(range(6, 21)),
        }
        train, val, test = data.build_temporal_rerank_splits(
            history_len=4,
            num_candidates=10,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42,
        )
        self.assertEqual(
            {s.user_id for s in train},
            {s.user_id for s in val},
        )
        self.assertEqual(
            {s.user_id for s in train},
            {s.user_id for s in test},
        )
        for split in (train, val, test):
            for sample in split:
                self.assertNotIn(sample.target_item_id, sample.history_item_ids)
                self.assertEqual(len(sample.candidate_item_ids), 10)
                self.assertEqual(sum(sample.labels), 1)
        for tr, va, te in zip(train, val, test):
            sequence = data.user_items[tr.user_id]
            self.assertLess(
                sequence.index(tr.target_item_id),
                sequence.index(va.target_item_id),
            )
            self.assertLess(
                sequence.index(va.target_item_id),
                sequence.index(te.target_item_id),
            )

    def test_user_profile_prompt_uses_history_not_held_out_target(self) -> None:
        data = object.__new__(MovieLens1M)
        data.movies = pd.DataFrame(
            {
                "movieId": [1, 2, 3],
                "title": ["History One", "History Two", "Held Out Target"],
                "genres": ["Drama", "Comedy", "Horror"],
            }
        )
        sample = RerankSample(
            user_id=1,
            history_item_ids=[1, 2],
            candidate_item_ids=[3],
            labels=[1],
            target_item_id=3,
        )
        prompt = _user_preference_prompt_for_sample(data, sample)
        self.assertIn("History One", prompt)
        self.assertIn("History Two", prompt)
        self.assertNotIn("Held Out Target", prompt)

    def test_knowledge_shards_require_complete_matching_metadata(self) -> None:
        signature = {
            "cache_version": 2,
            "generator": "template",
            "model_name": "template",
            "max_new_tokens": 8,
            "num_shards": 2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = KnowledgeStore(root)
            all_users = {str(i): f"user-{i}" for i in range(6)}
            all_items = {str(i): f"item-{i}" for i in range(8)}
            writers = []
            for shard_id in range(2):
                writer = ShardKnowledgeWriter(
                    store,
                    shard_id,
                    2,
                    {},
                    {},
                    signature,
                )
                writers.append(writer)
            writers[0].save_partial(all_users, all_items)
            writers[1].save_final(all_users, all_items)
            final_meta = {**signature, "complete": True}
            with self.assertRaises(ValueError):
                merge_knowledge_shards(root, 2, meta=final_meta)
            writers[0].save_final(all_users, all_items)
            n_items, n_users = merge_knowledge_shards(
                root,
                2,
                meta=final_meta,
            )
            self.assertEqual((n_items, n_users), (8, 6))
            self.assertEqual(json.loads((root / "meta.json").read_text()), final_meta)
            # A completed merge is safely idempotent and cannot duplicate keys.
            self.assertEqual(
                merge_knowledge_shards(root, 2, meta=final_meta),
                (8, 6),
            )


if __name__ == "__main__":
    unittest.main()
