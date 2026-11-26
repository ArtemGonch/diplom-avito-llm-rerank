import json
from hashlib import sha256
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
from pyarrow import parquet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from avito_bx_embedding_cg import compare_embedding_cg


class EmbeddingCGTests(unittest.TestCase):
    def test_recall_keeps_unknown_items_in_top_k_but_bx_masks_them(self):
        rows = [
            ("g", 1, 1, 3.0, None),
            ("g", 2, 2, 2.0, 0.0),
            ("g", 3, 3, 1.0, 1.0),
            ("missing", 1, 4, 9.0, 0.0),
            ("missing", 2, 5, 2.0, 1.0),
            ("all-positive", 1, 1, 6.0, 1.0),
            ("all-positive", 2, 3, 5.0, 1.0),
        ]
        queries = {"g": np.array([1.0, 0.0]), "all-positive": np.array([1.0, 0.0])}
        items = {
            1: np.array([0.8, 0.6]),
            2: np.array([0.0, 1.0]),
            3: np.array([1.0, 0.0]),
        }
        result = compare_embedding_cg(rows, queries, items, (1, 2))
        self.assertEqual(result["groups"], 3)
        self.assertEqual(result["groups_with_all_embeddings"], 2)
        self.assertEqual(result["groups_missing_query_embedding"], 1)
        self.assertEqual(result["groups_missing_item_embedding"], 1)
        all_groups = result["all_groups"]
        self.assertEqual(all_groups["recall_groups"], 3)
        self.assertEqual(all_groups["bx_ndcg_groups"], 2)
        self.assertAlmostEqual(all_groups["bx"]["recall@1"], 1 / 6)
        self.assertAlmostEqual(all_groups["embedding_cg"]["recall@1"], 1 / 2)
        self.assertAlmostEqual(all_groups["bx"]["recall@2"], 2 / 3)
        self.assertAlmostEqual(all_groups["embedding_cg"]["recall@2"], 1.0)
        self.assertAlmostEqual(all_groups["bx"]["bx_ndcg"], 0.8**2 / (0.8 + 1e-5))
        self.assertAlmostEqual(
            all_groups["embedding_cg"]["bx_ndcg"], (0.8 + 0.8**2) / (2 * (0.8 + 1e-5))
        )
        self.assertEqual(result["complete_groups"]["recall_groups"], 2)

    def test_rejects_reopened_group_and_duplicate_candidates(self):
        row = ("g", 1, 1, 1.0, 1.0)
        first_next = ("h", 2, 2, 1.0, 0.0)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            compare_embedding_cg([row, first_next, row], {}, {}, (1,))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare_embedding_cg([row, row], {}, {}, (1,))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare_embedding_cg([row, ("g", 1, 2, 1.0, 0.0)], {}, {}, (1,))

    def test_cli_uses_cosine_and_validates_vector_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "candidates.parquet"
            columns = {
                "serp_x": ["g", "h", "g"], "source_row_index": [1, 3, 2], "split": ["test"] * 3,
                "snapshot_id": [1, 1, 2], "baseline_score": [9.0, 0.0, 1.0],
                "target_contact_s_d7": [0.0, 0.0, 1.0],
            }
            parquet.write_table(pa.table(columns), path, row_group_size=1)
            metadata = {
                b"embedding_model": b"fixture", b"bx_split": b"test",
                b"bx_candidate_sha256": sha256(path.read_bytes()).hexdigest().encode(),
            }
            for folder, key, values, vectors in (
                ("query", "serp_x", ["g"], [[3.0, 0.0]]),
                ("item", "snapshot_id", [1, 2], [[2.0, 10.0], [1.0, 0.0]]),
            ):
                base = root / folder
                base.mkdir()
                keys = pa.table({key: values}).replace_schema_metadata(metadata)
                parquet.write_table(keys, base / "keys.parquet")
                np.save(base / "vectors.npy", np.asarray(vectors, dtype=np.float32))
            command = [
                sys.executable, str(ROOT / "scripts/avito/evaluate_bx_embedding_cg.py"),
                "--candidates", str(path), "--item-embeddings", str(root / "item"),
                "--query-embeddings", str(root / "query"), "--split", "test", "--k", "1",
            ]
            result = json.loads(subprocess.check_output(command))
            self.assertEqual(result["all_groups"]["bx"]["recall@1"], 0.0)
            self.assertEqual(result["all_groups"]["embedding_cg"]["recall@1"], 1.0)
            self.assertEqual(result["groups"], 2)
            item_keys = root / "item/keys.parquet"
            keys = parquet.read_table(item_keys)
            parquet.write_table(keys.replace_schema_metadata({**metadata, b"bx_split": b"train"}), item_keys)
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("do not match the candidate split and SHA-256", rejected.stderr)
            parquet.write_table(keys.replace_schema_metadata({
                **metadata, b"bx_candidate_sha256": b"0" * 64,
            }), item_keys)
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("do not match the candidate split and SHA-256", rejected.stderr)
            parquet.write_table(keys, item_keys)
            np.save(root / "item/vectors.npy", np.asarray([[0, 0], [1, 0]], dtype=np.float32))
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("invalid or zero vectors", rejected.stderr)

    def test_staging_rejects_source_row_duplicates_across_groups(self):
        import runpy

        ordered = runpy.run_path(str(ROOT / "scripts/avito/evaluate_bx_embedding_cg.py"))["ordered_candidates"]
        with self.assertRaisesRegex(ValueError, "Duplicate source_row_index"):
            with ordered(iter([("g", 1, 5, 1.0, 1.0), ("h", 1, 7, 1.0, 0.0)])):
                pass
        high = 2**64 - 1
        original = [(high, 1, 5, 1.0, 1.0), (str(high), 2, 7, 1.0, 0.0)]
        with ordered(iter(original)) as staged:
            self.assertCountEqual(list(staged), original)
