"""End-to-end builder outputs and cache failure boundary on synthetic inputs."""

import hashlib, json, subprocess, sys, tempfile, unittest
from pathlib import Path
from tests.test_avito_user_view import card, dataset

ROOT = Path(__file__).resolve().parents[1]


class OutputTests(unittest.TestCase):
    def bundle(self, root):
        (root / "dataset").mkdir()
        (root / "source").mkdir()
        (root / "dataset/users.json").write_text(
            json.dumps(dataset([card(i) for i in range(6)]))
        )
        (root / "source/btc_history.jsonl").write_text("")
        files = []
        for rel in ["dataset/users.json", "source/btc_history.jsonl"]:
            p = root / rel
            files.append(
                dict(
                    path=rel,
                    bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                )
            )
        (root / "manifest.json").write_text(json.dumps(dict(files=files)))

    def test_builder_writes_full_readable_report_and_separate_missing_data_request(
        self,
    ):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.bundle(root)
            out = root / "output"
            r = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/avito/build_user_view.py"),
                    "--input-dir",
                    str(root),
                    "--output-dir",
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("Общение", (out / "all_user_summaries.txt").read_text())
            self.assertIn("16.10", (out / "assignment_report.txt").read_text())
            self.assertIn(
                "LLM-отбор не применялся", (out / "assignment_report.txt").read_text()
            )
            self.assertIn(
                "содержание коммуникаций",
                (out / "missing_data_request.txt").read_text(),
            )

    def test_stale_cache_fails_before_replacing_a_verified_output(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.bundle(root)
            out = root / "output"
            out.mkdir()
            (out / "index.html").write_text("old verified")
            cache = root / "cache.json"
            cache.write_text(
                json.dumps(
                    dict(
                        provenance=dict(
                            provider="test", model="test", model_revision="test"
                        ),
                        selections=[
                            dict(
                                user_index=1,
                                fact_hash="stale",
                                overview_fact_ids=["overview"],
                            )
                        ],
                    )
                )
            )
            r = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/avito/build_user_view.py"),
                    "--input-dir",
                    str(root),
                    "--output-dir",
                    str(out),
                    "--editor-cache",
                    str(cache),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertEqual((out / "index.html").read_text(), "old verified")


if __name__ == "__main__":
    unittest.main()
