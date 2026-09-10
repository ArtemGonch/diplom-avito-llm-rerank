from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "avito"))

from avito_case_browser import (  # noqa: E402
    INSIGHT_VALIDATION_VERSION,
    build_browser_payload,
    build_insight_requests,
    build_item_request_rows,
    deterministic_order,
    extract_json_object,
    load_item_overlay,
    normalize_user_id,
    render_browser_html,
)
from avito_case_browser.core import (  # noqa: E402
    MAX_OVERLAY_IMAGE_BYTES,
    POST_EXPOSURE_FIELDS,
    copy_overlay_images,
    create_share_bundle,
)
from avito_case_browser.insights import (  # noqa: E402
    default_insight_contract,
    make_token_provenance_verifier,
    parsed_insight_sha256,
    text_sha256,
    validate_and_normalise_insight,
    validate_cached_insight,
)
from generate_case_insights import (  # noqa: E402
    _cache_signature,
    _generation_contract,
    _guarded_fallback,
    _load_cache,
    _load_requests,
    _validate_parsed,
)
from build_case_browser import (  # noqa: E402
    _bundle_should_be_written,
    _insight_build_status,
    _load_insight_contract,
)
from run_local_catboost import (  # noqa: E402
    CANONICAL_V2_GENERATOR_CODE_FINGERPRINTS,
    _display_path,
    _evaluate_frame as _evaluate_catboost,
    _preflight_model_outputs,
    _source_order_scores,
    _verify_reusable_model,
)
from run_llm_pointwise_diagnostic import _evaluate as _evaluate_llm  # noqa: E402


def _item(serp: str, item_id: int, *, contact: float, tc: int, user: float = 7.0) -> dict:
    return {
        "serp_x": serp,
        "item_id": item_id,
        "block": "items",
        "block_pos": item_id,
        "query_infm_logical_category": "Transport.UsedCars",
        "query_loc": 123.0,
        "user_id": user,
        "contacts_daily": contact,
        "has_tc_events": tc,
        "has_x_events": 0,
        "title": f"Car {item_id}",
        "description_short": f"Seller text {item_id}",
        "brand": "Brand",
        "model_name": f"Model {item_id}",
        "price": 1_000_000.0 + item_id,
        "mileage_km": 10_000.0,
        "year_text": "2020",
        "body_type": "sedan",
        "fuel_text": "petrol",
        "gearbox_text": "auto",
        "drive_text": "front",
        "doors_text": "4",
        "image_count": 3.0,
    }


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    items = pd.DataFrame(
        [
            _item("s1", 1, contact=1.0, tc=1),
            _item("s1", 2, contact=4.0, tc=0),
        ]
    )
    history = pd.DataFrame(
        [
            {
                "user_id": 7,
                "item_id": 99,
                "contact_date": "2026-05-01",
                "brand": "Raw category",
                "model_name": "Raw subcategory",
            }
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "split": "test",
                "serp_x": "s1",
                "item_id": 1,
                "evaluation_relevance": 0.25,
                "catboost_score": 0.1,
                "llm_score": 0.9,
                "gated_blend_score": 0.2,
                "llm_called": True,
            },
            {
                "split": "test",
                "serp_x": "s1",
                "item_id": 2,
                "evaluation_relevance": 1.0,
                "catboost_score": 0.9,
                "llm_score": 0.1,
                "gated_blend_score": 0.8,
                "llm_called": True,
            },
        ]
    )
    return items, history, scores


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(path, format="PNG")


def _write_image(path: Path, image_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(25, 80, 140)).save(path, format=image_format)


class _DeterministicTestTokenizer:
    """Small tokenizer double that exercises real provenance recomputation."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "\n".join(f"<{row['role']}>{row['content']}" for row in messages) + "\n<assistant>"

    def __call__(self, prompts, **_kwargs):
        import torch

        encoded = [max(1, byte) for byte in prompts[0].encode("utf-8")[:128]]
        tensor = torch.tensor([encoded], dtype=torch.int64)
        return {
            "input_ids": tensor,
            "attention_mask": torch.ones_like(tensor),
        }


class AvitoCaseBrowserTests(unittest.TestCase):
    def test_normalize_user_id_handles_parquet_float_and_null(self) -> None:
        self.assertEqual(normalize_user_id(123), "123")
        self.assertEqual(normalize_user_id(123.0), "123")
        self.assertEqual(normalize_user_id("123.0"), "123")
        self.assertIsNone(normalize_user_id(np.nan))
        with self.assertRaises(ValueError):
            normalize_user_id(1.5)

    def test_deterministic_order_uses_item_id_for_score_ties(self) -> None:
        order = deterministic_order([0.5, 0.5, 0.7], [20, 10, 30])
        self.assertEqual(order.tolist(), [2, 1, 0])

    def test_source_order_keeps_sentinel_positions_after_main_items(self) -> None:
        frame = pd.DataFrame(
            [
                {"serp_x": "s", "item_id": 1, "block": "extra-reg", "block_pos": -1},
                {"serp_x": "s", "item_id": 3, "block": "items", "block_pos": 1},
                {"serp_x": "s", "item_id": 2, "block": "items", "block_pos": 0},
            ]
        )
        scores = _source_order_scores(frame)
        ranked_ids = frame.iloc[np.argsort(-scores, kind="stable")]["item_id"].tolist()
        self.assertEqual(ranked_ids, [2, 3, 1])

    def test_llm_metrics_use_item_id_as_explicit_score_tie_break(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "serp_x": "s",
                    "item_id": 2,
                    "evaluation_relevance": 0.0,
                    "score": 0.5,
                },
                {
                    "serp_x": "s",
                    "item_id": 1,
                    "evaluation_relevance": 1.0,
                    "score": 0.5,
                },
            ]
        )
        metrics = _evaluate_llm(frame, "score")
        self.assertEqual(metrics["ndcg@5"], 1.0)
        self.assertEqual(metrics["map@5"], 1.0)

    def test_catboost_metrics_use_item_id_ties_independent_of_row_order(self) -> None:
        frame = pd.DataFrame(
            [
                {"serp_x": "s", "item_id": 2, "_evaluation_relevance": 0.0},
                {"serp_x": "s", "item_id": 1, "_evaluation_relevance": 1.0},
                {"serp_x": "s", "item_id": 3, "_evaluation_relevance": 0.0},
            ]
        )
        scores = np.array([0.5, 0.5, 0.0], dtype=np.float64)
        expected = _evaluate_catboost(frame, scores)
        permutation = np.array([1, 2, 0])
        actual = _evaluate_catboost(
            frame.iloc[permutation].reset_index(drop=True), scores[permutation]
        )
        self.assertEqual(actual, expected)
        self.assertEqual(expected["ndcg@5"], 1.0)

    def test_catboost_artifact_path_supports_external_verification_output(self) -> None:
        external = Path("/tmp/avito-catboost-verification/scores.parquet")
        self.assertEqual(_display_path(external), str(external))
        self.assertEqual(
            _display_path(ROOT / "items_with_attrs.parquet"), "items_with_attrs.parquet"
        )

    def test_catboost_reuse_verifies_metadata_and_model_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "seed_42.cbm"
            metadata = root / "seed_42.meta.json"
            expected = {"version": 2, "seed": 42}
            model.write_bytes(b"frozen model bytes")
            model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
            metadata.write_text(
                json.dumps({**expected, "model_sha256": model_hash}), encoding="utf-8"
            )
            _verify_reusable_model(model, metadata, expected)

            model.write_bytes(b"tampered model bytes")
            with self.assertRaisesRegex(ValueError, "corrupted model reuse"):
                _verify_reusable_model(model, metadata, expected)

    def test_catboost_preflight_fails_closed_for_partial_reuse_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models"
            model_dir.mkdir()
            scores = root / "scores.parquet"
            metrics = root / "metrics.json"
            (model_dir / "seed_42.meta.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "partial canonical reuse"):
                _preflight_model_outputs(
                    model_dir=model_dir,
                    seeds=[42],
                    scores_output=scores,
                    metrics_output=metrics,
                    reuse_existing_models=True,
                    force_retrain=False,
                )

            (model_dir / "seed_42.cbm").write_bytes(b"model")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                _preflight_model_outputs(
                    model_dir=model_dir,
                    seeds=[42],
                    scores_output=scores,
                    metrics_output=metrics,
                    reuse_existing_models=False,
                    force_retrain=False,
                )

            _preflight_model_outputs(
                model_dir=model_dir,
                seeds=[42],
                scores_output=scores,
                metrics_output=metrics,
                reuse_existing_models=False,
                force_retrain=True,
            )

    def test_catboost_reuse_preserves_pinned_generator_code_across_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "seed_42.cbm"
            metadata = root / "seed_42.meta.json"
            model.write_bytes(b"frozen model")
            expected = {
                "version": 2,
                "seed": 42,
                "code_fingerprints": {
                    "runner_sha256": "f" * 64,
                    "dataset_loader_sha256": "d" * 64,
                    "metrics_sha256": "m" * 64,
                },
            }
            metadata.write_text(
                json.dumps(
                    {
                        **expected,
                        "code_fingerprints": CANONICAL_V2_GENERATOR_CODE_FINGERPRINTS,
                        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            _verify_reusable_model(model, metadata, expected)

            tampered = json.loads(metadata.read_text(encoding="utf-8"))
            tampered["code_fingerprints"]["metrics_sha256"] = "e" * 64
            metadata.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown producing code fingerprints"):
                _verify_reusable_model(model, metadata, expected)

    def test_payload_separates_benchmark_relevance_and_actual_event(self) -> None:
        items, history, scores = _fixture()
        payload, _, _ = build_browser_payload(
            items,
            history,
            scores,
            split="test",
            generated_at="2026-09-08",
        )
        case = payload["cases"][0]
        self.assertEqual(case["user_id"], "7")
        self.assertEqual(case["history_summary"]["count"], 1)
        first = next(row for row in case["candidates"] if row["item_id"] == 1)
        self.assertTrue(first["outcomes"]["actual_tc_event"])
        self.assertEqual(first["outcomes"]["benchmark_relevance"], 0.25)
        self.assertEqual(case["metrics"]["actual_tc"]["qwen_l0"]["first_rank"], 1)
        self.assertEqual(case["metrics"]["actual_tc"]["catboost"]["first_rank"], 2)
        self.assertNotIn("position_baseline", first["ranks"])
        self.assertNotIn(
            "position_baseline", payload["summary"]["graded_contacts_daily"]["values"]
        )

    def test_relevance_contract_mismatch_is_rejected(self) -> None:
        items, history, scores = _fixture()
        scores.loc[scores["item_id"].eq(1), "evaluation_relevance"] = 0.75
        with self.assertRaisesRegex(ValueError, "relevance"):
            build_browser_payload(items, history, scores)

    def test_insight_prompts_exclude_post_exposure_field_names(self) -> None:
        items, history, scores = _fixture()
        payload, _, _ = build_browser_payload(items, history, scores)
        requests = build_insight_requests(payload, top_k_per_ranking=1)
        self.assertTrue(requests)
        for request in requests:
            for forbidden in POST_EXPOSURE_FIELDS:
                self.assertNotIn(forbidden, request["prompt"])
        user_request = next(row for row in requests if row["kind"] == "user")
        self.assertIn("raw_category", user_request["prompt"])
        self.assertIn("Временное отношение", user_request["prompt"])

    def test_insight_requests_carry_structured_evidence_for_validation(self) -> None:
        items, history, scores = _fixture()
        payload, _, _ = build_browser_payload(items, history, scores)
        requests = build_insight_requests(payload, top_k_per_ranking=1)

        user_request = next(row for row in requests if row["kind"] == "user")
        self.assertEqual(user_request["evidence"]["history_summary"]["count"], 1)
        self.assertEqual(len(user_request["evidence"]["history"]), 1)
        self.assertEqual(
            user_request["evidence"]["history"][0]["raw_category_level_1"],
            "Raw category",
        )

        item_request = next(row for row in requests if row["kind"] == "item")
        self.assertTrue(item_request["evidence"]["title"].startswith("Car "))
        self.assertEqual(item_request["evidence"]["attributes"]["brand"], "Brand")
        self.assertTrue(item_request["evidence"]["description"].startswith("Seller text "))

    def test_item_request_marks_missing_history_metadata(self) -> None:
        items, history, scores = _fixture()
        payload, _, _ = build_browser_payload(items, history, scores)
        rows = build_item_request_rows(payload, featured_only=True)
        history_row = next(row for row in rows if row["item_id"] == 99)
        self.assertIn("history_contact", history_row["roles"])
        self.assertTrue(history_row["need_title"])
        self.assertTrue(history_row["need_description"])
        self.assertTrue(history_row["need_image"])

    def test_overlay_rejects_duplicate_item_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "overlay.csv"
            path.write_text("item_id,title\n1,A\n1,B\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_item_overlay(path)

    def test_relative_overlay_image_is_resolved_and_copied_portably(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "images" / "one.png"
            _write_png(image_path)
            overlay_path = root / "overlay.csv"
            overlay_path.write_text("item_id,image_path\n1,images/one.png\n", encoding="utf-8")

            overlay = load_item_overlay(overlay_path)
            self.assertEqual(Path(overlay.iloc[0]["image_path"]), image_path.resolve())
            output = root / "browser"
            copied = copy_overlay_images(
                overlay, output, {1}, allowed_source_root=overlay_path.parent
            )
            self.assertEqual(copied.iloc[0]["image_path"], "assets/items/1.png")
            self.assertEqual(
                (output / "assets/items/1.png").read_bytes(), image_path.read_bytes()
            )

    def test_overlay_rejects_absolute_traversal_and_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root.parent / f"{root.name}-outside.png"
            _write_png(outside)
            try:
                absolute = root / "absolute.csv"
                absolute.write_text(f"item_id,image_path\n1,{outside}\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "overlay directory"):
                    load_item_overlay(absolute)

                traversal = root / "traversal.csv"
                traversal.write_text(
                    f"item_id,image_path\n1,../{outside.name}\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "overlay directory"):
                    load_item_overlay(traversal)

                link = root / "escaped.png"
                link.symlink_to(outside)
                symlink_manifest = root / "symlink.csv"
                symlink_manifest.write_text(
                    "item_id,image_path\n1,escaped.png\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "overlay directory"):
                    load_item_overlay(symlink_manifest)
            finally:
                outside.unlink(missing_ok=True)

    def test_overlay_copy_rejects_non_image_and_oversized_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "fake.jpg"
            fake.write_bytes(b"not an image")
            overlay_path = root / "overlay.csv"
            overlay_path.write_text("item_id,image_path\n1,fake.jpg\n", encoding="utf-8")
            overlay = load_item_overlay(overlay_path)
            with self.assertRaisesRegex(ValueError, "decodable image"):
                copy_overlay_images(
                    overlay, root / "browser", {1}, allowed_source_root=root
                )

            oversized = root / "oversized.png"
            with oversized.open("wb") as handle:
                handle.truncate(MAX_OVERLAY_IMAGE_BYTES + 1)
            overlay_path.write_text(
                "item_id,image_path\n1,oversized.png\n", encoding="utf-8"
            )
            overlay = load_item_overlay(overlay_path)
            with self.assertRaisesRegex(ValueError, "size must be"):
                copy_overlay_images(
                    overlay, root / "browser", {1}, allowed_source_root=root
                )

    def test_overlay_copy_rejects_truncated_jpeg_and_gif_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for suffix, image_format in (("jpg", "JPEG"), ("gif", "GIF")):
                with self.subTest(image_format=image_format):
                    image_path = root / f"truncated.{suffix}"
                    _write_image(image_path, image_format)
                    image_path.write_bytes(image_path.read_bytes()[:-1])
                    overlay = pd.DataFrame(
                        [{"item_id": 1, "image_path": str(image_path.resolve())}]
                    )
                    with self.assertRaisesRegex(ValueError, "fully decodable"):
                        copy_overlay_images(
                            overlay,
                            root / f"browser-{suffix}",
                            {1},
                            allowed_source_root=root,
                        )

    def test_overlay_copy_rejects_aggregate_animation_pixel_bomb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "multi.gif"
            frames = [
                Image.new("RGB", (8, 8), color=(index * 60, 10, 20))
                for index in range(3)
            ]
            frames[0].save(
                image_path,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=10,
                loop=0,
            )
            overlay = pd.DataFrame(
                [{"item_id": 1, "image_path": str(image_path.resolve())}]
            )
            with patch(
                "avito_case_browser.core.MAX_OVERLAY_IMAGE_TOTAL_PIXELS", 128
            ):
                with self.assertRaisesRegex(ValueError, "decoded animation is too large"):
                    copy_overlay_images(
                        overlay,
                        root / "browser",
                        {1},
                        allowed_source_root=root,
                    )

    def test_overlay_copy_rejects_destination_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "source.png"
            _write_png(image_path)
            output = root / "browser"
            destination = output / "assets" / "items" / "1.png"
            destination.parent.mkdir(parents=True)
            victim = root / "victim.txt"
            victim.write_text("keep me", encoding="utf-8")
            destination.symlink_to(victim)
            overlay = pd.DataFrame(
                [{"item_id": 1, "image_path": str(image_path.resolve())}]
            )

            with self.assertRaisesRegex(ValueError, "destination must not be a symlink"):
                copy_overlay_images(overlay, output, {1}, allowed_source_root=root)
            self.assertEqual(victim.read_text(encoding="utf-8"), "keep me")
            self.assertTrue(destination.is_symlink())

    def test_overlay_copy_rejects_symlinked_asset_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "source.png"
            _write_png(image_path)
            output = root / "browser"
            outside = root / "outside"
            outside.mkdir()
            (output / "assets").mkdir(parents=True)
            (output / "assets" / "items").symlink_to(outside, target_is_directory=True)
            overlay = pd.DataFrame(
                [{"item_id": 1, "image_path": str(image_path.resolve())}]
            )

            with self.assertRaisesRegex(ValueError, "asset directory must not be a symlink"):
                copy_overlay_images(overlay, output, {1}, allowed_source_root=root)
            self.assertEqual(list(outside.iterdir()), [])

    def test_extract_json_object_handles_fences_and_braces_in_string(self) -> None:
        parsed = extract_json_object('prefix ```json\n{"summary":"x } y","confidence":"low"}\n``` suffix')
        self.assertEqual(parsed, {"summary": "x } y", "confidence": "low"})
        recovered = extract_json_object(
            'broken {"summary":"unfinished" then ```json\n'
            '{"summary":"recovered","confidence":"high"}\n```'
        )
        self.assertEqual(recovered, {"summary": "recovered", "confidence": "high"})
        self.assertIsNone(extract_json_object("no object"))

    def test_html_payload_is_base64_encoded(self) -> None:
        payload = {"dangerous": "</script><script>alert(1)</script>", "ru": "тест"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.html"
            output = root / "index.html"
            template.write_text(
                '<script type="text/plain">__AVITO_BROWSER_PAYLOAD_BASE64__</script>',
                encoding="utf-8",
            )
            render_browser_html(template, output, payload)
            rendered = output.read_text(encoding="utf-8")
            self.assertNotIn("alert(1)", rendered)
            encoded = rendered.split(">", 1)[1].split("<", 1)[0]
            self.assertEqual(json.loads(base64.b64decode(encoded)), payload)

    def test_browser_javascript_does_not_use_invalid_at_sign_dot_properties(self) -> None:
        template = (ROOT / "apps/avito_case_browser/template.html").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"\.[A-Za-z_$][\w$]*@", template))

    def test_browser_empty_search_has_no_unfiltered_case_fallback(self) -> None:
        template = (ROOT / "apps/avito_case_browser/template.html").read_text(encoding="utf-8")
        selected_case = template.split("function selectedCase()", 1)[1].split(
            "function metricCard", 1
        )[0]
        self.assertNotIn("DATA.cases[0]", selected_case)
        self.assertIn("rows[0] || null", selected_case)
        self.assertIn("function renderEmptyCase()", template)
        self.assertIn("if (!c) { renderEmptyCase(); return; }", template)

    def test_incomplete_insights_never_have_done_status(self) -> None:
        self.assertEqual(_insight_build_status(72, 71), "needs_insights")
        self.assertEqual(_insight_build_status(72, 0), "needs_insights")
        self.assertEqual(_insight_build_status(72, 72), "done")
        with tempfile.TemporaryDirectory() as tmp:
            stale = Path(tmp) / "share_bundle.zip"
            stale.write_bytes(b"stale final artifact")
            self.assertFalse(_bundle_should_be_written("needs_insights", stale))
            self.assertFalse(stale.exists())
            stale.write_bytes(b"stale final artifact")
            self.assertTrue(_bundle_should_be_written("done", stale))
            self.assertFalse(stale.exists())

    def test_share_bundle_includes_portable_overlay_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            browser = root / "browser" / "index.html"
            asset = browser.parent / "assets" / "items" / "1.png"
            stale_asset = browser.parent / "assets" / "items" / "stale.jpg"
            metrics = browser.parent / "metrics.json"
            manifest = root / "item_request_featured.csv"
            all_manifest = root / "item_request_all_test.csv"
            overlay = root / "item_overlay_template.csv"
            bundle = root / "bundle.zip"
            asset.parent.mkdir(parents=True)
            browser.write_text("browser", encoding="utf-8")
            _write_png(asset)
            stale_asset.write_bytes(b"stale")
            metrics.write_text("{}", encoding="utf-8")
            manifest.write_text("item_id\n1\n", encoding="utf-8")
            all_manifest.write_text("item_id\n1\n", encoding="utf-8")
            overlay.write_text("item_id\n1\n", encoding="utf-8")

            create_share_bundle(
                bundle,
                browser,
                metrics,
                manifest,
                overlay,
                all_manifest,
                [asset],
            )
            with zipfile.ZipFile(bundle) as archive:
                self.assertIn("assets/items/1.png", archive.namelist())
                self.assertNotIn("assets/items/stale.jpg", archive.namelist())
                self.assertIn("item_request_all_test.csv", archive.namelist())
                self.assertEqual(archive.read("assets/items/1.png"), asset.read_bytes())

    def test_share_bundle_rejects_asset_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            browser_root = root / "browser"
            asset_dir = browser_root / "assets" / "items"
            asset_dir.mkdir(parents=True)
            outside = root / "outside.png"
            _write_png(outside)
            escaped = asset_dir / "1.png"
            escaped.symlink_to(outside)
            browser = browser_root / "index.html"
            metrics = browser_root / "metrics.json"
            featured = root / "featured.csv"
            overlay = root / "overlay.csv"
            browser.write_text("browser", encoding="utf-8")
            metrics.write_text("{}", encoding="utf-8")
            featured.write_text("item_id\n1\n", encoding="utf-8")
            overlay.write_text("item_id\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the allowed root"):
                create_share_bundle(
                    root / "bundle.zip",
                    browser,
                    metrics,
                    featured,
                    overlay,
                    referenced_assets=[escaped],
                )

    def test_insight_validator_rejects_wrong_language_and_sensitive_profile(self) -> None:
        chinese_item = {
            "summary": "这是一辆汽车",
            "observed_facts": ["状态良好"],
            "seller_claims": ["卖家说状态良好"],
            "risks_to_verify": ["检查"],
            "suitable_for": ["买家"],
            "limitations": ["未知"],
            "confidence": "medium",
        }
        parsed, error = _validate_parsed("item", chinese_item)
        self.assertIsNone(parsed)
        self.assertIn("CJK", str(error))

        unsafe_user = {
            "summary": "Пользователь приобрёл автомобиль после контакта с объявлением.",
            "observed_signals": ["Покупка автомобиля"],
            "hypotheses": ["Пользователь может быть мужчиной"],
            "limitations": ["Данных недостаточно для уверенного вывода"],
            "confidence": "low",
        }
        parsed, error = _validate_parsed("user", unsafe_user)
        self.assertIsNone(parsed)
        self.assertTrue(
            "event/frequency" in str(error)
            or "purchase" in str(error)
            or "sensitive" in str(error)
        )

        safe_user = {
            "summary": "В истории контактов повторяется категория грузовой техники.",
            "observed_signals": ["Три контакта относятся к одной товарной категории"],
            "hypotheses": ["Может сохраняться товарный интерес к грузовой технике"],
            "limitations": ["Временная связь с текущей выдачей неизвестна"],
            "confidence": "low",
        }
        parsed, error = _validate_parsed("user", safe_user)
        self.assertIsNotNone(parsed)
        self.assertIsNone(error)

        non_modal_user = dict(safe_user, hypotheses=["Интерес к грузовой технике"])
        parsed, error = _validate_parsed("user", non_modal_user)
        self.assertIsNone(parsed)
        self.assertIn("может", str(error))

        mixed_language_item = {
            "summary": "В экспорте указан автомобиль с пробегом.",
            "observed_facts": ["В экспорте указан 2020 год"],
            "seller_claims": [],
            "risks_to_verify": ["service history accuracy"],
            "suitable_for": ["Городские поездки"],
            "limitations": ["Нет данных о состоянии"],
            "confidence": "low",
        }
        parsed, error = _validate_parsed("item", mixed_language_item)
        self.assertIsNone(parsed)
        self.assertIn("non-Russian", str(error))

    def test_request_loader_requires_consistent_evidence(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "requests.jsonl"
            prompt = "тест"
            path.write_text(
                json.dumps(
                    {
                        "request_id": "user:1",
                        "kind": "user",
                        "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "evidence": {"history": [], "history_summary": {"count": 1}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                _load_requests(path)

    def test_item_claim_must_be_exact_description_span(self) -> None:
        request = {
            "request_id": "item:9999999001",
            "kind": "item",
            "entity_id": "9999999001",
            "prompt_version": 3,
            "evidence": {
                "title": "Объявление 9999999001",
                "description": "Разбор",
                "attributes": {"body": "минивэн", "fuel": "бензин"},
            },
        }
        parsed = {
            "summary": "В экспортированных структурированных данных указано автомобильное объявление.",
            "observed_facts": ["В экспорте указан автомобиль"],
            "seller_claims": ["Продавец заявляет: «минивэн»"],
            "risks_to_verify": ["Проверить фактическое состояние автомобиля"],
            "suitable_for": ["Может подойти после независимой проверки"],
            "limitations": ["Состояние автомобиля по экспорту неизвестно"],
            "confidence": "low",
        }
        normalised, error, _ = validate_and_normalise_insight(request, parsed)
        self.assertIsNone(normalised)
        self.assertIn("verbatim", str(error))

        parsed["seller_claims"] = ["Продавец заявляет: «Разбор»"]
        normalised, error, fields = validate_and_normalise_insight(request, parsed)
        self.assertIsNotNone(normalised)
        self.assertIsNone(error)
        self.assertEqual(normalised["seller_claims"], ["Продавец заявляет: «Разбор»"])
        self.assertEqual(
            fields,
            [
                "confidence",
                "limitations",
                "observed_facts",
                "risks_to_verify",
                "suitable_for",
                "summary",
            ],
        )

    def test_every_displayed_non_claim_field_is_evidence_normalised(self) -> None:
        item_request = {
            "request_id": "item:1",
            "kind": "item",
            "entity_id": "1",
            "prompt_version": 3,
            "evidence": {
                "title": "BMW X7",
                "description": "Один владелец",
                "attributes": {"brand": "BMW", "model": "X7", "year": "2021"},
            },
        }
        raw_item = {
            "summary": "В экспортированных данных указано автомобильное объявление.",
            "observed_facts": ["В экспорте указана марка автомобиля"],
            "seller_claims": ["Продавец заявляет: «Один владелец»"],
            "risks_to_verify": ["Не используется для коммерческих целей"],
            "suitable_for": ["Подходит обеспеченному владельцу бизнеса"],
            "limitations": ["Продавец гарантирует отсутствие вложений"],
            "confidence": "high",
        }
        item, error, fields = validate_and_normalise_insight(item_request, raw_item)
        self.assertIsNone(error)
        self.assertEqual(item["seller_claims"], raw_item["seller_claims"])
        self.assertEqual(item["suitable_for"], [])
        self.assertEqual(item["confidence"], "low")
        forbidden = " ".join(
            str(item[key]) for key in fields
        ).lower()
        self.assertNotIn("коммерчес", forbidden)
        self.assertNotIn("обеспеч", forbidden)
        self.assertNotIn("гарантир", forbidden)

        user_request = {
            "request_id": "user:7",
            "kind": "user",
            "entity_id": "7",
            "prompt_version": 3,
            "evidence": {
                "history": [
                    {
                        "contact_date": "2026-05-01",
                        "raw_category_level_1": "Другие грузовики",
                        "raw_category_level_2": "Разное",
                    }
                ],
                "history_summary": {
                    "count": 1,
                    "date_from": "2026-05-01",
                    "date_to": "2026-05-01",
                },
            },
        }
        raw_user = {
            "summary": "В истории контактов присутствует одна экспортированная запись.",
            "observed_signals": ["В истории контактов есть одна запись"],
            "hypotheses": ["Может сохраняться товарный интерес к грузовой технике"],
            "limitations": ["Пользователь, вероятно, мужчина с высоким доходом"],
            "confidence": "high",
        }
        user, error, fields = validate_and_normalise_insight(user_request, raw_user)
        self.assertIsNone(error)
        self.assertEqual(user["confidence"], "low")
        self.assertNotIn("мужчин", " ".join(map(str, user.values())).lower())
        self.assertNotIn("доход", " ".join(map(str, user.values())).lower())
        self.assertEqual(
            fields,
            ["confidence", "hypotheses", "limitations", "observed_signals", "summary"],
        )

    def test_user_hypotheses_are_derived_only_from_raw_categories(self) -> None:
        request = {
            "request_id": "user:7",
            "kind": "user",
            "entity_id": "7",
            "prompt_version": 3,
            "evidence": {
                "history": [
                    {
                        "contact_date": "2026-05-01",
                        "raw_category_level_1": "Другие грузовики",
                        "raw_category_level_2": "Разное",
                    }
                ],
                "history_summary": {
                    "count": 1,
                    "date_from": "2026-05-01",
                    "date_to": "2026-05-01",
                },
            },
        }
        parsed = {
            "summary": "В истории контактов присутствует одна экспортированная запись.",
            "observed_signals": ["В истории контактов есть одна запись"],
            "hypotheses": ["Может предпочитать дорогие автомобили премиального класса"],
            "limitations": ["Временная связь с текущей выдачей неизвестна"],
            "confidence": "low",
        }
        normalised, error, _ = validate_and_normalise_insight(request, parsed)
        self.assertIsNone(error)
        self.assertEqual(
            normalised["hypotheses"],
            ["Может сохраняться товарный интерес к raw category «Другие грузовики»."],
        )
        self.assertNotIn("премиаль", " ".join(normalised["hypotheses"]).lower())

    def test_cached_insight_contract_rejects_tampering_and_wrong_revision(self) -> None:
        prompt = "Проанализируй объявление по заданной схеме."
        request = {
            "request_id": "item:1",
            "kind": "item",
            "entity_id": "1",
            "prompt_version": 3,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "evidence": {
                "title": "Автомобиль 1",
                "description": "Один владелец",
                "attributes": {"brand": "Brand", "model": "Model"},
            },
        }
        contract = default_insight_contract([request])
        raw = {
            "summary": "В экспортированных структурированных данных указано автомобильное объявление.",
            "observed_facts": ["В экспорте указана марка"],
            "seller_claims": ["Продавец заявляет: «Один владелец»"],
            "risks_to_verify": ["Проверить документальное подтверждение утверждения"],
            "suitable_for": ["Может подойти после независимой проверки"],
            "limitations": ["Фактическое состояние автомобиля неизвестно"],
            "confidence": "low",
        }
        parsed, error, fields = validate_and_normalise_insight(request, raw)
        self.assertIsNone(error)
        llm_parsed, error = _validate_parsed("item", raw)
        self.assertIsNone(error)
        raw_text = json.dumps(raw, ensure_ascii=False)
        memberships = [request["request_id"]]
        token_verifier = make_token_provenance_verifier(
            _DeterministicTestTokenizer(), contract
        )
        primary_provenance = dict(token_verifier(request, "primary"))
        attempt = {
            "pass": "primary",
            "pass_batch_index": 0,
            "configured_batch_size": 1,
            "actual_batch_size": 1,
            **primary_provenance,
            "max_new_tokens": contract["max_new_tokens"],
            "generation_seconds": 1.0,
            "status": "ok",
            "validation_error": None,
            "raw_text": raw_text,
            "raw_text_sha256": text_sha256(raw_text),
            "llm_parsed_sha256": parsed_insight_sha256(llm_parsed),
        }
        result = {
            "request_id": request["request_id"],
            "kind": request["kind"],
            "entity_id": request["entity_id"],
            "prompt_version": request["prompt_version"],
            "prompt_sha256": request["prompt_sha256"],
            "signature": contract["signature"],
            "model": contract["model"],
            "model_revision": contract["model_revision"],
            "device": contract["device"],
            "device_type": contract["device_type"],
            "compute_dtype": contract["compute_dtype"],
            "torch_version": contract["torch_version"],
            "cuda_runtime_version": contract["cuda_runtime_version"],
            "transformers_version": contract["transformers_version"],
            "do_sample": False,
            "generation_contract": dict(contract),
            "generation_attempts": [attempt],
            "validation_version": INSIGHT_VALIDATION_VERSION,
            "status": "ok",
            "fallback_applied": False,
            "llm_output_accepted": True,
            "parsed": parsed,
            "parsed_sha256": parsed_insight_sha256(parsed),
            "llm_parsed_sha256": parsed_insight_sha256(llm_parsed),
            "evidence_normalized_fields": fields,
            "generation_pass": "primary",
            "effective_prompt_sha256": text_sha256(prompt),
            "max_new_tokens_used": contract["max_new_tokens"],
            "raw_text": raw_text,
            "raw_text_sha256": text_sha256(raw_text),
        }
        safe, error = validate_cached_insight(
            request,
            result,
            contract=contract,
            validation_version=INSIGHT_VALIDATION_VERSION,
            token_provenance_verifier=token_verifier,
        )
        self.assertIsNotNone(safe)
        self.assertIsNone(error)

        tampered = {**result, "parsed": {**parsed, "risks_to_verify": ["Подменённый вывод"]}}
        safe, error = validate_cached_insight(
            request,
            tampered,
            contract=contract,
            validation_version=INSIGHT_VALIDATION_VERSION,
            token_provenance_verifier=token_verifier,
        )
        self.assertIsNone(safe)
        self.assertTrue(
            "evidence-normalised" in str(error) or "parsed_sha256" in str(error)
        )

        tampered_with_new_hash = {
            **result,
            "parsed": {**parsed, "limitations": ["Подменённый вывод о владельце"]},
        }
        tampered_with_new_hash["parsed_sha256"] = parsed_insight_sha256(
            tampered_with_new_hash["parsed"]
        )
        safe, error = validate_cached_insight(
            request,
            tampered_with_new_hash,
            contract=contract,
            validation_version=INSIGHT_VALIDATION_VERSION,
            token_provenance_verifier=token_verifier,
        )
        self.assertIsNone(safe)
        self.assertIn("evidence-normalised", str(error))

        wrong_revision = {**result, "model_revision": "wrong"}
        safe, error = validate_cached_insight(
            request,
            wrong_revision,
            contract=contract,
            validation_version=INSIGHT_VALIDATION_VERSION,
            token_provenance_verifier=token_verifier,
        )
        self.assertIsNone(safe)
        self.assertIn("model_revision", str(error))

        provenance_tampers = {
            "effective_prompt_sha256": "wrong",
            "transformers_version": "wrong",
            "max_new_tokens_used": contract["max_new_tokens"] + 1,
            "generation_pass": "unknown",
            "raw_text_sha256": "wrong",
        }
        for field, value in provenance_tampers.items():
            with self.subTest(field=field):
                safe, _ = validate_cached_insight(
                    request,
                    {**result, field: value},
                    contract=contract,
                    validation_version=INSIGHT_VALIDATION_VERSION,
                    token_provenance_verifier=token_verifier,
                )
                self.assertIsNone(safe)

        tampered_attempt = {
            **result,
            "generation_attempts": [
                {**attempt, "input_ids_sha256": "9" * 64}
            ],
        }
        safe, error = validate_cached_insight(
            request,
            tampered_attempt,
            contract=contract,
            validation_version=INSIGHT_VALIDATION_VERSION,
            token_provenance_verifier=token_verifier,
        )
        self.assertIsNone(safe)
        self.assertIn("padded batch binding", str(error))

        coordinated = json.loads(json.dumps(result))
        coordinated_attempt = coordinated["generation_attempts"][0]
        coordinated_attempt["chat_template_sha256"] = "0" * 64
        coordinated_attempt["input_ids_sha256"] = "7" * 64
        coordinated_attempt["attention_mask_sha256"] = "8" * 64
        coordinated_attempt["input_ids_shape"] = [1, 1]
        coordinated_attempt["attention_mask_shape"] = [1, 1]
        coordinated_binding = {
            "ordered_request_ids": memberships,
            "input_ids_sha256": "7" * 64,
            "attention_mask_sha256": "8" * 64,
            "input_ids_shape": [1, 1],
            "attention_mask_shape": [1, 1],
        }
        coordinated_attempt["padded_batch_sha256"] = text_sha256(
            json.dumps(coordinated_binding, sort_keys=True, separators=(",", ":"))
        )
        safe, error = validate_cached_insight(
            request,
            coordinated,
            contract=contract,
            validation_version=INSIGHT_VALIDATION_VERSION,
            token_provenance_verifier=token_verifier,
        )
        self.assertIsNone(safe)
        self.assertIn("pinned-tokenizer recomputation", str(error))

        bad_text = "ответ без JSON"
        primary_bad = {
            **attempt,
            "status": "parse_error",
            "validation_error": "no balanced JSON object",
            "raw_text": bad_text,
            "raw_text_sha256": text_sha256(bad_text),
            "llm_parsed_sha256": None,
        }
        retry_provenance = dict(token_verifier(request, "retry"))
        retry_attempt = {
            **attempt,
            "pass": "retry",
            **retry_provenance,
            "max_new_tokens": contract["retry_max_new_tokens"],
        }
        retried = {
            **result,
            "generation_attempts": [primary_bad, retry_attempt],
            "generation_pass": "retry",
            "effective_prompt_sha256": retry_attempt["effective_prompt_sha256"],
            "max_new_tokens_used": contract["retry_max_new_tokens"],
        }
        safe, error = validate_cached_insight(
            request,
            retried,
            contract=contract,
            validation_version=INSIGHT_VALIDATION_VERSION,
            token_provenance_verifier=token_verifier,
        )
        self.assertIsNotNone(safe)
        self.assertIsNone(error)
        safe, error = validate_cached_insight(
            request,
            {**retried, "generation_attempts": [retry_attempt]},
            contract=contract,
            validation_version=INSIGHT_VALIDATION_VERSION,
            token_provenance_verifier=token_verifier,
        )
        self.assertIsNone(safe)
        self.assertIn("complete primary/retry audit", str(error))

        partial = {
            **result,
            "status": "parse_error",
            "parsed": None,
            "parsed_sha256": None,
            "validation_error": primary_bad["validation_error"],
            "llm_output_accepted": False,
            "generation_attempts": [primary_bad],
            "raw_text": bad_text,
            "raw_text_sha256": text_sha256(bad_text),
            "llm_parsed_sha256": None,
            "evidence_normalized_fields": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "partial.jsonl"
            cache_path.write_text(
                json.dumps(partial, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            resumed = _load_cache(
                cache_path,
                {request["request_id"]: request},
                contract,
                token_verifier,
            )
        self.assertEqual(resumed[request["request_id"]]["status"], "parse_error")

    def test_cache_signature_covers_revision_retry_budget_quantization_and_batch(self) -> None:
        base = Namespace(
            model="model",
            model_revision="rev-a",
            max_input_tokens=4096,
            max_new_tokens=320,
            retry_max_new_tokens=640,
            load_in_4bit=False,
            device="cuda:0",
            compute_dtype="float16",
            torch_version="2.7.1",
            cuda_runtime_version="12.6",
            transformers_version="4.52.4",
            batch_size=1,
        )
        signature = _cache_signature(base)
        self.assertNotEqual(
            signature,
            _cache_signature(Namespace(**{**vars(base), "model_revision": "rev-b"})),
        )
        self.assertNotEqual(
            signature,
            _cache_signature(Namespace(**{**vars(base), "retry_max_new_tokens": 768})),
        )
        self.assertNotEqual(
            signature,
            _cache_signature(Namespace(**{**vars(base), "load_in_4bit": True})),
        )
        self.assertNotEqual(
            signature,
            _cache_signature(Namespace(**{**vars(base), "device": "cuda:1"})),
        )
        self.assertNotEqual(
            signature,
            _cache_signature(Namespace(**{**vars(base), "compute_dtype": "bfloat16"})),
        )
        self.assertNotEqual(
            signature,
            _cache_signature(
                Namespace(**{**vars(base), "transformers_version": "4.53.0"})
            ),
        )
        self.assertNotEqual(
            signature,
            _cache_signature(Namespace(**{**vars(base), "batch_size": 64})),
        )

    def test_builder_accepts_only_current_canonical_generator_contract(self) -> None:
        prompt = "Проверяемый prompt"
        request = {
            "request_id": "item:1",
            "prompt_version": 3,
            "prompt_sha256": text_sha256(prompt),
        }
        defaults = default_insight_contract([request])
        generator_args = Namespace(
            model=defaults["model"],
            model_revision=defaults["model_revision"],
            max_input_tokens=defaults["max_input_tokens"],
            max_new_tokens=defaults["max_new_tokens"],
            retry_max_new_tokens=defaults["retry_max_new_tokens"],
            load_in_4bit=defaults["load_in_4bit"],
            device=defaults["device"],
            compute_dtype=defaults["compute_dtype"],
            batch_size=1,
            torch_version=defaults["torch_version"],
            cuda_runtime_version=defaults["cuda_runtime_version"],
            transformers_version=defaults["transformers_version"],
            local_files_only=True,
        )
        environment = {
            "accelerator": {
                "requested_device": "cuda:0",
                "logical_device_index": 0,
                "cuda_visible_devices": "2",
                "physical_device_selector": "2",
                "name": "NVIDIA A100",
                "compute_capability": "8.0",
                "total_memory_bytes": 1,
                "driver_version": "570.0",
                "nvidia_smi_uuid": "GPU-test",
            },
            "determinism": {
                "do_sample": False,
                "torch_deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "cublas_workspace_config": ":4096:8",
                "python_hash_seed": "0",
            },
        }
        contract = _generation_contract(
            generator_args,
            [request],
            environment=environment,
            tokenizer_provenance=defaults["tokenizer"],
        )
        builder_args = Namespace(
            insight_model=generator_args.model,
            insight_model_revision=generator_args.model_revision,
            insight_max_input_tokens=generator_args.max_input_tokens,
            insight_max_new_tokens=generator_args.max_new_tokens,
            insight_retry_max_new_tokens=generator_args.retry_max_new_tokens,
            insight_load_in_4bit=generator_args.load_in_4bit,
            insight_device=generator_args.device,
            insight_compute_dtype=generator_args.compute_dtype,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contract.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            loaded, source, error = _load_insight_contract(
                path, [request], builder_args
            )
            self.assertEqual(loaded, contract)
            self.assertEqual(source, "generator_contract")
            self.assertIsNone(error)

            tampered = json.loads(json.dumps(contract))
            tampered["determinism"]["torch_deterministic_algorithms"] = False
            from avito_case_browser.insights import insight_cache_signature

            tampered["signature"] = insight_cache_signature(tampered)
            path.write_text(json.dumps(tampered), encoding="utf-8")
            _, source, error = _load_insight_contract(path, [request], builder_args)
            self.assertEqual(source, "pending")
            self.assertIn("determinism flags", str(error))

    def test_guarded_fallback_keeps_only_structured_item_evidence(self) -> None:
        request = {
            "request_id": "item:1",
            "kind": "item",
            "entity_id": "1",
            "prompt_version": 3,
            "prompt_sha256": "abc",
            "evidence": {
                "title": "BMW X7",
                "description": "Продавец обещает идеальное состояние",
                "attributes": {
                    "brand": "BMW",
                    "model": "X7",
                    "year": "2021",
                    "mileage_km": 100000,
                    "price": 5000000,
                },
            },
        }
        rejected = {
            "status": "parse_error",
            "validation_error": "seller claim",
            "raw_text": "невалидный ответ об идеальном состоянии",
        }
        result = _guarded_fallback(request, rejected)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["fallback_applied"])
        self.assertFalse(result["llm_output_accepted"])
        self.assertEqual(result["parsed"]["seller_claims"], [])
        self.assertNotIn("идеаль", result["parsed"]["summary"].lower())


if __name__ == "__main__":
    unittest.main()
