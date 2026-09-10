#!/usr/bin/env python3
"""Build the self-contained Avito ranking case browser and sharing manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from avito_case_browser import (  # noqa: E402
    INSIGHT_PROMPT_VERSION,
    INSIGHT_VALIDATION_VERSION,
    build_browser_payload,
    build_insight_requests,
    build_item_request_rows,
    load_insight_results,
    load_item_overlay,
    render_browser_html,
)
from avito_case_browser.insights import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_COMPUTE_DTYPE,
    DEFAULT_INSIGHT_DEVICE,
    DEFAULT_INSIGHT_MODEL,
    DEFAULT_INSIGHT_MODEL_REVISION,
    DEFAULT_LOAD_IN_4BIT,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_RETRY_MAX_NEW_TOKENS,
    GENERATION_CONTRACT_VERSION,
    default_insight_contract,
    insight_cache_signature,
    load_pinned_tokenizer,
    make_token_provenance_verifier,
    request_manifest_sha256,
    validate_cached_insight,
    validate_tokenizer_provenance_contract,
)
from avito_case_browser.core import (  # noqa: E402
    copy_overlay_images,
    create_share_bundle,
    write_csv,
    write_json,
    write_jsonl,
)


def _reported_metric_audit(summary: dict, catboost_metrics: Path, llm_metrics: Path) -> dict:
    if not catboost_metrics.exists() or not llm_metrics.exists():
        return {"available": False}
    catboost = json.loads(catboost_metrics.read_text(encoding="utf-8"))
    llm = json.loads(llm_metrics.read_text(encoding="utf-8"))
    recomputed = summary["graded_contacts_daily"]["values"]
    reported = {
        "catboost": float(catboost["catboost_ensemble_test"]["ndcg@10"]),
        "qwen_l0": float(llm["llm_pointwise"]["test"]["ndcg@10"]),
        "gated_blend": float(llm["pre_inference_ambiguity_gate"]["test"]["ndcg@10"]),
    }
    differences = {key: float(recomputed[key] - value) for key, value in reported.items()}
    max_abs_difference = max(map(abs, differences.values()), default=0.0)
    return {
        "available": True,
        "reported": reported,
        "browser_recomputed": {key: float(recomputed[key]) for key in reported},
        "difference": differences,
        "explanation": (
            "Browser and diagnostics use float64 relevance plus explicit item_id ascending "
            "score ties; metrics agree within numerical tolerance."
            if max_abs_difference <= 1e-8
            else "Browser and reported metrics disagree beyond numerical tolerance."
        ),
    }


def _overlay_template_rows(featured_rows: list[dict]) -> list[dict]:
    return [
        {
            "item_id": row["item_id"],
            "title": "",
            "description_short": "",
            "image_path": "",
            "image_url": "",
        }
        for row in featured_rows
    ]


def _referenced_local_assets(payload: dict, output_dir: Path) -> list[Path]:
    """Return only local asset files actually referenced by the current payload."""
    root = output_dir.resolve()
    assets: set[Path] = set()
    for case in payload.get("cases", []):
        records = [*case.get("candidates", []), *case.get("history", [])]
        for record in records:
            raw = record.get("image")
            if not isinstance(raw, str) or not raw or raw.startswith(("http://", "https://", "data:")):
                continue
            path = Path(raw)
            resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            if relative.parts and relative.parts[0] == "assets" and resolved.is_file():
                assets.add(resolved)
    return sorted(assets)


def _insight_build_status(request_count: int, accepted_count: int) -> str:
    """A shareable final bundle exists only after every current request validates."""
    return "done" if accepted_count == request_count else "needs_insights"


def _bundle_should_be_written(build_status: str, bundle_path: Path) -> bool:
    """Remove stale output before deciding whether a fresh final ZIP is allowed."""
    bundle_path.unlink(missing_ok=True)
    return build_status == "done"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_insight_contract(
    path: Path,
    requests: list[dict],
    args: argparse.Namespace,
) -> tuple[dict, str, str | None]:
    """Load a SHA-256-bound generator contract or create a non-cacheable pending one."""
    expected_policy = {
        "contract_version": GENERATION_CONTRACT_VERSION,
        "model": args.insight_model,
        "model_revision": args.insight_model_revision,
        "max_input_tokens": args.insight_max_input_tokens,
        "max_new_tokens": args.insight_max_new_tokens,
        "retry_max_new_tokens": args.insight_retry_max_new_tokens,
        "load_in_4bit": args.insight_load_in_4bit,
        "device": args.insight_device,
        "device_type": args.insight_device.split(":", 1)[0],
        "compute_dtype": args.insight_compute_dtype,
        "batch_size": DEFAULT_BATCH_SIZE,
        "request_count": len(requests),
        "request_manifest_sha256": request_manifest_sha256(requests),
        "do_sample": False,
        "local_files_only": True,
        "tokenization": {
            "use_fast": True,
            "padding": True,
            "padding_side": "left",
            "truncation": True,
            "truncation_side": "right",
            "add_generation_prompt": True,
        },
        "generation": {
            "do_sample": False,
            "use_cache": True,
            "skip_special_tokens": True,
        },
        "code_fingerprints": {
            "generator_sha256": _file_sha256(ROOT / "scripts/avito/generate_case_insights.py"),
            "insight_contract_sha256": _file_sha256(
                ROOT / "src/avito_case_browser/insights.py"
            ),
            "browser_core_sha256": _file_sha256(ROOT / "src/avito_case_browser/core.py"),
        },
    }
    error: str | None = None
    if path.exists():
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(candidate, dict):
                raise ValueError("top-level JSON must be an object")
            mismatches = [
                field
                for field, expected in expected_policy.items()
                if candidate.get(field) != expected
            ]
            if mismatches:
                raise ValueError(f"policy/request mismatch: {mismatches}")
            tokenizer_error = validate_tokenizer_provenance_contract(
                candidate.get("tokenizer"),
                model=args.insight_model,
                revision=args.insight_model_revision,
            )
            if tokenizer_error:
                raise ValueError(tokenizer_error)
            accelerator = candidate.get("accelerator", {})
            determinism = candidate.get("determinism", {})
            expected_determinism = {
                "do_sample": False,
                "torch_deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "cublas_workspace_config": ":4096:8",
                "python_hash_seed": "0",
            }
            if determinism != expected_determinism:
                raise ValueError("determinism flags are not canonical")
            if args.insight_device.startswith("cuda") and any(
                accelerator.get(field) in {None, "", "unavailable", "<unset>"}
                for field in (
                    "cuda_visible_devices",
                    "physical_device_selector",
                    "name",
                    "compute_capability",
                    "total_memory_bytes",
                    "driver_version",
                    "nvidia_smi_uuid",
                )
            ):
                raise ValueError("CUDA accelerator provenance is not materialized")
            if candidate.get("signature") != insight_cache_signature(candidate):
                raise ValueError("signature mismatch")
            return candidate, "generator_contract", None
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            error = f"ignored invalid/stale contract {path}: {exc}"
    else:
        error = f"generation contract is missing: {path}"

    # The placeholder has the current request identity but an explicit
    # non-materialized environment.  Its signature cannot collide with a real
    # generator run, so no old cache row can accidentally validate.
    pending_environment = {
        "accelerator": {
            "requested_device": args.insight_device,
            "logical_device_index": 0,
            "cuda_visible_devices": "not_materialized",
            "physical_device_selector": "not_materialized",
            "name": "not_materialized",
            "compute_capability": "not_materialized",
            "total_memory_bytes": None,
            "driver_version": "not_materialized",
            "nvidia_smi_uuid": "not_materialized",
        },
        "determinism": {
            "do_sample": False,
            "torch_deterministic_algorithms": False,
            "cudnn_deterministic": False,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cublas_workspace_config": "not_materialized",
            "python_hash_seed": "not_materialized",
        },
    }
    pending = default_insight_contract(requests, environment=pending_environment)
    pending.update(expected_policy)
    pending["signature"] = insight_cache_signature(pending)
    return pending, "pending", error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=ROOT / "items_with_attrs.parquet")
    parser.add_argument("--history", type=Path, default=ROOT / "users_with_history.parquet")
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "data/avito/benchmark/local_llm_pointwise_eval_scores.parquet",
    )
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument(
        "--item-overlay",
        type=Path,
        help="Optional CSV/Parquet/JSON(L) from Roma with item_id, text and image path/URL.",
    )
    parser.add_argument(
        "--insights",
        type=Path,
        default=ROOT / "data/avito/case_browser/insights.jsonl",
    )
    parser.add_argument(
        "--insight-contract",
        type=Path,
        help=(
            "Signed generation contract JSON; defaults to "
            "<work-dir>/generation_contract.json."
        ),
    )
    parser.add_argument("--insight-model", default=DEFAULT_INSIGHT_MODEL)
    parser.add_argument("--insight-model-revision", default=DEFAULT_INSIGHT_MODEL_REVISION)
    parser.add_argument("--insight-device", default=DEFAULT_INSIGHT_DEVICE)
    parser.add_argument(
        "--insight-compute-dtype",
        choices=("float16", "bfloat16", "float32"),
        default=DEFAULT_COMPUTE_DTYPE,
    )
    parser.add_argument(
        "--insight-max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS
    )
    parser.add_argument("--insight-max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--insight-retry-max-new-tokens",
        type=int,
        default=DEFAULT_RETRY_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--insight-load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LOAD_IN_4BIT,
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "apps/avito_case_browser/template.html",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports/avito_case_browser",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "data/avito/case_browser",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=ROOT / "results/current/metrics/avito_case_browser.json",
    )
    parser.add_argument(
        "--catboost-metrics",
        type=Path,
        default=ROOT / "results/current/metrics/avito_local_catboost_diagnostic.json",
    )
    parser.add_argument(
        "--llm-metrics",
        type=Path,
        default=ROOT / "results/current/metrics/avito_local_llm_diagnostic.json",
    )
    parser.add_argument("--generated-at", default=date.today().isoformat())
    parser.add_argument(
        "--copy-overlay-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy local overlay images into portable browser assets (default: true).",
    )
    args = parser.parse_args()
    insight_contract_path = args.insight_contract or args.work_dir / "generation_contract.json"

    items = pd.read_parquet(args.items)
    history = pd.read_parquet(args.history)
    scores = pd.read_parquet(args.scores)
    overlay = load_item_overlay(args.item_overlay)
    loaded_insights = load_insight_results(args.insights)

    source_paths = {
        "items": args.items,
        "history": args.history,
        "scores": args.scores,
    }
    if args.item_overlay:
        source_paths["item_overlay"] = args.item_overlay
    if args.insights.exists():
        source_paths["insights"] = args.insights
    if insight_contract_path.exists():
        source_paths["insight_contract"] = insight_contract_path

    # Build once without insights to derive the exact current prompt hashes.
    # This prevents stale cached explanations from surviving a prompt change.
    payload, _, _ = build_browser_payload(
        items,
        history,
        scores,
        split=args.split,
        overlay=overlay,
        insight_results={},
        source_paths=source_paths,
        generated_at=args.generated_at,
    )
    if args.copy_overlay_images and not overlay.empty:
        requested = {
            int(row["item_id"])
            for row in build_item_request_rows(payload, featured_only=False)
        }
        if args.item_overlay is None:
            raise AssertionError("Non-empty overlay must have a source manifest")
        overlay = copy_overlay_images(
            overlay,
            args.output_dir,
            requested,
            allowed_source_root=args.item_overlay.parent,
        )
        payload, _, _ = build_browser_payload(
            items,
            history,
            scores,
            split=args.split,
            overlay=overlay,
            insight_results={},
            source_paths=source_paths,
            generated_at=args.generated_at,
        )

    requests = build_insight_requests(payload)
    insight_contract, insight_contract_source, insight_contract_error = _load_insight_contract(
        insight_contract_path, requests, args
    )
    token_provenance_verifier = None
    if insight_contract_source == "generator_contract":
        try:
            tokenizer, actual_tokenizer_provenance = load_pinned_tokenizer(
                str(insight_contract["model"]),
                str(insight_contract["model_revision"]),
                local_files_only=True,
            )
            if actual_tokenizer_provenance != insight_contract.get("tokenizer"):
                raise ValueError("local pinned-tokenizer artifacts differ from the contract")
            token_provenance_verifier = make_token_provenance_verifier(
                tokenizer, insight_contract
            )
        except Exception as exc:  # noqa: BLE001 - unavailable verifier rejects every cache row
            insight_contract_error = f"token provenance verifier unavailable: {exc}"
    insights: dict[str, dict] = {}
    insight_cache_rejections: Counter[str] = Counter()
    for request in requests:
        request_id = request["request_id"]
        result = loaded_insights.get(request_id)
        if result is None:
            insight_cache_rejections["missing"] += 1
            continue
        safe_result, error = validate_cached_insight(
            request,
            result,
            contract=insight_contract,
            validation_version=INSIGHT_VALIDATION_VERSION,
            token_provenance_verifier=token_provenance_verifier,
        )
        if safe_result is None:
            insight_cache_rejections[str(error)] += 1
            continue
        insights[request_id] = safe_result
    payload, _, _ = build_browser_payload(
        items,
        history,
        scores,
        split=args.split,
        overlay=overlay,
        insight_results=insights,
        source_paths=source_paths,
        generated_at=args.generated_at,
    )
    requests = build_insight_requests(payload)
    featured_rows = build_item_request_rows(payload, featured_only=True)
    all_rows = build_item_request_rows(payload, featured_only=False)

    browser_path = args.output_dir / "index.html"
    browser_metrics_path = args.output_dir / "metrics.json"
    requests_path = args.work_dir / "insight_requests.jsonl"
    featured_manifest_path = args.work_dir / "item_request_featured.csv"
    all_manifest_path = args.work_dir / "item_request_all_test.csv"
    overlay_template_path = args.work_dir / "item_overlay_template.csv"
    bundle_path = args.work_dir / "avito_case_browser_share_bundle.zip"
    build_status = _insight_build_status(len(requests), len(insights))

    render_browser_html(args.template, browser_path, payload)
    write_jsonl(requests_path, requests)
    write_csv(featured_manifest_path, featured_rows)
    write_csv(all_manifest_path, all_rows)
    write_csv(overlay_template_path, _overlay_template_rows(featured_rows))

    metrics = {
        "generated": args.generated_at,
        "status": build_status,
        "claim_scope": payload["meta"]["claim_scope"],
        "protocol": {
            "browser_version": payload["meta"]["browser_version"],
            "split": args.split,
            "score_tie_break": "item_id ascending",
            "graded_target": "contacts_daily / max within SERP",
            "actual_outcome_marker": "has_tc_events; post-exposure display/diagnostic only",
            "history_temporal_status": "unknown; no SERP rank_time",
            "qwen_ranking": "existing Qwen2.5-7B L0 score; no history and no position",
        },
        "summary": payload["summary"],
        "artifact_audit": payload["meta"]["audit"],
        "reported_metric_audit": _reported_metric_audit(
            payload["summary"], args.catboost_metrics, args.llm_metrics
        ),
        "featured_cases": [
            {
                "serp_x": case["serp_x"],
                "user_id": case["user_id"],
                "reasons": case["featured_reasons"],
                "history_count": case["history_summary"]["count"],
                "metrics": case["metrics"],
            }
            for case in payload["cases"]
            if case["serp_x"] in set(payload["featured_serps"])
        ],
        "insights": {
            "status": build_status,
            "requests": len(requests),
            "loaded": payload["meta"]["insight_results"]["loaded"],
            "ok": payload["meta"]["insight_results"]["ok"],
            "direct_llm_accepted": payload["meta"]["insight_results"]["direct_llm_accepted"],
            "guarded_fallback": payload["meta"]["insight_results"]["guarded_fallback"],
            "prompt_version": INSIGHT_PROMPT_VERSION,
            "validation_version": INSIGHT_VALIDATION_VERSION,
            "contract": insight_contract,
            "contract_source": insight_contract_source,
            "contract_path": str(insight_contract_path.relative_to(ROOT)),
            "contract_error": insight_contract_error,
            "cache_rejections": dict(sorted(insight_cache_rejections.items())),
        },
        "item_request": {
            "featured_unique_items": len(featured_rows),
            "all_test_unique_items": len(all_rows),
            "featured_manifest": str(featured_manifest_path.relative_to(ROOT)),
            "all_test_manifest": str(all_manifest_path.relative_to(ROOT)),
            "overlay_template": str(overlay_template_path.relative_to(ROOT)),
        },
        "local_artifacts": {
            "browser": str(browser_path.relative_to(ROOT)),
            "insight_requests": str(requests_path.relative_to(ROOT)),
            "share_bundle": (
                str(bundle_path.relative_to(ROOT)) if build_status == "done" else None
            ),
            "expected_share_bundle": str(bundle_path.relative_to(ROOT)),
        },
        "source_fingerprints": payload["meta"]["source_fingerprints"],
        "limitations": payload["meta"]["limitations"],
    }
    write_json(browser_metrics_path, metrics)
    write_json(args.metrics_output, metrics)
    if _bundle_should_be_written(build_status, bundle_path):
        create_share_bundle(
            bundle_path,
            browser_path,
            browser_metrics_path,
            featured_manifest_path,
            overlay_template_path,
            all_manifest_path,
            _referenced_local_assets(payload, args.output_dir),
        )

    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Browser: {browser_path}")
    print(f"Insight requests: {requests_path}")
    if build_status == "done":
        print(f"Share bundle: {bundle_path}")
    else:
        print(f"Share bundle not written: {len(insights)}/{len(requests)} insights validate")


if __name__ == "__main__":
    main()
