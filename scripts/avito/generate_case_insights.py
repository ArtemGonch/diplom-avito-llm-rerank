#!/usr/bin/env python3
"""Generate cached, structured Qwen insights for featured Avito browser cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from avito_case_browser import INSIGHT_VALIDATION_VERSION, extract_json_object  # noqa: E402
from avito_case_browser.core import write_json, write_jsonl  # noqa: E402
from avito_case_browser.insights import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_INSIGHT_MODEL,
    DEFAULT_INSIGHT_MODEL_REVISION,
    DEFAULT_INSIGHT_DEVICE,
    DEFAULT_LOAD_IN_4BIT,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_RETRY_MAX_NEW_TOKENS,
    GENERATION_CONTRACT_VERSION,
    INSIGHT_SCHEMAS,
    insight_cache_signature,
    load_pinned_tokenizer,
    make_token_provenance_verifier,
    materialize_attempt_token_provenance,
    normalised_evidence_fields,
    parsed_insight_sha256,
    request_manifest_sha256,
    runtime_generation_environment,
    runtime_generation_versions,
    text_sha256,
    validate_and_normalise_insight,
    validate_cached_insight,
    validate_insight_parsed,
)


SCHEMAS = INSIGHT_SCHEMAS


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_requests(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        request_id = str(row.get("request_id", ""))
        if not request_id or request_id in seen:
            raise ValueError(f"Missing/duplicate request_id at line {line_number}: {request_id!r}")
        kind = str(row.get("kind", ""))
        if kind not in SCHEMAS:
            raise ValueError(f"Unknown insight kind at line {line_number}: {kind!r}")
        prompt = str(row.get("prompt", ""))
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_hash != row.get("prompt_sha256"):
            raise ValueError(f"Prompt hash mismatch for {request_id}")
        evidence = row.get("evidence")
        if not isinstance(evidence, dict) or not evidence:
            raise ValueError(f"Missing structured evidence for {request_id}")
        if kind == "item" and not isinstance(evidence.get("attributes"), dict):
            raise ValueError(f"Item evidence lacks attributes for {request_id}")
        if kind == "user":
            history = evidence.get("history")
            summary = evidence.get("history_summary")
            if not isinstance(history, list) or not isinstance(summary, dict):
                raise ValueError(f"User evidence lacks history/summary for {request_id}")
            if summary.get("count") != len(history):
                raise ValueError(f"User evidence count mismatch for {request_id}")
        seen.add(request_id)
        rows.append(row)
    return rows


# Backwards-compatible private alias used by focused tests.  Runtime validation
# is shared with the builder, so the two paths cannot silently drift.
_validate_parsed = validate_insight_parsed


_normalised_evidence_fields = normalised_evidence_fields


def _apply_evidence_normalisation(
    request: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    parsed = result.get("parsed")
    if not isinstance(parsed, dict):
        return result
    updated = dict(result)
    normalised, error, fields = validate_and_normalise_insight(request, parsed)
    if normalised is None:
        raise ValueError(f"Insight failed shared validation: {error}")
    updated["parsed"] = normalised
    updated["evidence_normalized_fields"] = fields
    updated["parsed_sha256"] = parsed_insight_sha256(normalised)
    return updated


def _guarded_fallback(request: dict[str, Any], rejected: dict[str, Any]) -> dict[str, Any]:
    """Replace a rejected response with facts copied deterministically from evidence."""
    if request["kind"] == "item":
        parsed = {
            **_normalised_evidence_fields(request),
            "seller_claims": [],
        }
    else:
        parsed = _normalised_evidence_fields(request)
    validated, error, fields = validate_and_normalise_insight(request, parsed)
    if validated is None:
        raise AssertionError(f"Internal guarded fallback is invalid: {error}")
    result = dict(rejected)
    result.update(
        {
            "request_id": request["request_id"],
            "kind": request["kind"],
            "entity_id": request["entity_id"],
            "prompt_version": request["prompt_version"],
            "prompt_sha256": request["prompt_sha256"],
            "status": "ok",
            "parsed": validated,
            "original_status": rejected.get("status"),
            "original_validation_error": rejected.get("validation_error"),
            "validation_error": None,
            "validation_version": INSIGHT_VALIDATION_VERSION,
            "llm_output_accepted": False,
            "fallback_applied": True,
            "generation_pass": f"{rejected.get('generation_pass', 'primary')}+guarded_fallback",
            "evidence_normalized_fields": fields,
            "parsed_sha256": parsed_insight_sha256(validated),
        }
    )
    return result


def _generation_contract(
    args: argparse.Namespace,
    requests: list[dict[str, Any]],
    *,
    environment: dict[str, Any],
    tokenizer_provenance: dict[str, Any],
) -> dict[str, Any]:
    contract = {
        "contract_version": GENERATION_CONTRACT_VERSION,
        "model": args.model,
        "model_revision": args.model_revision,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "retry_max_new_tokens": args.retry_max_new_tokens,
        "load_in_4bit": args.load_in_4bit,
        "device": args.device,
        "device_type": args.device.split(":", 1)[0],
        "compute_dtype": args.compute_dtype,
        "batch_size": args.batch_size,
        "request_count": len(requests),
        "request_manifest_sha256": request_manifest_sha256(requests),
        "tokenizer": tokenizer_provenance,
        "code_fingerprints": {
            "generator_sha256": _file_sha256(Path(__file__)),
            "insight_contract_sha256": _file_sha256(
                ROOT / "src/avito_case_browser/insights.py"
            ),
            "browser_core_sha256": _file_sha256(ROOT / "src/avito_case_browser/core.py"),
        },
        "local_files_only": args.local_files_only,
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
        "torch_version": args.torch_version,
        "cuda_runtime_version": args.cuda_runtime_version,
        "transformers_version": args.transformers_version,
        "do_sample": False,
        **environment,
    }
    contract["signature"] = insight_cache_signature(contract)
    return contract


def _cache_signature(args: argparse.Namespace) -> str:
    """Focused-test helper: hash all supplied generation arguments."""
    values = dict(vars(args))
    values.pop("signature", None)
    return insight_cache_signature(values)


def _load_cache(
    path: Path,
    requests: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    token_provenance_verifier: Any,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        request_id = str(row.get("request_id", ""))
        request = requests.get(request_id)
        if request is None:
            continue
        candidate = row
        partial = row.get("status") == "parse_error"
        if partial:
            # Validate an interrupted primary/retry record through the exact
            # same guarded-fallback path used at finalization, then keep the
            # original partial row so the next run can resume at retry.
            try:
                candidate = _guarded_fallback(request, row)
            except (AssertionError, ValueError):
                continue
        safe, _ = validate_cached_insight(
            request,
            candidate,
            contract=contract,
            validation_version=INSIGHT_VALIDATION_VERSION,
            token_provenance_verifier=token_provenance_verifier,
        )
        if safe is not None:
            cached[request_id] = row if partial else safe
    return cached


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests",
        type=Path,
        default=ROOT / "data/avito/case_browser/insight_requests.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/avito/case_browser/insights.jsonl",
    )
    parser.add_argument(
        "--contract-output",
        type=Path,
        default=ROOT / "data/avito/case_browser/generation_contract.json",
        help="Standalone signed execution contract consumed by the browser builder.",
    )
    parser.add_argument("--model", default=DEFAULT_INSIGHT_MODEL)
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_INSIGHT_MODEL_REVISION,
    )
    parser.add_argument("--device", default=DEFAULT_INSIGHT_DEVICE)
    parser.add_argument(
        "--compute-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Generation precision; auto resolves to float16 on CUDA and float32 otherwise.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--retry-max-new-tokens",
        type=int,
        default=DEFAULT_RETRY_MAX_NEW_TOKENS,
        help="Second-pass budget used only when the primary JSON is truncated or invalid.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LOAD_IN_4BIT,
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.batch_size != DEFAULT_BATCH_SIZE:
        parser.error(
            "The canonical auditable insight cache requires --batch-size 1; "
            "non-canonical padded peer batches are deliberately unsupported."
        )
    if args.compute_dtype == "auto":
        args.compute_dtype = "float16" if args.device.startswith("cuda") else "float32"
    # These switches are set before the first torch import in the CLI process.
    # PYTHONHASHSEED itself must still be provided at process startup and its
    # actual environment value is captured below.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    versions = runtime_generation_versions()
    args.torch_version = versions["torch_version"]
    args.cuda_runtime_version = versions["cuda_runtime_version"]
    args.transformers_version = versions["transformers_version"]

    requests = _load_requests(args.requests)
    by_id = {row["request_id"]: row for row in requests}
    environment = runtime_generation_environment(args.device)
    tokenizer, tokenizer_provenance = load_pinned_tokenizer(
        args.model,
        args.model_revision,
        local_files_only=args.local_files_only,
    )
    contract = _generation_contract(
        args,
        requests,
        environment=environment,
        tokenizer_provenance=tokenizer_provenance,
    )
    signature = contract["signature"]
    write_json(args.contract_output, contract)
    token_provenance_verifier = make_token_provenance_verifier(tokenizer, contract)
    results = _load_cache(
        args.output,
        by_id,
        contract,
        token_provenance_verifier,
    )
    missing = [row for row in requests if row["request_id"] not in results]
    retry_pending = [
        row
        for row in requests
        if results.get(row["request_id"], {}).get("status") == "parse_error"
        and len(results[row["request_id"]].get("generation_attempts", [])) == 1
    ]
    print(
        f"Insight requests: total={len(requests)}, cached={len(results)}, "
        f"missing_primary={len(missing)}, pending_retry={len(retry_pending)}",
        flush=True,
    )

    if missing or retry_pending:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        torch_dtype = getattr(torch, args.compute_dtype)
        model_kwargs: dict[str, Any] = {
            "revision": args.model_revision,
            "local_files_only": args.local_files_only,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if args.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = {"": args.device}
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        if not args.load_in_4bit:
            model.to(args.device)
        model.eval()
        # Qwen's saved generation config contains sampling-only values.  They
        # are irrelevant in deterministic mode and otherwise trigger warnings.
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

        def generate_rows(rows: list[dict[str, Any]], max_new_tokens: int, pass_name: str) -> None:
            for offset in range(0, len(rows), args.batch_size):
                batch = rows[offset : offset + args.batch_size]
                if len(batch) != 1:
                    raise AssertionError("canonical token provenance requires one request per batch")
                cpu_inputs, prompt_text, effective_prompt, token_provenance = (
                    materialize_attempt_token_provenance(
                        tokenizer,
                        batch[0],
                        contract,
                        pass_name,
                    )
                )
                prompts = [prompt_text]
                effective_prompts = [effective_prompt]
                inputs = cpu_inputs.to(args.device)
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                elapsed = time.perf_counter() - started
                continuations = generated[:, inputs["input_ids"].shape[1] :]
                texts = tokenizer.batch_decode(continuations, skip_special_tokens=True)
                for row, prompt_text, effective_prompt, raw_text in zip(
                    batch, prompts, effective_prompts, texts
                ):
                    raw_object = extract_json_object(raw_text)
                    llm_parsed, _ = validate_insight_parsed(row["kind"], raw_object)
                    parsed, validation_error, fields = validate_and_normalise_insight(
                        row, raw_object
                    )
                    attempt = {
                        "pass": pass_name,
                        "pass_batch_index": offset // args.batch_size,
                        "configured_batch_size": args.batch_size,
                        "actual_batch_size": len(batch),
                        **token_provenance,
                        "max_new_tokens": max_new_tokens,
                        "generation_seconds": elapsed,
                        "status": "ok" if parsed is not None else "parse_error",
                        "validation_error": validation_error,
                        "raw_text": raw_text,
                        "raw_text_sha256": text_sha256(raw_text),
                        "llm_parsed_sha256": (
                            parsed_insight_sha256(llm_parsed)
                            if llm_parsed is not None
                            else None
                        ),
                    }
                    previous_attempts = results.get(row["request_id"], {}).get(
                        "generation_attempts", []
                    )
                    attempts = (
                        [attempt]
                        if pass_name == "primary"
                        else [*previous_attempts, attempt]
                    )
                    results[row["request_id"]] = {
                        "request_id": row["request_id"],
                        "kind": row["kind"],
                        "entity_id": row["entity_id"],
                        "prompt_version": row["prompt_version"],
                        "prompt_sha256": row["prompt_sha256"],
                        "effective_prompt_sha256": attempt["effective_prompt_sha256"],
                        "signature": signature,
                        "model": args.model,
                        "model_revision": args.model_revision,
                        "device": args.device,
                        "device_type": args.device.split(":", 1)[0],
                        "compute_dtype": args.compute_dtype,
                        **versions,
                        "do_sample": False,
                        "generation_contract": contract,
                        "generation_attempts": attempts,
                        "status": "ok" if parsed is not None else "parse_error",
                        "parsed": parsed,
                        "parsed_sha256": (
                            parsed_insight_sha256(parsed) if parsed is not None else None
                        ),
                        "validation_error": validation_error,
                        "validation_version": INSIGHT_VALIDATION_VERSION,
                        "llm_output_accepted": parsed is not None,
                        "fallback_applied": False,
                        "raw_text": raw_text,
                        "raw_text_sha256": attempt["raw_text_sha256"],
                        "llm_parsed_sha256": attempt["llm_parsed_sha256"],
                        "generation_pass": pass_name,
                        "max_new_tokens_used": max_new_tokens,
                        "batch_generation_seconds": elapsed,
                        "evidence_normalized_fields": fields,
                    }
                write_jsonl(args.output, [results[key] for key in sorted(results)])
                print(
                    f"{pass_name}: {min(offset + len(batch), len(rows))}/{len(rows)}; "
                    f"valid={sum(row.get('status') == 'ok' for row in results.values())}/{len(results)}",
                    flush=True,
                )

        generate_rows(missing, args.max_new_tokens, "primary")
        retry_ids = {
            row["request_id"] for row in requests
            if results.get(row["request_id"], {}).get("status") != "ok"
            and len(results.get(row["request_id"], {}).get("generation_attempts", [])) == 1
        }
        retry_rows = [row for row in requests if row["request_id"] in retry_ids]
        if retry_rows:
            generate_rows(retry_rows, args.retry_max_new_tokens, "retry")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    for request_id, row in list(results.items()):
        if row.get("status") != "ok":
            results[request_id] = _guarded_fallback(by_id[request_id], row)
        else:
            results[request_id] = _apply_evidence_normalisation(by_id[request_id], row)
    write_jsonl(args.output, [results[key] for key in sorted(results)])
    failed = [key for key, value in results.items() if value.get("status") != "ok"]
    fallbacks = sum(bool(value.get("fallback_applied")) for value in results.values())
    print(
        f"Wrote {args.output}; ok={len(results) - len(failed)}, failed={len(failed)}, "
        f"direct_llm={len(results) - len(failed) - fallbacks}, guarded_fallback={fallbacks}"
    )
    if failed:
        print(f"Failed request IDs: {failed}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
