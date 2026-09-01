#!/usr/bin/env python3
"""Leakage-free no-history LLM scoring on the local Avito diagnostic split.

The script deliberately does not claim to reproduce the team's CatBoost or a
personalized L1/L2/L3 experiment.  It provides the executable L0 control that
is possible with the current extract: query proxies plus rank-time item data.
When validation/test CatBoost scores are supplied, it also evaluates a
validation-selected pre-inference ambiguity gate (call the LLM only for SERPs
whose CatBoost top-two margin is small).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.metrics import evaluate_batch  # noqa: E402
from data.avito import AvitoSERP  # noqa: E402


PROMPT_FEATURES = [
    "query_infm_logical_category",
    "query_loc",
    "title",
    "description_short",
    "brand",
    "model_name",
    "price",
    "mileage_km",
    "year_text",
    "body_type",
    "fuel_text",
    "gearbox_text",
    "drive_text",
    "doors_text",
]

FORBIDDEN_PROMPT_FEATURES = {
    "contacts_daily",
    "clicks_daily",
    "has_tc_events",
    "has_x_events",
    "serp_is_positive",
    "block_pos",
}

PROMPT_TEMPLATE_VERSION = 1


def prompt_feature_columns() -> list[str]:
    overlap = FORBIDDEN_PROMPT_FEATURES & set(PROMPT_FEATURES)
    if overlap:
        raise AssertionError(f"Post-exposure/position leakage in LLM prompt: {sorted(overlap)}")
    return list(PROMPT_FEATURES)


def _clean(value: object, limit: int = 240) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "не указано"
    text = " ".join(str(value).split())
    return text[:limit] if text else "не указано"


def build_prompt(row: pd.Series) -> str:
    """Construct a no-history prompt from allow-listed rank-time fields."""
    _ = prompt_feature_columns()
    return (
        "Оцени релевантность объявления текущему поиску автомобилей на Avito. "
        "Не оценивай популярность и не используй клики/контакты. "
        "Шкала: 0 — не подходит, 1 — слабо, 2 — возможно подходит, "
        "3 — хорошо подходит, 4 — максимально подходит. Ответь одной цифрой.\n"
        f"Поиск: категория={_clean(row['query_infm_logical_category'], 80)}; "
        f"регион_id={_clean(row['query_loc'], 40)}.\n"
        f"Объявление: title={_clean(row['title'], 140)}; "
        f"brand={_clean(row['brand'], 50)}; model={_clean(row['model_name'], 60)}; "
        f"year={_clean(row['year_text'], 20)}; body={_clean(row['body_type'], 40)}; "
        f"fuel={_clean(row['fuel_text'], 30)}; gearbox={_clean(row['gearbox_text'], 30)}; "
        f"drive={_clean(row['drive_text'], 30)}; doors={_clean(row['doors_text'], 20)}; "
        f"price={_clean(row['price'], 30)}; mileage_km={_clean(row['mileage_km'], 30)}; "
        f"description={_clean(row['description_short'], 240)}.\nОценка:"
    )


def _prepare_frame(items: pd.DataFrame, serp_names: list[str], split: str) -> pd.DataFrame:
    columns = ["serp_x", "item_id", "contacts_daily", *prompt_feature_columns()]
    frame = items.loc[items["serp_x"].isin(set(serp_names)), columns].copy()
    # Candidate-id tie breaking prevents score ties from silently falling back
    # to the production position order, which is intentionally absent from L0.
    frame = frame.sort_values(["serp_x", "item_id"], kind="stable").reset_index(drop=True)
    relevance = frame["contacts_daily"].clip(lower=0).astype(np.float64)
    maxima = relevance.groupby(frame["serp_x"]).transform("max")
    frame["evaluation_relevance"] = np.where(maxima > 0, relevance / maxima, 0.0)
    frame["split"] = split
    return frame


def _evaluate(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for _, group in frame.groupby("serp_x", sort=False):
        scores.append(group[score_column].to_numpy(dtype=np.float64))
        labels.append(group["evaluation_relevance"].to_numpy(dtype=np.float64))
    return evaluate_batch(scores, labels, ks=(5, 10))


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    return (values - float(np.mean(values))) / std if std > 1e-12 else np.zeros_like(values)


def _add_blend_scores(frame: pd.DataFrame, alpha: float, margin_threshold: float) -> pd.DataFrame:
    out = frame.copy()
    blended = np.empty(len(out), dtype=np.float64)
    called = np.empty(len(out), dtype=bool)
    offset = 0
    for _, group in out.groupby("serp_x", sort=False):
        cb = group["catboost_score"].to_numpy(dtype=np.float64)
        llm = group["llm_score"].to_numpy(dtype=np.float64)
        descending = np.sort(cb)[::-1]
        margin = float(descending[0] - descending[1]) if len(descending) > 1 else float("inf")
        use_llm = margin <= margin_threshold
        n = len(group)
        blended[offset : offset + n] = _zscore(cb) + (alpha * _zscore(llm) if use_llm else 0.0)
        called[offset : offset + n] = use_llm
        offset += n
    out["gated_blend_score"] = blended
    out["llm_called"] = called
    return out


def _select_gate(dev: pd.DataFrame) -> tuple[float, float, list[dict[str, float]]]:
    margins: list[float] = []
    for _, group in dev.groupby("serp_x", sort=False):
        scores = np.sort(group["catboost_score"].to_numpy(dtype=np.float64))[::-1]
        margins.append(float(scores[0] - scores[1]) if len(scores) > 1 else float("inf"))
    finite = np.asarray([x for x in margins if np.isfinite(x)], dtype=np.float64)
    no_call_threshold = float(np.min(finite) - max(1.0, abs(float(np.min(finite)))) * 1e-9)
    thresholds = sorted(
        set([no_call_threshold, *np.quantile(finite, np.linspace(0.1, 1.0, 10)).tolist()])
    )
    rows: list[dict[str, float]] = []
    best: tuple[float, float, float, float] | None = None
    for alpha in np.linspace(0.1, 1.0, 10):
        for threshold in thresholds:
            scored = _add_blend_scores(dev, float(alpha), threshold)
            ndcg = _evaluate(scored, "gated_blend_score")["ndcg@10"]
            call_fraction = float(scored.groupby("serp_x")["llm_called"].first().mean())
            rows.append(
                {
                    "alpha": float(alpha),
                    "catboost_margin_threshold": float(threshold),
                    "ndcg@10": float(ndcg),
                    "llm_call_fraction": call_fraction,
                }
            )
            key = (float(ndcg), -call_fraction, -float(alpha), -float(threshold))
            if best is None or key > best:
                best = key
                selected = (float(alpha), float(threshold))
    return selected[0], selected[1], rows


def merge_exact_catboost_scores(frame: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Join a score artifact only when its candidate keys match exactly."""
    key_columns = ["split", "serp_x", "item_id"]
    required = [*key_columns, "catboost_score"]
    missing_columns = set(required) - set(scores.columns)
    if missing_columns:
        raise ValueError(f"CatBoost score artifact lacks columns: {sorted(missing_columns)}")
    score_view = scores[required].copy()
    if score_view.duplicated(key_columns).any():
        raise ValueError("CatBoost score artifact contains duplicate candidate keys")
    frame_keys = frame[key_columns]
    if frame_keys.duplicated(key_columns).any():
        raise ValueError("Evaluation frame contains duplicate candidate keys")
    audit = frame_keys.merge(
        score_view[key_columns],
        on=key_columns,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    counts = audit["_merge"].value_counts().to_dict()
    if counts.get("left_only", 0) or counts.get("right_only", 0):
        raise ValueError(
            "CatBoost artifact does not match the exact dev/test candidate set: "
            f"missing={counts.get('left_only', 0)}, extra={counts.get('right_only', 0)}"
        )
    return frame.merge(score_view, on=key_columns, how="left", validate="one_to_one")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, default=ROOT / "items_with_attrs.parquet")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--model-revision",
        default="a09a35458c702b33eeacc393d103063234e8bc28",
        help="Pinned Hugging Face commit for reproducible logits.",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data/avito/benchmark/llm_pointwise_cache.jsonl",
    )
    parser.add_argument(
        "--catboost-scores",
        type=Path,
        default=ROOT / "data/avito/benchmark/local_catboost_eval_scores.parquet",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=ROOT / "data/avito/benchmark/local_llm_pointwise_eval_scores.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/current/metrics/avito_local_llm_diagnostic.json",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, __version__

    started = time.time()
    data = AvitoSERP.from_parquet(args.items, users_path=None, label_field="contacts", min_serp_size=10)
    _, val_ids, test_ids = data.train_val_test_split(seed=args.split_seed)
    dev = _prepare_frame(data.items, [data.idx2serp[i] for i in val_ids], "dev")
    test = _prepare_frame(data.items, [data.idx2serp[i] for i in test_ids], "test")
    frame = pd.concat([dev, test], ignore_index=True)
    frame["_prompt"] = [build_prompt(row) for _, row in frame.iterrows()]
    frame["_prompt_sha256"] = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in frame["_prompt"]
    ]

    signature_payload = {
        "version": 1,
        "model": args.model,
        "model_revision": args.model_revision,
        "split_seed": args.split_seed,
        "items_sha256": _file_sha256(args.items),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_features": prompt_feature_columns(),
        "max_prompt_tokens": args.max_prompt_tokens,
        "scoring": "expected value over next-token labels 0..4",
        "candidate_tie_break": "item_id ascending",
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    expected_prompt_hashes = {
        (str(row["serp_x"]), int(row["item_id"])): str(row["_prompt_sha256"])
        for _, row in frame.iterrows()
    }
    cached: dict[tuple[str, int], float] = {}
    if args.cache.exists():
        for line in args.cache.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (str(row["serp_x"]), int(row["item_id"]))
            if (
                row.get("signature") == signature
                and row.get("prompt_sha256") == expected_prompt_hashes.get(key)
            ):
                cached[key] = float(row["llm_score"])

    missing_indices = [
        i
        for i, row in frame.iterrows()
        if (str(row["serp_x"]), int(row["item_id"])) not in cached
    ]
    inference_started = time.time()
    scored_this_run = len(missing_indices)
    previous_full_pass_seconds: float | None = None
    if not missing_indices and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        previous_full_pass_seconds = float(
            previous.get("full_pass_seconds_observed", previous.get("inference_seconds_this_run", 0.0))
        )
    if missing_indices:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=args.model_revision, use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        label_ids: list[int] = []
        for label in range(5):
            encoded = tokenizer.encode(str(label), add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Label {label} is not one token for {args.model}: {encoded}")
            label_ids.append(encoded[0])
        kwargs: dict[str, object] = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        if args.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.model_revision, **kwargs
        )
        model.eval()
        device = next(model.parameters()).device
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with args.cache.open("a", encoding="utf-8") as cache_file:
            for offset in range(0, len(missing_indices), args.batch_size):
                indices = missing_indices[offset : offset + args.batch_size]
                prompts = []
                for index in indices:
                    messages = [
                        {"role": "system", "content": "Ты точный ранжировщик объявлений."},
                        {"role": "user", "content": str(frame.iloc[index]["_prompt"])},
                    ]
                    prompts.append(
                        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    )
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_prompt_tokens,
                ).to(device)
                with torch.inference_mode():
                    logits = model(**inputs).logits[:, -1, label_ids].float()
                    probabilities = torch.softmax(logits, dim=-1)
                    values = torch.arange(5, device=probabilities.device, dtype=probabilities.dtype)
                    batch_scores = (probabilities * values).sum(dim=-1).cpu().numpy()
                for index, score in zip(indices, batch_scores):
                    row = frame.iloc[index]
                    key = (str(row["serp_x"]), int(row["item_id"]))
                    cached[key] = float(score)
                    cache_file.write(
                        json.dumps(
                            {
                                "signature": signature,
                                "serp_x": key[0],
                                "item_id": key[1],
                                "prompt_sha256": str(row["_prompt_sha256"]),
                                "llm_score": float(score),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                cache_file.flush()
                print(f"Scored {min(offset + len(indices), len(missing_indices))}/{len(missing_indices)}", flush=True)
        del model
        torch.cuda.empty_cache()

    frame["llm_score"] = [
        cached[(str(row["serp_x"]), int(row["item_id"]))] for _, row in frame.iterrows()
    ]
    dev = frame[frame["split"] == "dev"].copy()
    test = frame[frame["split"] == "test"].copy()
    payload: dict[str, object] = {
        "claim_scope": "local diagnostic L0 only; no user history; not team CatBoost protocol",
        "protocol": signature_payload,
        "counts": {
            "dev_serps": int(dev["serp_x"].nunique()),
            "test_serps": int(test["serp_x"].nunique()),
            "dev_candidates": len(dev),
            "test_candidates": len(test),
        },
        "llm_pointwise": {
            "dev": _evaluate(dev, "llm_score"),
            "test": _evaluate(test, "llm_score"),
        },
        "candidates_scored_this_run": scored_this_run,
        "candidates_reused_from_cache": len(frame) - scored_this_run,
        "inference_seconds_this_run": time.time() - inference_started,
        "full_pass_seconds_observed": (
            time.time() - inference_started
            if scored_this_run == len(frame)
            else previous_full_pass_seconds
        ),
        "transformers_version": __version__,
    }

    score_columns = ["split", "serp_x", "item_id", "evaluation_relevance", "llm_score"]
    if args.catboost_scores.exists():
        cb = pd.read_parquet(args.catboost_scores)
        frame = merge_exact_catboost_scores(frame, cb)
        dev = frame[frame["split"] == "dev"].copy()
        test = frame[frame["split"] == "test"].copy()
        alpha, threshold, sweep = _select_gate(dev)
        dev_gated = _add_blend_scores(dev, alpha, threshold)
        test_gated = _add_blend_scores(test, alpha, threshold)
        payload["local_catboost_control"] = {
            "dev": _evaluate(dev, "catboost_score"),
            "test": _evaluate(test, "catboost_score"),
        }
        payload["pre_inference_ambiguity_gate"] = {
            "selection": "alpha and CatBoost top-two margin threshold selected on dev NDCG@10; test once",
            "alpha": alpha,
            "catboost_margin_threshold": threshold,
            "dev": _evaluate(dev_gated, "gated_blend_score"),
            "test": _evaluate(test_gated, "gated_blend_score"),
            "dev_llm_call_fraction": float(dev_gated.groupby("serp_x")["llm_called"].first().mean()),
            "test_llm_call_fraction": float(test_gated.groupby("serp_x")["llm_called"].first().mean()),
            "dev_sweep": sweep,
        }
        frame = pd.concat([dev_gated, test_gated], ignore_index=True)
        score_columns += ["catboost_score", "gated_blend_score", "llm_called"]

    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    frame[score_columns].to_parquet(args.scores_output, index=False)
    payload["scores_artifact"] = str(args.scores_output.relative_to(ROOT))
    payload["elapsed_seconds"] = time.time() - started
    payload["blocking_dependency"] = {
        "personalized_l1_l3": "No rank-time SERP timestamp/cutoff in current extract",
        "authoritative_comparison": "Roma/team CatBoost split, scores and relevance contract absent",
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
