#!/usr/bin/env python3
"""Evaluate text-only, image-only and late-fusion on fixed Amazon-C4 candidates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.amazon_c4 import aggregate_metric_rows, ranking_metrics  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def _load_manifest(path: Path) -> dict[str, dict]:
    return {str(row["item_id"]): row for row in _read_jsonl(path)}


def _encode_images(
    rows: Sequence[dict],
    *,
    model_name: str,
    model_revision: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name, revision=model_revision)
    model = CLIPModel.from_pretrained(
        model_name,
        revision=model_revision,
        use_safetensors=True,
    ).to(device).eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images: list[Image.Image] = []
        for row in batch_rows:
            with Image.open(ROOT / row["local_path"]) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            features = model.get_image_features(pixel_values=pixel_values)
            features = torch.nn.functional.normalize(features.float(), dim=1)
        chunks.append(features.cpu().numpy())
        print(f"CLIP images {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(chunks, axis=0)


def _load_or_encode_images(
    manifest: dict[str, dict],
    *,
    cache_dir: Path,
    model_name: str,
    model_revision: str,
    batch_size: int,
    device: str,
) -> tuple[list[str], np.ndarray]:
    rows = sorted(
        (row for row in manifest.values() if row.get("status") == "ok"),
        key=lambda row: str(row["item_id"]),
    )
    item_ids = [str(row["item_id"]) for row in rows]
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = cache_dir / "clip_image_embeddings.npy"
    metadata_path = cache_dir / "clip_image_embeddings.meta.json"
    expected = {
        "model": model_name,
        "model_revision": model_revision,
        "n_images": len(item_ids),
        "first_item_id": item_ids[0],
        "last_item_id": item_ids[-1],
        "item_ids": item_ids,
    }
    if embeddings_path.exists() and metadata_path.exists():
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual == expected:
            embeddings = np.load(embeddings_path)
            if embeddings.shape[0] == len(item_ids):
                print(f"Reusing CLIP image cache: {embeddings_path}")
                return item_ids, _normalise_rows(embeddings.astype(np.float32, copy=False))
    embeddings = _encode_images(
        rows,
        model_name=model_name,
        model_revision=model_revision,
        batch_size=batch_size,
        device=device,
    )
    np.save(embeddings_path, embeddings.astype(np.float16))
    _write_json(metadata_path, expected)
    return item_ids, embeddings


def _encode_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    model_revision: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name, revision=model_revision)
    model = CLIPModel.from_pretrained(
        model_name,
        revision=model_revision,
        use_safetensors=True,
    ).to(device).eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        inputs = processor(text=batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = model.get_text_features(**inputs)
            features = torch.nn.functional.normalize(features.float(), dim=1)
        chunks.append(features.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _zscore(values: np.ndarray, finite: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if finite is None:
        finite = np.isfinite(values)
    out = np.zeros_like(values)
    selected = values[finite]
    if selected.size:
        scale = float(np.std(selected))
        if scale < 1e-12:
            scale = 1.0
        out[finite] = (selected - float(np.mean(selected))) / scale
    return out


def _ordered_item_ids(item_ids: Sequence[str], scores: np.ndarray) -> list[str]:
    indices = np.arange(len(item_ids))
    order = np.lexsort((indices, -np.asarray(scores)))
    return [item_ids[int(index)] for index in order]


def _score_queries(
    rows: Sequence[dict],
    query_embeddings: np.ndarray,
    image_lookup: dict[str, np.ndarray],
    *,
    text_weight: float,
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_query: list[dict] = []
    for row, query_embedding in zip(rows, query_embeddings):
        item_ids = [str(candidate["item_id"]) for candidate in row["candidates"]]
        text_scores = np.asarray(
            [float(candidate["retrieval_score"]) for candidate in row["candidates"]],
            dtype=np.float64,
        )
        image_scores = np.full(len(item_ids), np.nan, dtype=np.float64)
        for index, item_id in enumerate(item_ids):
            embedding = image_lookup.get(item_id)
            if embedding is not None:
                image_scores[index] = float(query_embedding @ embedding)
        has_image = np.isfinite(image_scores)
        text_z = _zscore(text_scores)
        image_z = _zscore(image_scores, has_image)
        if has_image.any():
            image_only_scores = image_scores.copy()
            image_only_scores[~has_image] = float(np.min(image_scores[has_image]) - 1.0)
        else:
            image_only_scores = np.zeros_like(image_scores)
        fusion_scores = text_weight * text_z + (1.0 - text_weight) * image_z
        method_scores = {
            "mm0_text_only": text_scores,
            "mm1_image_only": image_only_scores,
            "mm2_late_fusion": fusion_scores,
        }
        row_metrics: dict[str, dict[str, float]] = {}
        for method, scores in method_scores.items():
            ranked = _ordered_item_ids(item_ids, scores)
            result = ranking_metrics(ranked, str(row["positive_item_id"]), k=10)
            metrics[method].append(result)
            row_metrics[method] = result
        positive_manifest = str(row["positive_item_id"]) in image_lookup
        per_query.append(
            {
                "split": row["split"],
                "qid": row["qid"],
                "user_id": row["user_id"],
                "positive_item_id": row["positive_item_id"],
                "positive_in_candidates": row["positive_item_id"] in item_ids,
                "positive_has_image": positive_manifest,
                "candidate_image_coverage": float(np.mean(has_image)),
                "metrics": row_metrics,
            }
        )
    return {method: aggregate_metric_rows(values) for method, values in metrics.items()}, per_query


def _select_text_weight(
    rows: Sequence[dict],
    query_embeddings: np.ndarray,
    image_lookup: dict[str, np.ndarray],
    grid: Sequence[float],
) -> tuple[float, list[dict]]:
    sweep: list[dict] = []
    for weight in grid:
        metrics, _ = _score_queries(
            rows,
            query_embeddings,
            image_lookup,
            text_weight=float(weight),
        )
        sweep.append(
            {
                "text_weight": float(weight),
                **metrics["mm2_late_fusion"],
            }
        )
    # Prefer more text when dev NDCG ties: missing images then become harmless.
    selected = max(sweep, key=lambda row: (row["ndcg@10"], row["text_weight"]))
    return float(selected["text_weight"]), sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev-candidates",
        type=Path,
        default=ROOT / "data/amazon-c4-automotive/candidates/blair_dev_top100.jsonl",
    )
    parser.add_argument(
        "--test-candidates",
        type=Path,
        default=ROOT / "data/amazon-c4-automotive/candidates/blair_test_top100.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/amazon-c4-automotive/images/manifest.jsonl",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/amazon-c4-automotive/cache/clip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/current/metrics/amazon_c4_automotive_multimodal.json",
    )
    parser.add_argument(
        "--per-query-output",
        type=Path,
        default=ROOT / "data/amazon-c4-automotive/results/multimodal_per_query.jsonl",
    )
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--model-revision",
        default="c237dc49a33fc61debc9276459120b7eac67e7ef",
        help="Pinned official-model revision containing safetensors weights.",
    )
    parser.add_argument("--image-batch-size", type=int, default=256)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fusion-text-weight-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    args = parser.parse_args()

    started = time.time()
    manifest = _load_manifest(args.manifest)
    image_item_ids, image_embeddings = _load_or_encode_images(
        manifest,
        cache_dir=args.cache_dir,
        model_name=args.model,
        model_revision=args.model_revision,
        batch_size=args.image_batch_size,
        device=args.device,
    )
    image_lookup = {
        item_id: image_embeddings[index]
        for index, item_id in enumerate(image_item_ids)
    }
    dev_rows = _read_jsonl(args.dev_candidates)
    test_rows = _read_jsonl(args.test_candidates)
    all_rows = dev_rows + test_rows
    query_embeddings = _encode_texts(
        [str(row["query"]) for row in all_rows],
        model_name=args.model,
        model_revision=args.model_revision,
        batch_size=args.text_batch_size,
        device=args.device,
    )
    dev_embeddings = query_embeddings[: len(dev_rows)]
    test_embeddings = query_embeddings[len(dev_rows) :]

    selected_weight, sweep = _select_text_weight(
        dev_rows,
        dev_embeddings,
        image_lookup,
        args.fusion_text_weight_grid,
    )
    dev_metrics, dev_per_query = _score_queries(
        dev_rows,
        dev_embeddings,
        image_lookup,
        text_weight=selected_weight,
    )
    test_metrics, test_per_query = _score_queries(
        test_rows,
        test_embeddings,
        image_lookup,
        text_weight=selected_weight,
    )
    payload = {
        "protocol": {
            "version": 1,
            "dataset": "Amazon-C4 User Purchase History / Automotive",
            "candidate_source": "BLaIR top-100",
            "candidate_set_immutable": True,
            "positive_forced_into_candidates": False,
            "image_model": args.model,
            "image_model_revision": args.model_revision,
            "missing_image_policy": {
                "candidate_removed": False,
                "image_only": "rank after candidates with valid images",
                "late_fusion": "neutral zero after within-query z-score",
            },
            "fusion": "text_weight * z(BLaIR) + (1-text_weight) * z(CLIP-image)",
            "selection": "text_weight selected on dev NDCG@10; test evaluated once",
        },
        "counts": {
            "manifest_items": len(manifest),
            "valid_image_embeddings": len(image_lookup),
            "dev_queries": len(dev_rows),
            "test_queries": len(test_rows),
        },
        "selected_text_weight": selected_weight,
        "dev_weight_sweep": sweep,
        "dev": dev_metrics,
        "test": test_metrics,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(args.output, payload)
    _write_jsonl(args.per_query_output, [*dev_per_query, *test_per_query])
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
