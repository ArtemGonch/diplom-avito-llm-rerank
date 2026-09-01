#!/usr/bin/env python3
"""Build and optionally download the fixed Amazon-C4 candidate image manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.amazon_c4 import image_filename, select_main_image_url  # noqa: E402


def _load_candidate_membership(paths: Iterable[Path]) -> tuple[set[str], dict[str, set[str]]]:
    requested: set[str] = set()
    by_split: dict[str, set[str]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for line_number, line in enumerate(path.open("r", encoding="utf-8"), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            split = str(row["split"])
            split_items = by_split.setdefault(split, set())
            for candidate in row["candidates"]:
                item_id = str(candidate["item_id"])
                split_items.add(item_id)
                requested.add(item_id)
    return requested, by_split


def _read_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.open("r", encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            out[str(row["item_id"])] = row
    return out


def _write_manifest(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item_id in sorted(rows):
            handle.write(json.dumps(rows[item_id], ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def materialise_manifest(
    requested: set[str],
    by_split: dict[str, set[str]],
    metadata_gzip: Path,
    manifest_path: Path,
    image_dir: Path,
) -> dict[str, dict]:
    previous = _read_manifest(manifest_path)
    found: dict[str, dict] = {}
    started = time.time()
    with gzip.open(metadata_gzip, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            item_id = str(row.get("parent_asin", ""))
            if item_id not in requested:
                continue
            url = select_main_image_url(row.get("images"))
            old = previous.get(item_id, {})
            local_path = image_dir / image_filename(item_id)
            status = str(old.get("status", "pending" if url else "missing"))
            if local_path.exists():
                status = "ok"
            elif not url:
                status = "missing"
            found[item_id] = {
                "manifest_version": 1,
                "item_id": item_id,
                "url": url,
                "local_path": str(local_path.relative_to(ROOT)),
                "status": status,
                "content_type": old.get("content_type"),
                "bytes": old.get("bytes"),
                "width": old.get("width"),
                "height": old.get("height"),
                "attempts": int(old.get("attempts", 0)),
                "error": old.get("error"),
                "splits": sorted(split for split, ids in by_split.items() if item_id in ids),
            }
            if len(found) == len(requested):
                break
            if line_number % 250_000 == 0:
                print(
                    f"metadata rows={line_number} found={len(found)}/{len(requested)}",
                    flush=True,
                )
    for item_id in requested - found.keys():
        local_path = image_dir / image_filename(item_id)
        found[item_id] = {
            "manifest_version": 1,
            "item_id": item_id,
            "url": None,
            "local_path": str(local_path.relative_to(ROOT)),
            "status": "missing",
            "content_type": None,
            "bytes": None,
            "width": None,
            "height": None,
            "attempts": int(previous.get(item_id, {}).get("attempts", 0)),
            "error": "parent_asin absent from Automotive metadata",
            "splits": sorted(split for split, ids in by_split.items() if item_id in ids),
        }
    _write_manifest(manifest_path, found)
    print(
        f"Materialised {len(found)} rows in {time.time() - started:.1f}s: {manifest_path}",
        flush=True,
    )
    return found


def _download_one(
    row: dict,
    *,
    timeout: float,
    retries: int,
) -> tuple[str, dict]:
    item_id = str(row["item_id"])
    url = row.get("url")
    if not url:
        return item_id, {**row, "status": "missing"}
    destination = ROOT / row["local_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            with Image.open(destination) as image:
                image.verify()
            with Image.open(destination) as image:
                width, height = image.size
            return item_id, {
                **row,
                "status": "ok",
                "bytes": destination.stat().st_size,
                "width": width,
                "height": height,
                "error": None,
            }
        except (OSError, UnidentifiedImageError):
            destination.unlink(missing_ok=True)

    error = "unknown download failure"
    attempts = int(row.get("attempts", 0))
    for attempt in range(1, retries + 1):
        attempts += 1
        temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
        try:
            request = Request(
                str(url),
                headers={"User-Agent": "diplom-amazon-c4-image-downloader/1.0"},
            )
            with urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get_content_type()
                if not content_type.startswith("image/"):
                    raise ValueError(f"unexpected content-type={content_type}")
                payload = response.read()
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            with Image.open(io.BytesIO(payload)) as image:
                width, height = image.size
                image.convert("RGB").save(temporary, format="JPEG", quality=92, optimize=True)
            temporary.replace(destination)
            return item_id, {
                **row,
                "status": "ok",
                "content_type": content_type,
                "bytes": destination.stat().st_size,
                "width": width,
                "height": height,
                "attempts": attempts,
                "error": None,
            }
        except (HTTPError, URLError, TimeoutError, ValueError, OSError, UnidentifiedImageError) as exc:
            temporary.unlink(missing_ok=True)
            error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return item_id, {
        **row,
        "status": "failed",
        "attempts": attempts,
        "error": error,
    }


def download_manifest(
    manifest_path: Path,
    *,
    workers: int,
    timeout: float,
    retries: int,
    checkpoint_every: int,
    limit: int | None,
) -> dict[str, dict]:
    rows = _read_manifest(manifest_path)
    pending = [
        row
        for row in rows.values()
        if row.get("url") and row.get("status") != "ok"
    ]
    if limit is not None:
        pending = pending[:limit]
    print(f"Downloading {len(pending)} images with {workers} workers", flush=True)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, row, timeout=timeout, retries=retries): row["item_id"]
            for row in pending
        }
        for future in concurrent.futures.as_completed(futures):
            item_id, updated = future.result()
            rows[item_id] = updated
            completed += 1
            if completed % checkpoint_every == 0 or completed == len(pending):
                _write_manifest(manifest_path, rows)
                counts: dict[str, int] = {}
                for row in rows.values():
                    status = str(row["status"])
                    counts[status] = counts.get(status, 0) + 1
                print(f"downloaded={completed}/{len(pending)} status={counts}", flush=True)
    return rows


def write_coverage(
    path: Path,
    rows: dict[str, dict],
    by_split: dict[str, set[str]],
) -> dict:
    payload: dict = {
        "manifest_version": 1,
        "candidate_set_immutable": True,
        "missing_images_remove_candidates": False,
        "selection_policy": "variant=MAIN; large -> hi_res -> thumb",
        "overall": {},
        "by_split": {},
    }
    for name, item_ids in [("overall", set(rows)), *sorted(by_split.items())]:
        counts: dict[str, int] = {}
        for item_id in item_ids:
            status = str(rows[item_id]["status"])
            counts[status] = counts.get(status, 0) + 1
        entry = {
            "candidates": len(item_ids),
            "status": counts,
            "url_coverage": sum(bool(rows[item_id].get("url")) for item_id in item_ids) / len(item_ids),
            "download_coverage": sum(rows[item_id]["status"] == "ok" for item_id in item_ids) / len(item_ids),
        }
        if name == "overall":
            payload["overall"] = entry
        else:
            payload["by_split"][name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-files",
        type=Path,
        nargs="+",
        default=[
            ROOT / f"data/amazon-c4-automotive/candidates/blair_{split}_top100.jsonl"
            for split in ("train", "dev", "test")
        ],
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "data/amazon-reviews-2023/Automotive/meta_Automotive.jsonl.gz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/amazon-c4-automotive/images/manifest.jsonl",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=ROOT / "data/amazon-c4-automotive/images/files",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=ROOT / "data/amazon-c4-automotive/images/coverage.json",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--skip-materialize",
        action="store_true",
        help="Reuse an existing manifest instead of rescanning the metadata gzip.",
    )
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    requested, by_split = _load_candidate_membership(args.candidate_files)
    if args.skip_materialize:
        rows = _read_manifest(args.manifest)
        if set(rows) != requested:
            raise ValueError(
                "Existing manifest item ids do not match the fixed candidate set; "
                "rerun without --skip-materialize"
            )
    else:
        rows = materialise_manifest(
            requested,
            by_split,
            args.metadata,
            args.manifest,
            args.image_dir,
        )
    if args.download:
        rows = download_manifest(
            args.manifest,
            workers=args.workers,
            timeout=args.timeout,
            retries=args.retries,
            checkpoint_every=args.checkpoint_every,
            limit=args.limit,
        )
    write_coverage(args.coverage_output, rows, by_split)


if __name__ == "__main__":
    main()
