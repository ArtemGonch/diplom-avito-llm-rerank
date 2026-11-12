#!/usr/bin/env python3
"""Produce identifier-free aggregate diagnostics from existing Avito artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from avito_user_view.core import verify_archive
from avito_user_view.evaluation import annotation_report, temporal_search_report
from avito_user_view.serp_contract import inspect_serps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    temporal = sub.add_parser("search-holdout")
    temporal.add_argument("--input-dir", type=Path, required=True)
    annotation = sub.add_parser("annotations")
    annotation.add_argument("--key", type=Path, required=True)
    annotation.add_argument(
        "--confirm-same-bundle",
        required=True,
        action="store_true",
        help="Confirm that every legacy CSV came from the bundle matching this key",
    )
    annotation.add_argument("answers", metavar="EXPERT=CSV", nargs="+")
    preflight = sub.add_parser("serp-preflight")
    preflight.add_argument("--input", type=Path, required=True, help="Normalized JSONL")
    args = parser.parse_args()
    if args.mode == "search-holdout":
        manifest = verify_archive(args.input_dir)
        if not any(x["path"] == "dataset/users.json" for x in manifest["files"]):
            parser.error("dataset/users.json is not in the verified manifest")
        source = (args.input_dir / "dataset/users.json").read_bytes()
        result = temporal_search_report(json.loads(source))
        result["input_sha256"] = hashlib.sha256(source).hexdigest()
        result["manifest_verified_files"] = len(manifest["files"])
    elif args.mode == "annotations":
        pairs = []
        for arg in args.answers:
            expert, separator, path = arg.partition("=")
            if not separator or not expert or not path:
                parser.error("Provide one EXPERT=CSV for each expert")
            pairs.append((expert, Path(path).resolve()))
        if len({p for _, p in pairs}) != len(pairs):
            parser.error("Each expert export must be supplied at most once")
        keys = json.loads(args.key.read_text(encoding="utf-8"))
        with ExitStack() as stack:
            files = [
                (expert, stack.enter_context(p.open(encoding="utf-8-sig", newline="")))
                for expert, p in pairs
            ]
            result = annotation_report(keys, files)
    else:
        with args.input.open(encoding="utf-8") as f:
            result = inspect_serps(json.loads(line) for line in f if line.strip())
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
