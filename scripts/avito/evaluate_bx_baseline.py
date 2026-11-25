#!/usr/bin/env python3
"""Recalculate the ten BX UsedCars metrics from the preview or normalized Parquet."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from avito_bx_baseline import audit_candidates, audit_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Original SHA-256-pinned BX preview Parquet")
    parser.add_argument(
        "--candidates", type=Path, action="extend", nargs="+", default=[], metavar="PARQUET",
        help="Normalized candidate files with an embedded split column. Repeatable.",
    )
    parser.add_argument(
        "--candidate-split", action="append", nargs=2, default=[], metavar=("SPLIT", "PARQUET"),
        help="Normalized candidate file with an explicit split. Repeatable.",
    )
    parser.add_argument("--original", type=Path, help="Original preview for normalized pilot comparison")
    args = parser.parse_args()
    if args.candidates or args.candidate_split:
        if args.input:
            parser.error("Use --original with normalized candidates")
        result = audit_candidates(
            args.candidates,
            split_paths=[(split, Path(path)) for split, path in args.candidate_split],
            original=args.original,
        )
    elif args.input and not args.original:
        result = audit_preview(args.input)
    else:
        parser.error("Use --input or supply normalized candidates")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
