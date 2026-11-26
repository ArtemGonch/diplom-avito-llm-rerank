#!/usr/bin/env python3
"""Recalculate the ten BX UsedCars metrics from the preview or normalized Parquet."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from avito_bx_baseline import audit_candidate_archive, audit_candidates, audit_preview, list_archive_members


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
    parser.add_argument("--archive", type=Path, help="ZIP containing normalized candidate Parquet")
    parser.add_argument("--list-members", action="store_true", help="List files in the ZIP without extracting")
    parser.add_argument(
        "--member", action="append", default=[], metavar="NAME",
        help="Candidate Parquet member with an embedded split. Repeatable.",
    )
    parser.add_argument(
        "--member-split", action="append", nargs=2, default=[], metavar=("SPLIT", "NAME"),
        help="Candidate Parquet member with an explicit split. Repeatable.",
    )
    args = parser.parse_args()
    if args.archive:
        if args.input or args.candidates or args.candidate_split:
            parser.error("Use either --archive or direct Parquet inputs")
        if args.list_members:
            if args.member or args.member_split or args.original:
                parser.error("--list-members cannot be combined with ranking inputs")
            result = list_archive_members(args.archive)
        elif args.member or args.member_split:
            result = audit_candidate_archive(
                args.archive, args.member, split_members=args.member_split, original=args.original,
            )
        else:
            parser.error("Choose --list-members, --member, or --member-split with --archive")
    elif args.list_members or args.member or args.member_split:
        parser.error("ZIP member options require --archive")
    elif args.candidates or args.candidate_split:
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
