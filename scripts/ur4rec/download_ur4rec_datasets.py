#!/usr/bin/env python3
"""Download public UR4Rec benchmark datasets (paper Table 1)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ur4rec.data.amazon_books import download_amazon_books
from ur4rec.data.ml1m import download_movielens_1m
from ur4rec.data.steam import download_steam

DATASETS = {
    "movielens-1m": {
        "dir": "data/movielens-1m",
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "fn": lambda p: download_movielens_1m(p),
    },
    "amazon-books": {
        "dir": "data/amazon-books",
        "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/reviews_Books_5.json.gz",
        "fn": lambda p: download_amazon_books(p),
    },
    "steam": {
        "dir": "data/steam",
        "url": "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
        "fn": lambda p: download_steam(p),
    },
}


def _ts() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    n = path.stat().st_size
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def print_status() -> None:
    checks = {
        "movielens-1m": [
            ROOT / "data/movielens-1m/ml-1m/ratings.dat",
            ROOT / "data/movielens-1m/ml-1m/movies.dat",
        ],
        "amazon-books": [
            ROOT / "data/amazon-books/reviews_Books_5.json.gz",
            ROOT / "data/amazon-books/meta_Books.json.gz",
        ],
        "steam": [
            ROOT / "data/steam/steam_games.json.gz",
            ROOT / "data/steam/steam_reviews.json.gz",
        ],
    }
    print(f"Dataset status ({_ts()})\n")
    for name, files in checks.items():
        ok = all(p.exists() for p in files)
        mark = "OK" if ok else "INCOMPLETE"
        print(f"  [{mark}] {name}")
        for p in files:
            part = p.with_suffix(p.suffix + ".part")
            line = f"    {p.name}: {_fmt_size(p)}"
            if part.exists():
                line += f"  (partial: {_fmt_size(part)})"
            print(line)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UR4Rec public datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["all"],
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Append download progress to this log file (stdout is always used)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print which dataset files exist and exit",
    )
    args = parser.parse_args()
    if args.status:
        print_status()
        return
    names = list(DATASETS.keys()) if "all" in args.datasets else args.datasets

    print("UR4Rec paper datasets (Zhang et al., COLING 2025, Table 1):\n")
    for name in names:
        spec = DATASETS[name]
        print(f"  [{name}]")
        print(f"    dir: {ROOT / spec['dir']}")
        print(f"    ref: {spec['url']}")
    print()

    log_fp = None
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(args.log_file, "a", encoding="utf-8")
        print(f"Logging to {args.log_file.resolve()}\n", flush=True)

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data: str) -> None:
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self) -> None:
            for s in self.streams:
                s.flush()

    old_stdout = sys.stdout
    if log_fp is not None:
        sys.stdout = _Tee(old_stdout, log_fp)

    try:
        for name in names:
            spec = DATASETS[name]
            out = ROOT / spec["dir"]
            print(f"==> {name}  ({_ts()})", flush=True)
            path = spec["fn"](out)
            print(f"    ready: {path}\n", flush=True)
    finally:
        sys.stdout = old_stdout
        if log_fp is not None:
            log_fp.close()


if __name__ == "__main__":
    main()
