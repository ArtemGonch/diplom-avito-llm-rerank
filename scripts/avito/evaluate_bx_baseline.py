#!/usr/bin/env python3
"""Recalculate all ten BX UsedCars preview metrics from the original Parquet."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from avito_bx_baseline import audit_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original BX Parquet")
    args = parser.parse_args()
    result = audit_preview(args.input)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
