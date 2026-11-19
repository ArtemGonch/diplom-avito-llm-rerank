#!/usr/bin/env python3
"""Validate the November history bundle and evaluate fixed personalized controls."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from avito_bx_history import audit_history_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_history_bundle(args.original, args.archive), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
