#!/usr/bin/env python3
"""Backward-compatible wrapper — use scripts/ur4rec/run_ur4rec.py."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "ur4rec" / "run_ur4rec.py"), run_name="__main__")
