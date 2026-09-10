#!/usr/bin/env python3
"""Compatibility entrypoint for the audited Exp3RT-style Avito diagnostic."""

from __future__ import annotations

import runpy
from pathlib import Path


TARGET = Path(__file__).resolve().parent / "exp3rt" / "exp3rt_avito_attribute_rerank.py"
runpy.run_path(str(TARGET), run_name="__main__")
