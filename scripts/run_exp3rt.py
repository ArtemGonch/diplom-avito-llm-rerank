#!/usr/bin/env python3
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent / "exp3rt" / "run_exp3rt.py"), run_name="__main__")
