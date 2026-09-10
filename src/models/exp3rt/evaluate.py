"""Compute RMSE/MAE from Exp3RT inference output."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    pairs = [(a, p) for a, p in zip(actual, predicted) if p >= 0]
    if not pairs:
        return {"rmse": float("inf"), "mae": float("inf"), "n": 0}
    n = len(pairs)
    mse = sum((a - p) ** 2 for a, p in pairs) / n
    mae = sum(abs(a - p) for a, p in pairs) / n
    return {"rmse": math.sqrt(mse), "mae": mae, "n": n}


def evaluate_rating_prediction(test_json: Path, result_json: Path) -> dict[str, Any]:
    test_data = json.loads(Path(test_json).read_text(encoding="utf-8"))
    result_data = json.loads(Path(result_json).read_text(encoding="utf-8"))
    actual = [float(row["score"]) for row in test_data]
    max_prob = [float(result_data[str(i)]["max_prob_rating"]) for i in range(len(test_data))]
    expected = [float(result_data[str(i)]["expected_rating"]) for i in range(len(test_data))]
    return {
        "max_prob_rating": _metrics(actual, max_prob),
        "expected_rating": _metrics(actual, expected),
    }
