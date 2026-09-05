"""Statistically explicit aggregation helpers for experiment reports."""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Dict, Iterable


def aggregate(values: Iterable[float]) -> Dict[str, float | int]:
    """Return mean, sample standard deviation and sample count.

    Non-finite values are excluded rather than silently poisoning a report.
    A singleton has zero observed spread; an empty input has ``NaN`` values.
    """
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": mean(clean),
        "std": stdev(clean) if len(clean) > 1 else 0.0,
        "n": len(clean),
    }


__all__ = ["aggregate"]
