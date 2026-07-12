"""Compare generation backends on the same fixed prompts.

The point of the Stage 2 n-gram baseline (docs/STAGE3_PLAN.md, section 5)
is that every backend answers the *same* held-out prompts and gets scored
by the *same* metrics, so "beats the baseline" is a number, not an
impression. This harness is that comparison: one row of averaged metrics
per backend.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Sequence

from ..models.base import GenerationBackend
from .metrics import controls_adherence, evaluate_arrangement
from .prompts import HELD_OUT_PROMPTS


def compare_backends(
    backends: Sequence[GenerationBackend],
    prompts: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Run every backend over every prompt and average the metric suite.

    Returns ``{backend name: {metric name: mean value}}``, combining
    :func:`~creative_audio_lab.evaluation.metrics.evaluate_arrangement`
    and :func:`~creative_audio_lab.evaluation.metrics.controls_adherence`.
    Prompts default to the fixed held-out list.
    """
    prompt_list = list(prompts if prompts is not None else HELD_OUT_PROMPTS)
    if not prompt_list:
        raise ValueError("compare_backends needs at least one prompt")

    results: Dict[str, Dict[str, float]] = {}
    for backend in backends:
        totals: Dict[str, float] = defaultdict(float)
        for prompt in prompt_list:
            arrangement = backend.generate(prompt)
            scores = {**evaluate_arrangement(arrangement), **controls_adherence(arrangement)}
            for metric, value in scores.items():
                totals[metric] += value
        results[backend.name] = {metric: total / len(prompt_list) for metric, total in totals.items()}
    return results


__all__ = ["compare_backends"]
