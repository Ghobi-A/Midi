"""Lightweight, dependency-free evaluation metrics for generated MIDI."""

from .metrics import (
    evaluate_arrangement,
    harmonic_fit_score,
    motif_retention_score,
    note_density,
    novelty_score,
    pitch_range,
    repetition_score,
    rhythmic_complexity_score,
    scale_adherence_score,
)
from .aggregation import aggregate
from .error_analysis import analyse_generation

__all__ = [
    "aggregate",
    "analyse_generation",
    "evaluate_arrangement",
    "harmonic_fit_score",
    "motif_retention_score",
    "note_density",
    "novelty_score",
    "pitch_range",
    "repetition_score",
    "rhythmic_complexity_score",
    "scale_adherence_score",
]
