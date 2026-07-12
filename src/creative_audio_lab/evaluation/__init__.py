"""Lightweight, dependency-free evaluation metrics for generated MIDI."""

from .comparison import compare_backends
from .metrics import (
    DENSITY_BANDS,
    controls_adherence,
    distribution_similarity_scores,
    evaluate_arrangement,
    harmonic_fit_score,
    motif_retention_score,
    note_density,
    novelty_score,
    pitch_range,
    plagiarism_score,
    repetition_score,
    rhythmic_complexity_score,
    scale_adherence_score,
)
from .prompts import HELD_OUT_PROMPTS

__all__ = [
    "DENSITY_BANDS",
    "HELD_OUT_PROMPTS",
    "compare_backends",
    "controls_adherence",
    "distribution_similarity_scores",
    "evaluate_arrangement",
    "harmonic_fit_score",
    "motif_retention_score",
    "note_density",
    "novelty_score",
    "pitch_range",
    "plagiarism_score",
    "repetition_score",
    "rhythmic_complexity_score",
    "scale_adherence_score",
]
