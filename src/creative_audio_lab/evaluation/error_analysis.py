"""Interpretable failure-mode analysis for generated melody lines."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence

from ..music_theory import Note
from .metrics import repetition_score, rhythmic_complexity_score, scale_adherence_score


def analyse_generation(
    notes: Sequence[Note], *, scale_pcs: Sequence[int], expected_beats: float
) -> Dict[str, object]:
    """Flag concrete pathologies and return their underlying measurements."""
    intervals = [b.pitch - a.pitch for a, b in zip(notes, notes[1:])]
    durations = [note.duration for note in notes]
    pitch_counts = Counter(note.pitch for note in notes)
    length = max((note.end() for note in notes), default=0.0)
    repeated_ratio = (max(pitch_counts.values()) / len(notes)) if notes else 1.0
    step_ratio = (sum(abs(i) <= 2 for i in intervals) / len(intervals)) if intervals else 0.0
    leap_ratio = (sum(abs(i) >= 12 for i in intervals) / len(intervals)) if intervals else 0.0
    scale_fit = scale_adherence_score(notes, scale_pcs)
    rhythm_diversity = rhythmic_complexity_score(notes)
    repetition = repetition_score(notes)
    unique_pitch_ratio = len(pitch_counts) / len(notes) if notes else 0.0
    measurements = {
        "notes": len(notes), "actual_beats": length, "expected_beats": expected_beats,
        "dominant_pitch_ratio": repeated_ratio, "stepwise_ratio": step_ratio,
        "large_leap_ratio": leap_ratio, "scale_adherence": scale_fit,
        "rhythmic_diversity": rhythm_diversity, "repetition": repetition,
        "unique_pitch_ratio": unique_pitch_ratio,
    }
    flags = {
        "repeated_note_collapse": len(notes) >= 8 and repeated_ratio >= 0.65,
        "excessive_stepwise_motion": len(intervals) >= 8 and step_ratio >= 0.9,
        "excessive_leaps": len(intervals) >= 8 and leap_ratio >= 0.3,
        "tonal_drift": len(notes) >= 8 and scale_fit < 0.75,
        "rhythmic_collapse": len(durations) >= 8 and rhythm_diversity < 0.1,
        "motif_copying": len(notes) >= 12 and repetition >= 0.6,
        "premature_eos": expected_beats > 0 and length < expected_beats * 0.5,
        "pathological_sequence_length": expected_beats > 0 and length > expected_beats * 1.1,
        "low_diversity_generation": len(notes) >= 8 and unique_pitch_ratio < 0.2,
    }
    return {"flags": flags, "measurements": measurements}


__all__ = ["analyse_generation"]
