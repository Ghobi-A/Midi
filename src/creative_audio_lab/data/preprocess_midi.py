"""Preprocessing primitives for the future ML training pipeline.

Quantization, transposition, and range filtering are the kind of
normalization steps a training pipeline applies before feature extraction
or model input. They're implemented now, ahead of any model, so the
pipeline described in docs/ML_ROADMAP.md has real building blocks to
start from.
"""

from __future__ import annotations

from typing import List, Sequence

from ..music_theory import Note


def quantize_notes(notes: Sequence[Note], grid: float = 0.25) -> List[Note]:
    """Snap each note's onset to the nearest multiple of ``grid`` beats."""
    quantized = []
    for note in notes:
        snapped_start = round(note.start / grid) * grid
        quantized.append(Note(start=snapped_start, pitch=note.pitch, duration=note.duration, velocity=note.velocity))
    return quantized


def transpose_notes(notes: Sequence[Note], semitones: int) -> List[Note]:
    """Shift every note by a fixed number of semitones, clamped to the valid MIDI range."""
    return [
        Note(start=note.start, pitch=min(max(note.pitch + semitones, 0), 127), duration=note.duration, velocity=note.velocity)
        for note in notes
    ]


def filter_pitch_range(notes: Sequence[Note], low: int, high: int) -> List[Note]:
    """Keep only notes whose pitch falls within ``[low, high]`` inclusive."""
    return [note for note in notes if low <= note.pitch <= high]


__all__ = ["quantize_notes", "transpose_notes", "filter_pitch_range"]
