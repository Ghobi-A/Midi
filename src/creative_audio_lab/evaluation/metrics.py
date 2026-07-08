"""Lightweight evaluation metrics for generated MIDI.

None of these require a trained model or external corpus — they are simple,
explainable statistics intended as an honest baseline that a future
ML-based evaluator (e.g. a learned harmonic-fit or usability model) can be
compared against.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Sequence, Tuple, TYPE_CHECKING

from ..motif_detection import extract_intervals
from ..music_theory import Note, scale_pitch_classes

if TYPE_CHECKING:  # pragma: no cover
    from ..generators.arrangement import Arrangement
    from ..generators.chords import ChordEvent


def note_density(notes: Sequence[Note], bars: int) -> float:
    """Average number of notes per bar."""
    if bars <= 0:
        return 0.0
    return len(notes) / bars


def pitch_range(notes: Sequence[Note]) -> Tuple[int, int]:
    """Return ``(lowest_pitch, highest_pitch)``, or ``(0, 0)`` for an empty sequence."""
    if not notes:
        return (0, 0)
    pitches = [note.pitch for note in notes]
    return (min(pitches), max(pitches))


def repetition_score(notes: Sequence[Note], n: int = 4) -> float:
    """Fraction of length-``n`` interval n-grams that repeat an earlier one."""
    intervals = extract_intervals(notes)
    if len(intervals) < n:
        return 0.0
    total = len(intervals) - n + 1
    seen = set()
    repeated = 0
    for i in range(total):
        ngram = tuple(intervals[i : i + n])
        if ngram in seen:
            repeated += 1
        else:
            seen.add(ngram)
    return repeated / total


def scale_adherence_score(notes: Sequence[Note], scale_pcs: Sequence[int]) -> float:
    """Fraction of notes whose pitch class belongs to ``scale_pcs``."""
    if not notes:
        return 1.0
    in_scale = sum(1 for note in notes if note.pitch % 12 in scale_pcs)
    return in_scale / len(notes)


def _chord_at(beat: float, chord_events: Sequence["ChordEvent"]):
    for event in chord_events:
        if event.start_beat <= beat < event.start_beat + event.duration_beats:
            return event.chord
    return chord_events[-1].chord if chord_events else None


def harmonic_fit_score(notes: Sequence[Note], chord_events: Sequence["ChordEvent"]) -> float:
    """Fraction of notes whose pitch class is a chord tone of the chord sounding at their onset."""
    if not notes or not chord_events:
        return 1.0
    fits = 0
    for note in notes:
        chord = _chord_at(note.start, chord_events)
        if chord is not None and note.pitch % 12 in chord.pitch_classes():
            fits += 1
    return fits / len(notes)


def _normalized_entropy(values: Sequence) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    if len(counts) <= 1:
        return 0.0
    return entropy / math.log2(len(counts))


def rhythmic_complexity_score(notes: Sequence[Note]) -> float:
    """Normalized Shannon entropy of the note-duration distribution (0 = uniform, 1 = maximally varied)."""
    return _normalized_entropy([note.duration for note in notes])


def novelty_score(notes: Sequence[Note]) -> float:
    """Normalized Shannon entropy of the melodic-interval distribution: how unpredictable the line is."""
    return _normalized_entropy(extract_intervals(notes))


def motif_retention_score(original: Sequence[Note], variation: Sequence[Note], n: int = 3) -> float:
    """Fraction of ``original``'s length-``n`` interval n-grams still present in ``variation``."""
    original_intervals = extract_intervals(original)
    variation_intervals = extract_intervals(variation)
    if len(original_intervals) < n:
        return 1.0 if original_intervals == variation_intervals else 0.0

    original_ngrams = {tuple(original_intervals[i : i + n]) for i in range(len(original_intervals) - n + 1)}
    if len(variation_intervals) < n:
        variation_ngrams = set()
    else:
        variation_ngrams = {tuple(variation_intervals[i : i + n]) for i in range(len(variation_intervals) - n + 1)}

    if not original_ngrams:
        return 1.0
    return len(original_ngrams & variation_ngrams) / len(original_ngrams)


def evaluate_arrangement(arrangement: "Arrangement") -> Dict[str, float]:
    """Compute the full evaluation-metric suite for a generated :class:`Arrangement`.

    Metrics are computed on the melody line (the most analysis-relevant
    part) plus an overall note density across every part.
    """
    melody = arrangement.parts.get("melody", [])
    all_notes = [note for notes in arrangement.parts.values() for note in notes]
    scale_pcs = scale_pitch_classes(arrangement.controls.key, arrangement.controls.mode)
    low, high = pitch_range(melody)

    return {
        "note_density": note_density(all_notes, arrangement.bars),
        "pitch_min": low,
        "pitch_max": high,
        "pitch_range": high - low,
        "repetition_score": repetition_score(melody),
        "harmonic_fit_score": harmonic_fit_score(melody, arrangement.chord_events),
        "rhythmic_complexity_score": rhythmic_complexity_score(melody),
        "novelty_score": novelty_score(melody),
        "scale_adherence_score": scale_adherence_score(melody, scale_pcs),
    }


__all__ = [
    "note_density",
    "pitch_range",
    "repetition_score",
    "scale_adherence_score",
    "harmonic_fit_score",
    "rhythmic_complexity_score",
    "novelty_score",
    "motif_retention_score",
    "evaluate_arrangement",
]
