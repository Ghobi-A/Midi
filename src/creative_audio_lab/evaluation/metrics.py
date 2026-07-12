"""Lightweight evaluation metrics for generated MIDI.

None of these require a trained model — they are simple, explainable
statistics intended as an honest baseline that a future ML-based evaluator
(e.g. a learned harmonic-fit or usability model) can be compared against.
The reference-based metrics (:func:`distribution_similarity_scores`,
:func:`plagiarism_score`) compare against any note sets you supply — a
held-out corpus split, training pieces — but never load or require one
themselves. Held-out token perplexity deliberately does *not* live here:
it requires a trained model, and this module's contract is that nothing in
it does (see docs/STAGE3_PLAN.md, section 4).
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


def _histogram_overlap(a_values: Sequence, b_values: Sequence) -> float:
    """Overlap of two empirical distributions: 1.0 = identical, 0.0 = disjoint.

    Both value sequences are treated as categorical (durations should already
    be grid-quantized, which everything the tokenizer touches is).
    """
    if not a_values or not b_values:
        return 0.0
    a_counts = Counter(a_values)
    b_counts = Counter(b_values)
    a_total = len(a_values)
    b_total = len(b_values)
    return sum(
        min(a_counts[key] / a_total, b_counts[key] / b_total)
        for key in a_counts.keys() | b_counts.keys()
    )


def distribution_similarity_scores(notes: Sequence[Note], reference: Sequence[Note]) -> Dict[str, float]:
    """Compare ``notes`` against a reference note set on three distributions.

    Returns per-feature histogram overlaps (pitch class, melodic interval,
    duration), each in ``[0, 1]``. This is what "statistically resembles the
    reference material" means for a generated set: pass held-out corpus
    notes as ``reference`` and expect scores inside the band that held-out
    corpus pieces score against *each other*.
    """
    return {
        "pitch_class_similarity": _histogram_overlap(
            [note.pitch % 12 for note in notes], [note.pitch % 12 for note in reference]
        ),
        "interval_similarity": _histogram_overlap(extract_intervals(notes), extract_intervals(reference)),
        "duration_similarity": _histogram_overlap(
            [note.duration for note in notes], [note.duration for note in reference]
        ),
    }


def plagiarism_score(candidate: Sequence[Note], corpus_pieces: Sequence[Sequence[Note]], n: int = 3) -> float:
    """How much of ``candidate``'s melodic material appears in its closest corpus piece.

    Reuses :func:`motif_retention_score` with the roles swapped: the score is
    the *maximum* fraction of the candidate's interval n-grams found in any
    single corpus piece. Near 1.0 means the candidate is substantially a copy
    of one training piece — a memorization (and, for corpora of copyrighted
    works, a rights) red flag, whereas every other metric here would score a
    perfect copy flatteringly.
    """
    if not corpus_pieces:
        return 0.0
    return max(motif_retention_score(candidate, piece, n=n) for piece in corpus_pieces)


# Acceptable melody notes-per-bar per requested density. Derived from the
# deterministic generator's RHYTHM_TEMPLATES (3/5/7 slots per bar for
# low/medium/high) plus headroom for the passing tones it inserts at
# medium/high density. Bands overlap deliberately — they express tolerance,
# not a classifier.
DENSITY_BANDS: Dict[str, Tuple[float, float]] = {
    "low": (1.0, 4.5),
    "medium": (3.5, 8.5),
    "high": (5.5, float("inf")),
}


def controls_adherence(arrangement: "Arrangement") -> Dict[str, float]:
    """Score an arrangement against its *own* requested controls.

    This is the style-conditioning check for any generative backend: did
    "sad ballad, low density" actually come out sparse and in the requested
    scale. Deterministic output satisfies these by construction; a learned
    model has to earn them.
    """
    melody = arrangement.parts.get("melody", [])
    controls = arrangement.controls
    scale_pcs = scale_pitch_classes(controls.key, controls.mode)
    notes_per_bar = note_density(melody, arrangement.bars)
    low, high = DENSITY_BANDS.get(controls.density, DENSITY_BANDS["medium"])

    return {
        "scale_adherence": scale_adherence_score(melody, scale_pcs),
        "melody_notes_per_bar": notes_per_bar,
        "density_in_band": 1.0 if low <= notes_per_bar <= high else 0.0,
    }


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
    "distribution_similarity_scores",
    "plagiarism_score",
    "DENSITY_BANDS",
    "controls_adherence",
    "evaluate_arrangement",
]
