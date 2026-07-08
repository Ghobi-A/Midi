"""Motif analysis: interval/rhythm/contour extraction, repeated-n-gram detection, and ranking.

Powers the "MIDI Motif Lab" supporting feature — given any parsed note
sequence (typically the generated melody), find the phrases that repeat,
and rank them by how strongly they read as a musical "motif".
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Sequence, Tuple

from .music_theory import Note


def extract_intervals(notes: Sequence[Note]) -> List[int]:
    """Return the semitone interval between each consecutive pair of notes."""
    return [b.pitch - a.pitch for a, b in zip(notes, notes[1:])]


def extract_rhythm(notes: Sequence[Note]) -> List[float]:
    """Return each note's duration in beats."""
    return [note.duration for note in notes]


def extract_inter_onset_intervals(notes: Sequence[Note]) -> List[float]:
    """Return the time (in beats) between each note's onset and the next."""
    return [b.start - a.start for a, b in zip(notes, notes[1:])]


def extract_contour(notes: Sequence[Note]) -> List[str]:
    """Return ``"U"``/``"D"``/``"R"`` (up/down/repeat) for each consecutive pitch pair."""
    contour = []
    for a, b in zip(notes, notes[1:]):
        if b.pitch > a.pitch:
            contour.append("U")
        elif b.pitch < a.pitch:
            contour.append("D")
        else:
            contour.append("R")
    return contour


def find_repeated_ngrams(sequence: Sequence, n: int) -> Dict[Tuple, List[int]]:
    """Return ``{ngram: [start_indices]}`` for every length-``n`` slice of ``sequence`` occurring 2+ times."""
    seen: Dict[Tuple, List[int]] = {}
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i : i + n])
        seen.setdefault(ngram, []).append(i)
    return {ngram: idxs for ngram, idxs in seen.items() if len(idxs) >= 2}


@dataclass(frozen=True)
class Motif:
    """A candidate motif: a repeated melodic or rhythmic n-gram plus its scoring metadata."""

    start_indices: Tuple[int, ...]
    length: int
    intervals: Tuple[int, ...]
    durations: Tuple[float, ...]
    notes: Tuple[Note, ...]
    repetition_count: int
    kind: str = "melodic"
    score: float = 0.0


def detect_melodic_motifs(notes: Sequence[Note], lengths: Sequence[int] = range(3, 7)) -> List[Motif]:
    """Detect repeated interval-sequence ("melodic") n-grams."""
    intervals = extract_intervals(notes)
    motifs: List[Motif] = []
    for n in lengths:
        for ngram, idxs in find_repeated_ngrams(intervals, n).items():
            first = idxs[0]
            motif_notes = tuple(notes[first : first + n + 1])
            motifs.append(
                Motif(
                    start_indices=tuple(idxs),
                    length=n + 1,
                    intervals=ngram,
                    durations=tuple(note.duration for note in motif_notes),
                    notes=motif_notes,
                    repetition_count=len(idxs),
                    kind="melodic",
                )
            )
    return motifs


def detect_rhythmic_motifs(notes: Sequence[Note], lengths: Sequence[int] = range(3, 7)) -> List[Motif]:
    """Detect repeated duration-sequence ("rhythmic") n-grams."""
    durations = extract_rhythm(notes)
    motifs: List[Motif] = []
    for n in lengths:
        for ngram, idxs in find_repeated_ngrams(durations, n).items():
            first = idxs[0]
            motif_notes = tuple(notes[first : first + n])
            motifs.append(
                Motif(
                    start_indices=tuple(idxs),
                    length=n,
                    intervals=tuple(extract_intervals(motif_notes)),
                    durations=ngram,
                    notes=motif_notes,
                    repetition_count=len(idxs),
                    kind="rhythmic",
                )
            )
    return motifs


def _rhythmic_clarity(durations: Sequence[float]) -> float:
    """Higher when a motif uses few distinct duration values (a clear, simple rhythm)."""
    if not durations:
        return 0.0
    return 1.0 / len(set(durations))


def _salience(motif: Motif) -> float:
    """Reward a wider melodic span and a moderate (neither trivial nor sprawling) phrase length."""
    pitch_span = max((abs(i) for i in motif.intervals), default=0)
    length_factor = max(1.0 - abs(motif.length - 4) / 8.0, 0.1)
    return length_factor * (1.0 + min(pitch_span, 12) / 12.0)


def rank_motifs(motifs: Sequence[Motif]) -> List[Motif]:
    """Score motifs by repetition count, phrase length, rhythmic clarity, and salience; sort best-first."""
    ranked = []
    for motif in motifs:
        clarity = _rhythmic_clarity(motif.durations)
        salience = _salience(motif)
        score = motif.repetition_count * 2.0 + motif.length * 0.5 + clarity * 2.0 + salience * 1.5
        ranked.append(replace(motif, score=score))
    return sorted(ranked, key=lambda m: m.score, reverse=True)


def detect_motifs(notes: Sequence[Note], lengths: Sequence[int] = range(3, 7)) -> List[Motif]:
    """Detect both melodic and rhythmic motifs in ``notes`` and return them ranked best-first."""
    combined = detect_melodic_motifs(notes, lengths) + detect_rhythmic_motifs(notes, lengths)
    return rank_motifs(combined)


__all__ = [
    "Motif",
    "extract_intervals",
    "extract_rhythm",
    "extract_inter_onset_intervals",
    "extract_contour",
    "find_repeated_ngrams",
    "detect_melodic_motifs",
    "detect_rhythmic_motifs",
    "rank_motifs",
    "detect_motifs",
]
