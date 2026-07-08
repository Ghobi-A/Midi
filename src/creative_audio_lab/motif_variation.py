"""Generate variations of a motif: transposition, inversion, rhythmic augmentation/
compression, passing-tone insertion, and call-and-response continuations.
"""

from __future__ import annotations

from typing import List, Sequence

from .export.midi_export import export_notes_to_bytes
from .music_theory import Note, nearest_scale_tone


def transpose(notes: Sequence[Note], semitones: int) -> List[Note]:
    """Shift every note in ``notes`` by ``semitones``."""
    return [Note(start=n.start, pitch=n.pitch + semitones, duration=n.duration, velocity=n.velocity) for n in notes]


def invert_contour(notes: Sequence[Note], axis_pitch: int | None = None) -> List[Note]:
    """Mirror the melodic contour around ``axis_pitch`` (defaults to the first note's pitch)."""
    if not notes:
        return []
    axis = axis_pitch if axis_pitch is not None else notes[0].pitch
    return [Note(start=n.start, pitch=2 * axis - n.pitch, duration=n.duration, velocity=n.velocity) for n in notes]


def augment_rhythm(notes: Sequence[Note], factor: float = 2.0) -> List[Note]:
    """Stretch onsets and durations by ``factor`` (> 1 slows the motif down)."""
    if not notes:
        return []
    origin = notes[0].start
    return [
        Note(start=origin + (n.start - origin) * factor, pitch=n.pitch, duration=n.duration * factor, velocity=n.velocity)
        for n in notes
    ]


def compress_rhythm(notes: Sequence[Note], factor: float = 0.5) -> List[Note]:
    """Shrink onsets and durations by ``factor`` (< 1 speeds the motif up)."""
    return augment_rhythm(notes, factor=factor)


def add_passing_tones(notes: Sequence[Note], scale_pcs: Sequence[int]) -> List[Note]:
    """Insert a short passing tone into any leap of a third or more between consecutive notes."""
    if len(notes) < 2:
        return list(notes)
    result: List[Note] = [notes[0]]
    for note in notes[1:]:
        prev = result[-1]
        interval = note.pitch - prev.pitch
        if abs(interval) >= 3 and note.duration >= 0.5:
            direction = 1 if interval > 0 else -1
            passing_pitch = nearest_scale_tone(prev.pitch + direction * 2, scale_pcs)
            half = note.duration / 2
            result.append(Note(start=note.start, pitch=passing_pitch, duration=half, velocity=max(note.velocity - 15, 40)))
            result.append(Note(start=note.start + half, pitch=note.pitch, duration=note.duration - half, velocity=note.velocity))
        else:
            result.append(note)
    return result


def call_and_response(notes: Sequence[Note], transpose_semitones: int = -5) -> List[Note]:
    """Append a transposed, slightly softer "response" phrase immediately after the motif ends."""
    if not notes:
        return []
    first_start = notes[0].start
    motif_end = max(n.end() for n in notes)
    response = transpose(notes, transpose_semitones)
    shifted = [
        Note(start=n.start - first_start + motif_end, pitch=n.pitch, duration=n.duration, velocity=max(n.velocity - 10, 40))
        for n in response
    ]
    return list(notes) + shifted


def export_variation(notes: Sequence[Note], name: str = "motif_variation", program: int = 0, bpm: float = 100.0) -> bytes:
    """Serialize a motif variation as standalone MIDI bytes, ready for a download button or disk write."""
    return export_notes_to_bytes(list(notes), name=name, program=program, bpm=bpm)


__all__ = [
    "transpose",
    "invert_contour",
    "augment_rhythm",
    "compress_rhythm",
    "add_passing_tones",
    "call_and_response",
    "export_variation",
]
