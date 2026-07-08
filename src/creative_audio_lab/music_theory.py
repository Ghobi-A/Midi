"""Core music-theory primitives shared across the generation pipeline.

Defines the canonical :class:`Note` event type, key/scale handling, and
diatonic chord construction used by every generator and analysis module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Note names
# ---------------------------------------------------------------------------

NOTE_NAME_TO_PITCH_CLASS: Dict[str, int] = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "F": 5, "E#": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11,
}

PITCH_CLASS_NAMES_SHARP: List[str] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
]


def note_name_to_pitch_class(name: str) -> int:
    """Convert a key/root name like ``"C"``, ``"F#"`` or ``"Eb"`` to a pitch class (0-11)."""
    key = name.strip().upper()
    if key not in NOTE_NAME_TO_PITCH_CLASS:
        raise ValueError(f"Unknown note name: {name!r}")
    return NOTE_NAME_TO_PITCH_CLASS[key]


def midi_note(pitch_class: int, octave: int) -> int:
    """Return the MIDI note number for ``pitch_class`` (0-11) in ``octave`` (MIDI octave, C4=60)."""
    return pitch_class + (octave + 1) * 12


# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------

# Semitone offsets from the root, one full octave.
SCALES: Dict[str, Tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
}

SCALE_ALIASES: Dict[str, str] = {
    "major": "major",
    "ionian": "major",
    "minor": "natural_minor",
    "natural minor": "natural_minor",
    "natural_minor": "natural_minor",
    "aeolian": "natural_minor",
    "harmonic minor": "harmonic_minor",
    "harmonic_minor": "harmonic_minor",
    "dorian": "dorian",
    "phrygian": "phrygian",
}


def normalize_mode(mode: str) -> str:
    """Map a user-facing mode name (e.g. ``"harmonic minor"``) to a canonical :data:`SCALES` key."""
    key = mode.strip().lower()
    if key not in SCALE_ALIASES:
        raise ValueError(f"Unknown scale/mode: {mode!r}")
    return SCALE_ALIASES[key]


def scale_pitch_classes(key: str, mode: str) -> List[int]:
    """Return the seven pitch classes (0-11) of ``key``/``mode``, root first."""
    root = note_name_to_pitch_class(key)
    intervals = SCALES[normalize_mode(mode)]
    return [(root + interval) % 12 for interval in intervals]


# ---------------------------------------------------------------------------
# Note events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Note:
    """A single symbolic note event.

    Attributes
    ----------
    start:
        Onset time in beats from the start of the arrangement.
    pitch:
        MIDI note number (0-127).
    duration:
        Length in beats.
    velocity:
        MIDI velocity (1-127).
    """

    start: float
    pitch: int
    duration: float
    velocity: int = 96

    def end(self) -> float:
        return self.start + self.duration


# ---------------------------------------------------------------------------
# Diatonic chords
# ---------------------------------------------------------------------------

CHORD_QUALITY_INTERVALS: Dict[str, Tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
}


def _quality_from_intervals(intervals: Tuple[int, int, int]) -> str:
    third, fifth = intervals[1], intervals[2]
    if third == 4 and fifth == 7:
        return "maj"
    if third == 3 and fifth == 7:
        return "min"
    if third == 3 and fifth == 6:
        return "dim"
    if third == 4 and fifth == 8:
        return "aug"
    # Fallback for unusual stacks (shouldn't occur for the supported modes).
    return "maj"


@dataclass(frozen=True)
class Chord:
    """A diatonic triad expressed as pitch classes relative to the tonic."""

    degree: int  # 1-indexed scale degree
    root_pc: int  # pitch class of the chord root (0-11)
    quality: str  # "maj" | "min" | "dim" | "aug"

    @property
    def interval_notes(self) -> Tuple[int, int, int]:
        """Semitone offsets of root/third/fifth relative to ``root_pc``."""
        return CHORD_QUALITY_INTERVALS[self.quality]

    def pitch_classes(self) -> Tuple[int, int, int]:
        return tuple((self.root_pc + i) % 12 for i in self.interval_notes)


def diatonic_chord(key: str, mode: str, degree: int) -> Chord:
    """Build the diatonic triad on ``degree`` (1-7) of ``key``/``mode`` by stacking thirds."""
    if not 1 <= degree <= 7:
        raise ValueError("degree must be between 1 and 7")
    pcs = scale_pitch_classes(key, mode)
    root_idx = degree - 1
    third_idx = (degree - 1 + 2) % 7
    fifth_idx = (degree - 1 + 4) % 7

    root = pcs[root_idx]
    # Reconstruct absolute semitone distance (not just pitch class) so we can
    # correctly classify the triad quality even when it wraps the octave.
    third_semitones = (pcs[third_idx] - root) % 12
    fifth_semitones = (pcs[fifth_idx] - root) % 12
    quality = _quality_from_intervals((0, third_semitones, fifth_semitones))
    return Chord(degree=degree, root_pc=root, quality=quality)


def chord_notes(chord: Chord, octave: int = 3) -> List[int]:
    """Return absolute MIDI note numbers for ``chord`` voiced in root position starting at ``octave``."""
    root_note = midi_note(chord.root_pc, octave)
    return [root_note + i for i in chord.interval_notes]


def nearest_scale_tone(pitch: int, scale_pcs: Sequence[int]) -> int:
    """Return the MIDI pitch closest to ``pitch`` whose pitch class is in ``scale_pcs``."""
    if pitch % 12 in scale_pcs:
        return pitch
    for distance in range(1, 7):
        if (pitch - distance) % 12 in scale_pcs:
            return pitch - distance
        if (pitch + distance) % 12 in scale_pcs:
            return pitch + distance
    return pitch


__all__ = [
    "Note",
    "Chord",
    "SCALES",
    "SCALE_ALIASES",
    "note_name_to_pitch_class",
    "midi_note",
    "normalize_mode",
    "scale_pitch_classes",
    "diatonic_chord",
    "chord_notes",
    "nearest_scale_tone",
    "CHORD_QUALITY_INTERVALS",
    "PITCH_CLASS_NAMES_SHARP",
]
