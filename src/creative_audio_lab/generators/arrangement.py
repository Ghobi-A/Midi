"""Combine chords, melody, bass, and drums into a full arrangement.

This is the orchestration layer: it decides General MIDI program numbers
from the requested instrument set/style and assembles every part produced
by the other generators into a single :class:`Arrangement` ready for export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..music_theory import Note
from ..prompt_parser import GenerationControls
from .bass import generate_bassline
from .chords import ChordEvent, generate_chord_progression
from .drums import generate_drum_pattern
from .melody import generate_melody

# General MIDI program numbers (0-indexed) for the instrument tags the
# prompt parser can produce.
INSTRUMENT_PROGRAMS: Dict[str, int] = {
    "piano": 0,
    "strings": 48,   # String Ensemble 1
    "brass": 61,     # Brass Section
    "bells": 14,     # Tubular Bells
    "flute": 73,     # Flute
    "taiko_drums": 116,  # Taiko Drum (melodic/ostinato colour, not the GM drum kit)
}

BASS_PROGRAM_STANDARD = 33  # Electric Bass (finger)
BASS_PROGRAM_SYNTH = 38     # Synth Bass 1
DEFAULT_PROGRAM = 0         # Acoustic Grand Piano


@dataclass
class Arrangement:
    """A complete generated arrangement, ready for evaluation and MIDI export."""

    controls: GenerationControls
    chord_events: List[ChordEvent]
    parts: Dict[str, List[Note]] = field(default_factory=dict)
    programs: Dict[str, int] = field(default_factory=dict)
    bpm: float = 100.0
    bars: int = 8
    ticks_per_beat: int = 480


def _select_program(instruments: List[str], candidates: tuple, default: int = DEFAULT_PROGRAM) -> int:
    for tag in candidates:
        if tag in instruments:
            return INSTRUMENT_PROGRAMS[tag]
    return default


def _chord_program(controls: GenerationControls) -> int:
    return _select_program(controls.instruments, ("strings", "piano", "brass"))


def _melody_program(controls: GenerationControls) -> int:
    return _select_program(controls.instruments, ("flute", "bells", "piano", "strings", "brass"))


def _bass_program(controls: GenerationControls) -> int:
    if controls.style in ("trap", "drill"):
        return BASS_PROGRAM_SYNTH
    return BASS_PROGRAM_STANDARD


def build_arrangement(controls: GenerationControls) -> Arrangement:
    """Generate a full, DAW-ready arrangement from resolved :class:`GenerationControls`."""
    chord_events, chord_notes = generate_chord_progression(controls)
    melody_notes = generate_melody(controls, chord_events)
    bass_notes = generate_bassline(controls, chord_events)
    drum_notes = generate_drum_pattern(controls, chord_events)

    parts = {
        "chords": chord_notes,
        "melody": melody_notes,
        "bass": bass_notes,
        "drums": drum_notes,
    }
    programs = {
        "chords": _chord_program(controls),
        "melody": _melody_program(controls),
        "bass": _bass_program(controls),
    }

    return Arrangement(
        controls=controls,
        chord_events=chord_events,
        parts=parts,
        programs=programs,
        bpm=controls.bpm,
        bars=controls.bars,
    )


__all__ = [
    "Arrangement",
    "INSTRUMENT_PROGRAMS",
    "BASS_PROGRAM_STANDARD",
    "BASS_PROGRAM_SYNTH",
    "DEFAULT_PROGRAM",
    "build_arrangement",
]
