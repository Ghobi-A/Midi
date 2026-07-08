"""Deterministic chord-progression generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..music_theory import Chord, Note, chord_notes, diatonic_chord
from ..prompt_parser import DEFAULT_STYLE, STYLE_PRESETS, GenerationControls

BEATS_PER_BAR = 4.0


@dataclass(frozen=True)
class ChordEvent:
    """One bar's worth of harmony: a diatonic :class:`Chord` and its timing."""

    start_beat: float
    duration_beats: float
    chord: Chord


def generate_progression(controls: GenerationControls) -> List[ChordEvent]:
    """Build a bar-by-bar chord progression by cycling the style's degree pattern."""
    preset = STYLE_PRESETS.get(controls.style, STYLE_PRESETS[DEFAULT_STYLE])
    degrees: Tuple[int, ...] = preset.progression
    events: List[ChordEvent] = []
    for bar in range(controls.bars):
        degree = degrees[bar % len(degrees)]
        chord = diatonic_chord(controls.key, controls.mode, degree)
        events.append(ChordEvent(start_beat=bar * BEATS_PER_BAR, duration_beats=BEATS_PER_BAR, chord=chord))
    return events


def render_chords(events: List[ChordEvent], octave: int = 3, velocity: int = 72) -> List[Note]:
    """Voice each :class:`ChordEvent` as a sustained block chord."""
    notes: List[Note] = []
    for event in events:
        for pitch in chord_notes(event.chord, octave=octave):
            notes.append(Note(start=event.start_beat, pitch=pitch, duration=event.duration_beats, velocity=velocity))
    return notes


def generate_chord_progression(controls: GenerationControls) -> Tuple[List[ChordEvent], List[Note]]:
    """Return both the chord-event timeline and its rendered block-chord notes."""
    events = generate_progression(controls)
    notes = render_chords(events)
    return events, notes


__all__ = ["ChordEvent", "BEATS_PER_BAR", "generate_progression", "render_chords", "generate_chord_progression"]
