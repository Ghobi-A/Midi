"""Deterministic drum-pattern generation using the General MIDI percussion map."""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..music_theory import Note
from ..prompt_parser import GenerationControls
from .chords import ChordEvent

# General MIDI percussion key numbers (channel 10).
KICK = 36
SNARE = 38
CLAP = 39
LOW_TOM = 41
CLOSED_HAT = 42
OPEN_HAT = 46
CRASH = 49
RIDE = 51

# Each template is a list of (start_beat, drum_note, velocity, duration_beats)
# events for a single 4/4 bar, tiled across every bar of the arrangement.
DrumHit = Tuple[float, int, int, float]

_TEMPLATES: Dict[str, List[DrumHit]] = {
    "basic": [
        (0.0, KICK, 108, 0.4),
        (1.0, SNARE, 100, 0.4),
        (2.0, KICK, 108, 0.4),
        (3.0, SNARE, 100, 0.4),
        (0.0, CLOSED_HAT, 76, 0.4), (0.5, CLOSED_HAT, 64, 0.4),
        (1.0, CLOSED_HAT, 76, 0.4), (1.5, CLOSED_HAT, 64, 0.4),
        (2.0, CLOSED_HAT, 76, 0.4), (2.5, CLOSED_HAT, 64, 0.4),
        (3.0, CLOSED_HAT, 76, 0.4), (3.5, CLOSED_HAT, 64, 0.4),
    ],
    "trap": [
        (0.0, KICK, 112, 0.3),
        (1.75, KICK, 100, 0.3),
        (2.5, KICK, 104, 0.3),
        (2.0, CLAP, 104, 0.4),
        (0.0, CLOSED_HAT, 70, 0.2), (0.25, CLOSED_HAT, 60, 0.2), (0.5, CLOSED_HAT, 70, 0.2),
        (0.75, CLOSED_HAT, 60, 0.2), (1.0, CLOSED_HAT, 70, 0.2), (1.25, CLOSED_HAT, 60, 0.2),
        (1.5, CLOSED_HAT, 70, 0.2), (1.75, CLOSED_HAT, 60, 0.2), (2.0, CLOSED_HAT, 70, 0.2),
        (2.25, CLOSED_HAT, 60, 0.2), (2.5, CLOSED_HAT, 70, 0.2), (2.75, CLOSED_HAT, 60, 0.2),
        (3.0, CLOSED_HAT, 70, 0.2), (3.25, CLOSED_HAT, 60, 0.2), (3.5, CLOSED_HAT, 70, 0.2),
        (3.75, OPEN_HAT, 80, 0.2),
    ],
    "cinematic": [
        (0.0, LOW_TOM, 108, 0.6),
        (2.0, LOW_TOM, 96, 0.6),
        (2.75, LOW_TOM, 90, 0.3),
    ],
    "dance": [
        (0.0, KICK, 110, 0.4),
        (1.0, KICK, 110, 0.4),
        (2.0, KICK, 110, 0.4),
        (3.0, KICK, 110, 0.4),
        (1.0, CLAP, 100, 0.4), (3.0, CLAP, 100, 0.4),
        (0.5, OPEN_HAT, 78, 0.4), (1.5, OPEN_HAT, 78, 0.4),
        (2.5, OPEN_HAT, 78, 0.4), (3.5, OPEN_HAT, 78, 0.4),
    ],
}


def select_template(controls: GenerationControls) -> str:
    """Choose a rhythmic template name based on style and energy."""
    if controls.style in ("trap", "drill"):
        return "trap"
    if controls.style in ("cinematic", "orchestral"):
        return "cinematic"
    if controls.energy == "high":
        return "dance"
    return "basic"


def wants_drums(controls: GenerationControls) -> bool:
    """Decide whether this prompt/style combination should include a drum part at all."""
    if controls.style in ("trap", "drill"):
        return True
    if "drums" in controls.instruments or "taiko_drums" in controls.instruments:
        return True
    if controls.section in ("boss_battle", "battle"):
        return True
    return False


def generate_drum_pattern(controls: GenerationControls, chord_events: List[ChordEvent]) -> List[Note]:
    """Tile the selected drum template across every bar in ``chord_events``."""
    if not chord_events or not wants_drums(controls):
        return []

    template = _TEMPLATES[select_template(controls)]
    notes: List[Note] = []
    for chord_event in chord_events:
        bar_start = chord_event.start_beat
        for offset, drum_note, velocity, duration in template:
            notes.append(Note(start=bar_start + offset, pitch=drum_note, duration=duration, velocity=velocity))
    return notes


__all__ = [
    "KICK", "SNARE", "CLAP", "LOW_TOM", "CLOSED_HAT", "OPEN_HAT", "CRASH", "RIDE",
    "select_template", "wants_drums", "generate_drum_pattern",
]
