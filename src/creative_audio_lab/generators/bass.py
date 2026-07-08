"""Deterministic bassline generation.

Follows chord roots and fifths by default, with style-specific behaviour:
a repeating root/fifth ostinato for cinematic and boss-battle style cues, a
"sliding" grace-note glide into the second half of the bar for trap/drill
presets, and a walking passing tone for high-energy styles.
"""

from __future__ import annotations

from typing import List

from ..music_theory import Note, midi_note
from ..prompt_parser import GenerationControls
from .chords import ChordEvent

BASS_OCTAVE = 2


def _wants_ostinato(controls: GenerationControls) -> bool:
    return (
        "ostinato" in controls.texture
        or controls.section in ("boss_battle", "battle")
        or controls.style in ("cinematic", "orchestral")
    )


def _wants_sliding(controls: GenerationControls) -> bool:
    return controls.style in ("trap", "drill") or "sliding_bass" in controls.texture


def generate_bassline(controls: GenerationControls, chord_events: List[ChordEvent]) -> List[Note]:
    """Generate a bassline that tracks ``chord_events`` using style-appropriate rhythm."""
    notes: List[Note] = []
    ostinato = _wants_ostinato(controls)
    sliding = _wants_sliding(controls)

    for chord_event in chord_events:
        root_pc, _, fifth_pc = chord_event.chord.pitch_classes()
        root = midi_note(root_pc, BASS_OCTAVE)
        fifth = midi_note(fifth_pc, BASS_OCTAVE)
        bar_start = chord_event.start_beat
        bar_len = chord_event.duration_beats

        if ostinato:
            subdivisions = 8
            step = bar_len / subdivisions
            pattern = [root, root, fifth, root, root, root, fifth, root]
            for i, pitch in enumerate(pattern):
                notes.append(Note(start=bar_start + i * step, pitch=pitch, duration=step * 0.9, velocity=94))

        elif sliding:
            half = bar_len / 2
            slide_target = fifth if fifth != root else root + 12
            glide_len = half * 0.2
            notes.append(Note(start=bar_start, pitch=root, duration=half - glide_len, velocity=102))
            direction = 1 if slide_target > root else -1
            notes.append(Note(start=bar_start + half - glide_len, pitch=root + direction, duration=glide_len, velocity=70))
            notes.append(Note(start=bar_start + half, pitch=slide_target, duration=half, velocity=96))

        elif controls.energy == "high":
            quarter = bar_len / 4
            passing = root + 2
            pattern = [root, passing, fifth, root]
            for i, pitch in enumerate(pattern):
                notes.append(Note(start=bar_start + i * quarter, pitch=pitch, duration=quarter * 0.9, velocity=88))

        else:
            half = bar_len / 2
            notes.append(Note(start=bar_start, pitch=root, duration=half, velocity=90))
            notes.append(Note(start=bar_start + half, pitch=fifth, duration=half, velocity=84))

    return notes


__all__ = ["BASS_OCTAVE", "generate_bassline"]
