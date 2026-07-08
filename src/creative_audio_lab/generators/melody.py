"""Deterministic, motif-based melody generation.

The generator builds one short scale-degree contour ("motif") per prompt,
then re-states it across the chord progression with light transposition,
occasional passing tones, and a resolved phrase ending every four bars —
instead of sampling pitches independently, which produces unmusical "note
soup".
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from ..music_theory import Note, midi_note, nearest_scale_tone, scale_pitch_classes
from ..prompt_parser import GenerationControls
from .chords import ChordEvent

# Sensible playable ranges per lead instrument (MIDI note numbers).
INSTRUMENT_RANGES: Dict[str, Tuple[int, int]] = {
    "flute": (67, 96),
    "piano": (57, 84),
    "strings": (55, 79),
    "brass": (50, 72),
    "bells": (72, 96),
    "default": (60, 81),
}

# One bar (4 beats) of rhythmic subdivisions, keyed by rhythmic density.
RHYTHM_TEMPLATES: Dict[str, Tuple[float, ...]] = {
    "low": (2.0, 1.0, 1.0),
    "medium": (1.0, 1.0, 0.5, 0.5, 1.0),
    "high": (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0),
}

_STEP_CHOICES = (-2, -1, -1, 0, 1, 1, 2)


def _seed_for(controls: GenerationControls) -> int:
    return abs(hash((controls.prompt, controls.key, controls.mode, controls.bpm, controls.style, "melody"))) % (2**32)


def lead_instrument(controls: GenerationControls) -> str:
    """Pick which instrument tag should carry the melody."""
    for tag in ("flute", "piano", "strings", "brass", "bells"):
        if tag in controls.instruments:
            return tag
    return "default"


def _build_motif(rhythm: Tuple[float, ...], rng: random.Random) -> List[int]:
    """Return a scale-degree-step contour (relative to a chord root), one entry per rhythm slot."""
    steps = [0]
    for _ in range(len(rhythm) - 1):
        steps.append(steps[-1] + rng.choice(_STEP_CHOICES))
    return steps


def _insert_passing_tones(notes: List[Note], scale_pcs: List[int], density: str, rng: random.Random) -> List[Note]:
    """Bridge large melodic leaps with a short passing tone, more often at higher density."""
    if density == "low" or len(notes) < 2:
        return notes

    chance = 0.55 if density == "high" else 0.35
    result: List[Note] = [notes[0]]
    for note in notes[1:]:
        prev = result[-1]
        interval = note.pitch - prev.pitch
        if abs(interval) >= 3 and note.duration >= 0.5 and rng.random() < chance:
            direction = 1 if interval > 0 else -1
            passing_pitch = nearest_scale_tone(prev.pitch + direction * 2, scale_pcs)
            half = note.duration / 2
            result.append(Note(start=note.start, pitch=passing_pitch, duration=half, velocity=max(note.velocity - 18, 40)))
            result.append(Note(start=note.start + half, pitch=note.pitch, duration=note.duration - half, velocity=note.velocity))
        else:
            result.append(note)
    return result


def generate_melody(controls: GenerationControls, chord_events: List[ChordEvent]) -> List[Note]:
    """Generate a scale-constrained, motif-driven melody following ``chord_events``."""
    if not chord_events:
        return []

    rng = random.Random(_seed_for(controls))
    scale = scale_pitch_classes(controls.key, controls.mode)
    scale_len = len(scale)

    instrument = lead_instrument(controls)
    low_pitch, high_pitch = INSTRUMENT_RANGES.get(instrument, INSTRUMENT_RANGES["default"])
    base_octave = 5 if instrument in ("flute", "bells") else 4

    rhythm = RHYTHM_TEMPLATES.get(controls.density, RHYTHM_TEMPLATES["medium"])
    motif = _build_motif(rhythm, rng)
    motif_length = len(motif)

    notes: List[Note] = []
    total_bars = len(chord_events)

    for bar_index, chord_event in enumerate(chord_events):
        is_phrase_end = (bar_index + 1) % 4 == 0 or bar_index == total_bars - 1
        transpose = 0 if bar_index % 2 == 0 else rng.choice((0, 1, -1, 2))
        root_degree_index = chord_event.chord.degree - 1

        cursor = chord_event.start_beat
        for slot, (step, duration) in enumerate(zip(motif, rhythm)):
            last_in_bar = slot == motif_length - 1
            if is_phrase_end and last_in_bar:
                pitch_class = chord_event.chord.pitch_classes()[0]
            else:
                degree_index = (root_degree_index + step + transpose) % scale_len
                pitch_class = scale[degree_index]

            pitch = midi_note(pitch_class, base_octave)
            while pitch < low_pitch:
                pitch += 12
            while pitch > high_pitch:
                pitch -= 12

            velocity = 106 if (is_phrase_end and last_in_bar) else rng.randint(76, 102)
            notes.append(Note(start=cursor, pitch=pitch, duration=duration, velocity=velocity))
            cursor += duration

    return _insert_passing_tones(notes, scale, controls.density, rng)


__all__ = ["INSTRUMENT_RANGES", "RHYTHM_TEMPLATES", "lead_instrument", "generate_melody"]
