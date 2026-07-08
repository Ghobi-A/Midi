"""Parse MIDI files/tracks into the package's canonical :class:`Note` events."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import mido

from .music_theory import Note


def parse_midi_file(path: Union[str, Path]) -> Dict[str, List[Note]]:
    """Load a ``.mid`` file from disk and return its tracks as :class:`Note` lists."""
    mid = mido.MidiFile(str(path))
    return parse_midi(mid)


def parse_midi(mid: mido.MidiFile) -> Dict[str, List[Note]]:
    """Convert a loaded :class:`mido.MidiFile` into ``{track_name: [Note, ...]}``."""
    ticks_per_beat = mid.ticks_per_beat
    tracks: Dict[str, List[Note]] = {}
    for index, track in enumerate(mid.tracks):
        notes = track_to_notes(track, ticks_per_beat)
        if notes:
            name = track.name or f"track_{index}"
            tracks[name] = notes
    return tracks


def track_to_notes(track, ticks_per_beat: int) -> List[Note]:
    """Convert a single :class:`mido.MidiTrack` into a chronological list of :class:`Note`."""
    notes: List[Note] = []
    pending: Dict[int, List[Tuple[float, int]]] = {}
    current_tick = 0

    for msg in track:
        current_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            pending.setdefault(msg.note, []).append((current_tick / ticks_per_beat, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            queue = pending.get(msg.note)
            if queue:
                start_beat, velocity = queue.pop(0)
                duration = max(current_tick / ticks_per_beat - start_beat, 0.0)
                notes.append(Note(start=start_beat, pitch=msg.note, duration=duration, velocity=velocity))

    notes.sort(key=lambda note: note.start)
    return notes


def flatten_notes(tracks: Dict[str, List[Note]]) -> List[Note]:
    """Merge every track's notes into a single chronological list."""
    merged = [note for notes in tracks.values() for note in notes]
    merged.sort(key=lambda note: note.start)
    return merged


__all__ = ["parse_midi_file", "parse_midi", "track_to_notes", "flatten_notes"]
