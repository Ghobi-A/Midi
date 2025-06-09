"""Utilities for creating layered MIDI sequences."""

from typing import Dict, Iterable, Tuple

try:  # pragma: no cover - optional dependency
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except Exception:  # pragma: no cover - mido missing
    MidiFile = MidiTrack = Message = MetaMessage = None

NoteEvent = Tuple[float, int, float, int]
"""time (beats), note number, duration (beats), velocity"""


def create_orchestral_midi(
    layers: Dict[str, Iterable[NoteEvent]],
    tempo: int = 500000,
    ticks_per_beat: int = 480,
) -> MidiFile:
    """Return a ``MidiFile`` representing multiple instrument layers.

    Parameters
    ----------
    layers:
        Mapping of track names to iterables of ``NoteEvent`` tuples.
    tempo:
        Microseconds per beat (default 500000, i.e. 120 BPM).
    ticks_per_beat:
        Ticks per beat in the created ``MidiFile``.
    """

    if MidiFile is None:
        raise ImportError("mido is required for create_orchestral_midi")

    mid = MidiFile(ticks_per_beat=ticks_per_beat)

    tempo_track = MidiTrack()
    tempo_track.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    mid.tracks.append(tempo_track)

    for name, events in layers.items():
        track = MidiTrack()
        track.name = name
        mid.tracks.append(track)

        current_tick = 0
        for start, note, duration, velocity in sorted(events, key=lambda e: e[0]):
            start_tick = int(start * ticks_per_beat)
            delta = start_tick - current_tick
            track.append(Message("note_on", note=note, velocity=velocity, time=delta))
            track.append(
                Message(
                    "note_off",
                    note=note,
                    velocity=0,
                    time=int(duration * ticks_per_beat),
                )
            )
            current_tick = start_tick + int(duration * ticks_per_beat)

    return mid


__all__ = ["create_orchestral_midi", "NoteEvent"]
