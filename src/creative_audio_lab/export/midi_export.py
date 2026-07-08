"""Render :class:`~creative_audio_lab.music_theory.Note` sequences to MIDI bytes/files.

Everything here is a pure function that returns a :class:`mido.MidiFile` or raw
``bytes`` — nothing is written to disk automatically, so callers (the CLI, the
Streamlit app, tests) decide where output goes.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, TYPE_CHECKING

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack

from ..music_theory import Note

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type hints only
    from ..generators.arrangement import Arrangement

# General MIDI percussion is always channel 10 (index 9).
DRUM_CHANNEL = 9

# Fixed channel assignment for the four arrangement parts.
PART_CHANNELS: Dict[str, int] = {
    "chords": 0,
    "melody": 1,
    "bass": 2,
    "drums": DRUM_CHANNEL,
}


def bpm_to_tempo(bpm: float) -> int:
    """Convert beats-per-minute to MIDI microseconds-per-beat."""
    return mido.bpm2tempo(bpm)


def notes_to_track(
    notes: List[Note],
    name: str,
    program: Optional[int] = None,
    channel: int = 0,
    ticks_per_beat: int = 480,
) -> MidiTrack:
    """Convert a list of :class:`Note` into a single named :class:`mido.MidiTrack`."""
    track = MidiTrack()
    track.name = name
    if program is not None and channel != DRUM_CHANNEL:
        track.append(Message("program_change", program=program, channel=channel, time=0))

    events = []
    for note in notes:
        start_tick = round(note.start * ticks_per_beat)
        end_tick = max(round(note.end() * ticks_per_beat), start_tick + 1)
        events.append((start_tick, 1, Message("note_on", note=note.pitch, velocity=note.velocity, channel=channel, time=0)))
        events.append((end_tick, 0, Message("note_off", note=note.pitch, velocity=0, channel=channel, time=0)))

    # note_off (priority 0) must sort before note_on (priority 1) at equal ticks
    # so back-to-back notes of the same pitch don't get stuck "on".
    events.sort(key=lambda event: (event[0], event[1]))

    current_tick = 0
    for tick, _, message in events:
        message.time = tick - current_tick
        track.append(message)
        current_tick = tick

    return track


def export_stem(
    part_name: str,
    notes: List[Note],
    program: Optional[int],
    bpm: float,
    ticks_per_beat: int = 480,
) -> MidiFile:
    """Build a single-track MIDI file for one arrangement part (e.g. ``"bass"``)."""
    channel = PART_CHANNELS.get(part_name, 0)
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    tempo_track = MidiTrack()
    tempo_track.append(MetaMessage("set_tempo", tempo=bpm_to_tempo(bpm), time=0))
    mid.tracks.append(tempo_track)
    mid.tracks.append(notes_to_track(notes, part_name, program=program, channel=channel, ticks_per_beat=ticks_per_beat))
    return mid


def export_full_arrangement(arrangement: "Arrangement") -> MidiFile:
    """Build a single multi-track MIDI file containing every non-empty arrangement part."""
    mid = MidiFile(ticks_per_beat=arrangement.ticks_per_beat)
    tempo_track = MidiTrack()
    tempo_track.append(MetaMessage("set_tempo", tempo=bpm_to_tempo(arrangement.bpm), time=0))
    mid.tracks.append(tempo_track)

    for part_name in ("chords", "melody", "bass", "drums"):
        notes = arrangement.parts.get(part_name, [])
        if not notes:
            continue
        program = arrangement.programs.get(part_name)
        channel = PART_CHANNELS.get(part_name, 0)
        mid.tracks.append(
            notes_to_track(notes, part_name, program=program, channel=channel, ticks_per_beat=arrangement.ticks_per_beat)
        )
    return mid


def midi_to_bytes(mid: MidiFile) -> bytes:
    """Serialize a :class:`mido.MidiFile` to raw bytes (no temp files needed)."""
    buffer = io.BytesIO()
    mid.save(file=buffer)
    return buffer.getvalue()


def export_arrangement_files(arrangement: "Arrangement") -> Dict[str, bytes]:
    """Return ``{filename: midi_bytes}`` for the full arrangement plus every stem."""
    files: Dict[str, bytes] = {
        "full_arrangement.mid": midi_to_bytes(export_full_arrangement(arrangement))
    }
    for part_name in ("chords", "melody", "bass", "drums"):
        notes = arrangement.parts.get(part_name, [])
        program = arrangement.programs.get(part_name)
        stem = export_stem(part_name, notes, program, arrangement.bpm, arrangement.ticks_per_beat)
        files[f"{part_name}.mid"] = midi_to_bytes(stem)
    return files


def export_notes_to_bytes(
    notes: List[Note],
    name: str = "track",
    program: int = 0,
    channel: int = 0,
    bpm: float = 100.0,
    ticks_per_beat: int = 480,
) -> bytes:
    """Export an arbitrary note list (e.g. a motif variation) as a standalone MIDI file."""
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    tempo_track = MidiTrack()
    tempo_track.append(MetaMessage("set_tempo", tempo=bpm_to_tempo(bpm), time=0))
    mid.tracks.append(tempo_track)
    mid.tracks.append(notes_to_track(notes, name, program=program, channel=channel, ticks_per_beat=ticks_per_beat))
    return midi_to_bytes(mid)


__all__ = [
    "DRUM_CHANNEL",
    "PART_CHANNELS",
    "bpm_to_tempo",
    "notes_to_track",
    "export_stem",
    "export_full_arrangement",
    "midi_to_bytes",
    "export_arrangement_files",
    "export_notes_to_bytes",
]
