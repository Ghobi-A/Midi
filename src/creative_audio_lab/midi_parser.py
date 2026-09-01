"""Parse MIDI files/tracks into the package's canonical :class:`Note` events.

Notes are matched by ``(channel, pitch)`` rather than pitch alone, so two
channels playing the same pitch in one track do not steal each other's
note-off events. Duplicate track names are disambiguated instead of
overwriting each other.

:func:`parse_midi_info` extracts the file-level facts a training pipeline
needs to filter and describe a corpus — time signatures, tempos, per-track
programs and channels, drum tracks, and sustain-pedal use — none of which
survive into the :class:`Note` representation itself. Those limitations are
deliberate and documented in ``docs/DATASETS.md``: the canonical ``Note`` is
a pitch/time/velocity event, so pedal, program changes mid-track, and
channel identity are corpus *metadata*, not note content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import mido

from .music_theory import Note

#: MIDI channel reserved for percussion in General MIDI (0-based).
DRUM_CHANNEL = 9
#: Controller number for the sustain (damper) pedal.
SUSTAIN_CONTROLLER = 64


@dataclass(frozen=True)
class TrackInfo:
    """What a training pipeline needs to know about one track beyond its notes."""

    index: int
    name: str
    channels: Tuple[int, ...] = ()
    programs: Tuple[int, ...] = ()
    note_count: int = 0
    is_drum: bool = False
    uses_sustain_pedal: bool = False


@dataclass(frozen=True)
class MidiFileInfo:
    """File-level metadata that the flat ``{track: [Note]}`` view discards."""

    ticks_per_beat: int
    time_signatures: Tuple[str, ...] = ()
    tempos_bpm: Tuple[float, ...] = ()
    tracks: Tuple[TrackInfo, ...] = field(default=())

    @property
    def uses_sustain_pedal(self) -> bool:
        return any(track.uses_sustain_pedal for track in self.tracks)

    @property
    def is_single_time_signature(self) -> bool:
        return len(set(self.time_signatures)) <= 1

    def primary_time_signature(self, default: str = "4/4") -> str:
        """The file's first time signature, or ``default`` when none is declared."""
        return self.time_signatures[0] if self.time_signatures else default


def parse_midi_file(path: Union[str, Path]) -> Dict[str, List[Note]]:
    """Load a ``.mid`` file from disk and return its tracks as :class:`Note` lists."""
    return parse_midi(mido.MidiFile(str(path)))


def parse_midi_file_with_info(
    path: Union[str, Path],
) -> Tuple[Dict[str, List[Note]], MidiFileInfo]:
    """Load a ``.mid`` file and return both its notes and its file-level metadata."""
    mid = mido.MidiFile(str(path))
    return parse_midi(mid), parse_midi_info(mid)


def _unique_track_name(name: str, taken: Dict[str, int]) -> str:
    """Disambiguate a repeated track name as ``name#2``, ``name#3``, ... ."""
    seen = taken.get(name, 0) + 1
    taken[name] = seen
    return name if seen == 1 else f"{name}#{seen}"


def parse_midi(mid: mido.MidiFile) -> Dict[str, List[Note]]:
    """Convert a loaded :class:`mido.MidiFile` into ``{track_name: [Note, ...]}``.

    Tracks that share a name are kept apart by suffixing the second and
    later occurrences (``piano``, ``piano#2``), so no track is silently
    overwritten.
    """
    ticks_per_beat = mid.ticks_per_beat
    tracks: Dict[str, List[Note]] = {}
    taken: Dict[str, int] = {}
    for index, track in enumerate(mid.tracks):
        notes = track_to_notes(track, ticks_per_beat)
        if notes:
            tracks[_unique_track_name(track.name or f"track_{index}", taken)] = notes
    return tracks


def parse_midi_info(mid: mido.MidiFile) -> MidiFileInfo:
    """Extract time signatures, tempos, and per-track channel/program/pedal facts."""
    time_signatures: List[str] = []
    tempos: List[float] = []
    track_infos: List[TrackInfo] = []
    taken: Dict[str, int] = {}

    for index, track in enumerate(mid.tracks):
        channels: List[int] = []
        programs: List[int] = []
        note_count = 0
        uses_sustain = False
        for msg in track:
            if msg.type == "time_signature":
                signature = f"{msg.numerator}/{msg.denominator}"
                if signature not in time_signatures:
                    time_signatures.append(signature)
            elif msg.type == "set_tempo":
                bpm = round(mido.tempo2bpm(msg.tempo), 3)
                if bpm not in tempos:
                    tempos.append(bpm)
            elif msg.type == "program_change":
                if msg.program not in programs:
                    programs.append(msg.program)
            elif msg.type == "control_change" and msg.control == SUSTAIN_CONTROLLER:
                uses_sustain = True
            if getattr(msg, "channel", None) is not None and msg.channel not in channels:
                channels.append(msg.channel)
            if msg.type == "note_on" and msg.velocity > 0:
                note_count += 1
        track_infos.append(
            TrackInfo(
                index=index,
                name=_unique_track_name(track.name or f"track_{index}", taken),
                channels=tuple(sorted(channels)),
                programs=tuple(programs),
                note_count=note_count,
                is_drum=DRUM_CHANNEL in channels,
                uses_sustain_pedal=uses_sustain,
            )
        )

    return MidiFileInfo(
        ticks_per_beat=mid.ticks_per_beat,
        time_signatures=tuple(time_signatures),
        tempos_bpm=tuple(tempos),
        tracks=tuple(track_infos),
    )


def track_to_notes(track, ticks_per_beat: int) -> List[Note]:
    """Convert a single :class:`mido.MidiTrack` into a chronological list of :class:`Note`.

    Sounding notes are keyed by ``(channel, pitch)``, so the same pitch on
    two channels is tracked independently.
    """
    notes: List[Note] = []
    pending: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}
    current_tick = 0

    for msg in track:
        current_tick += msg.time
        if msg.type not in ("note_on", "note_off"):
            continue
        key = (getattr(msg, "channel", 0), msg.note)
        if msg.type == "note_on" and msg.velocity > 0:
            pending.setdefault(key, []).append((current_tick / ticks_per_beat, msg.velocity))
        else:
            queue = pending.get(key)
            if queue:
                start_beat, velocity = queue.pop(0)
                duration = max(current_tick / ticks_per_beat - start_beat, 0.0)
                notes.append(Note(start=start_beat, pitch=msg.note, duration=duration, velocity=velocity))

    notes.sort(key=lambda note: (note.start, note.pitch))
    return notes


def flatten_notes(tracks: Dict[str, List[Note]]) -> List[Note]:
    """Merge every track's notes into a single chronological list."""
    merged = [note for notes in tracks.values() for note in notes]
    merged.sort(key=lambda note: note.start)
    return merged


def polyphony_ratio(notes: List[Note]) -> float:
    """Fraction of notes that share an onset with another note.

    0.0 for a strictly monophonic line; higher for chordal or multi-voice
    content. Used by corpus intake to pick a melody track.
    """
    if not notes:
        return 0.0
    counts: Dict[float, int] = {}
    for note in notes:
        counts[note.start] = counts.get(note.start, 0) + 1
    stacked = sum(count for count in counts.values() if count > 1)
    return stacked / len(notes)


__all__ = [
    "DRUM_CHANNEL",
    "SUSTAIN_CONTROLLER",
    "MidiFileInfo",
    "TrackInfo",
    "parse_midi_file",
    "parse_midi_file_with_info",
    "parse_midi",
    "parse_midi_info",
    "track_to_notes",
    "flatten_notes",
    "polyphony_ratio",
]
