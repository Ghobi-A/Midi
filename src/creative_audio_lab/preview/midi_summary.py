"""Per-track and whole-piece summaries of symbolic note content.

Pure functions over :class:`Note` lists — usable on generated arrangements
and on parsed MIDI files alike (pair with
:func:`creative_audio_lab.midi_parser.parse_midi_file` for the latter).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Tuple

from ..music_theory import Note

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type hints only
    from ..generators.arrangement import Arrangement


@dataclass(frozen=True)
class TrackSummary:
    """Summary statistics for one track/part."""

    name: str
    program: Optional[int]
    is_drum: bool
    note_count: int
    pitch_min: Optional[int]
    pitch_max: Optional[int]
    duration_beats: float
    duration_bars: float
    notes_per_bar: float


@dataclass(frozen=True)
class MidiSummary:
    """Summary of a whole piece: per-track stats plus piece-level totals."""

    tracks: Tuple[TrackSummary, ...]
    total_notes: int
    duration_beats: float
    duration_bars: float
    has_drums: bool
    bpm: Optional[float] = None

    @property
    def pitch_range(self) -> Optional[Tuple[int, int]]:
        """Lowest/highest pitch across all non-drum tracks, or ``None`` if there are no pitched notes."""
        lows = [t.pitch_min for t in self.tracks if not t.is_drum and t.pitch_min is not None]
        highs = [t.pitch_max for t in self.tracks if not t.is_drum and t.pitch_max is not None]
        if not lows or not highs:
            return None
        return (min(lows), max(highs))


def summarize_notes(
    name: str,
    notes: Sequence[Note],
    *,
    program: Optional[int] = None,
    is_drum: bool = False,
    beats_per_bar: float = 4.0,
) -> TrackSummary:
    """Summarize one note list as a :class:`TrackSummary`.

    ``duration_bars`` is rounded up to whole bars; ``notes_per_bar`` is the
    density over that span (0.0 for an empty track).
    """
    if not notes:
        return TrackSummary(
            name=name,
            program=program,
            is_drum=is_drum,
            note_count=0,
            pitch_min=None,
            pitch_max=None,
            duration_beats=0.0,
            duration_bars=0.0,
            notes_per_bar=0.0,
        )
    duration_beats = max(note.end() for note in notes)
    duration_bars = math.ceil(duration_beats / beats_per_bar) if duration_beats > 0 else 0
    pitches = [note.pitch for note in notes]
    return TrackSummary(
        name=name,
        program=program,
        is_drum=is_drum,
        note_count=len(notes),
        pitch_min=min(pitches),
        pitch_max=max(pitches),
        duration_beats=duration_beats,
        duration_bars=float(duration_bars),
        notes_per_bar=len(notes) / duration_bars if duration_bars else 0.0,
    )


def summarize_tracks(
    tracks: Mapping[str, Sequence[Note]],
    *,
    programs: Optional[Mapping[str, int]] = None,
    drum_tracks: Sequence[str] = ("drums",),
    bpm: Optional[float] = None,
    beats_per_bar: float = 4.0,
) -> MidiSummary:
    """Summarize a ``{track_name: notes}`` mapping into a :class:`MidiSummary`.

    ``drum_tracks`` names which tracks are percussion (no meaningful pitch
    range); ``programs`` optionally maps track names to General MIDI program
    numbers.
    """
    programs = programs or {}
    drum_names = set(drum_tracks)
    summaries: List[TrackSummary] = [
        summarize_notes(
            name,
            notes,
            program=programs.get(name),
            is_drum=name in drum_names,
            beats_per_bar=beats_per_bar,
        )
        for name, notes in tracks.items()
    ]
    duration_beats = max((t.duration_beats for t in summaries), default=0.0)
    duration_bars = max((t.duration_bars for t in summaries), default=0.0)
    return MidiSummary(
        tracks=tuple(summaries),
        total_notes=sum(t.note_count for t in summaries),
        duration_beats=duration_beats,
        duration_bars=duration_bars,
        has_drums=any(t.is_drum and t.note_count > 0 for t in summaries),
        bpm=bpm,
    )


def summarize_arrangement(arrangement: "Arrangement", *, beats_per_bar: float = 4.0) -> MidiSummary:
    """Summarize a generated :class:`Arrangement` (parts, programs, BPM included)."""
    parts: Dict[str, Sequence[Note]] = dict(arrangement.parts)
    return summarize_tracks(
        parts,
        programs=arrangement.programs,
        drum_tracks=("drums",),
        bpm=arrangement.bpm,
        beats_per_bar=beats_per_bar,
    )


__all__ = ["TrackSummary", "MidiSummary", "summarize_notes", "summarize_tracks", "summarize_arrangement"]
