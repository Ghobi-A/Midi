"""Lightweight MIDI preview/analysis layer (no audio rendering).

Summaries (:mod:`.midi_summary`) and piano-roll grids (:mod:`.piano_roll`)
are pure data structures computed from :class:`Note` events — no DAW,
plugin host, or audio dependency involved. They give the app (and future
dataset tooling) something inspectable to show before a user drags the
MIDI into a DAW.
"""

from .midi_summary import MidiSummary, TrackSummary, summarize_arrangement, summarize_tracks
from .piano_roll import PianoRoll, build_piano_roll

__all__ = [
    "MidiSummary",
    "TrackSummary",
    "summarize_arrangement",
    "summarize_tracks",
    "PianoRoll",
    "build_piano_roll",
]
