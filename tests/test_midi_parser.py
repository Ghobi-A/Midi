import io

import mido
import pytest

from creative_audio_lab.export.midi_export import export_notes_to_bytes
from creative_audio_lab.midi_parser import flatten_notes, parse_midi
from creative_audio_lab.music_theory import Note


def test_round_trip_export_and_parse():
    original = [
        Note(start=0.0, pitch=60, duration=1.0, velocity=100),
        Note(start=1.0, pitch=64, duration=0.5, velocity=90),
        Note(start=1.5, pitch=67, duration=1.5, velocity=80),
    ]
    data = export_notes_to_bytes(original, name="melody", program=0, bpm=120.0)
    mid = mido.MidiFile(file=io.BytesIO(data))

    tracks = parse_midi(mid)
    assert "melody" in tracks
    recovered = tracks["melody"]
    assert len(recovered) == len(original)

    for expected, actual in zip(original, recovered):
        assert actual.pitch == expected.pitch
        assert actual.start == pytest.approx(expected.start, abs=1e-3)
        assert actual.duration == pytest.approx(expected.duration, abs=1e-3)


def test_flatten_notes_merges_and_sorts_tracks():
    tracks = {
        "a": [Note(start=1.0, pitch=60, duration=1.0)],
        "b": [Note(start=0.0, pitch=62, duration=1.0)],
    }
    merged = flatten_notes(tracks)
    assert [note.start for note in merged] == [0.0, 1.0]
    assert len(merged) == 2
