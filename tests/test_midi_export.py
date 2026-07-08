import io

import mido

from creative_audio_lab.export.midi_export import DRUM_CHANNEL, export_arrangement_files
from creative_audio_lab.generators.arrangement import build_arrangement
from creative_audio_lab.prompt_parser import parse_prompt


def _arrangement():
    controls = parse_prompt("dark orchestral boss battle theme, 140 BPM, brass ostinato, strings, piano arps")
    return build_arrangement(controls)


def test_export_arrangement_files_returns_all_expected_filenames():
    files = export_arrangement_files(_arrangement())
    assert set(files) == {"full_arrangement.mid", "chords.mid", "melody.mid", "bass.mid", "drums.mid"}
    for data in files.values():
        assert isinstance(data, bytes)
        assert len(data) > 0


def test_exported_files_are_valid_midi():
    files = export_arrangement_files(_arrangement())
    for name, data in files.items():
        mid = mido.MidiFile(file=io.BytesIO(data))
        assert len(mid.tracks) >= 1, f"{name} should contain at least one track"


def test_full_arrangement_has_a_track_per_nonempty_part():
    arrangement = _arrangement()
    files = export_arrangement_files(arrangement)
    mid = mido.MidiFile(file=io.BytesIO(files["full_arrangement.mid"]))
    track_names = {track.name for track in mid.tracks if getattr(track, "name", "")}
    expected_parts = {name for name, notes in arrangement.parts.items() if notes}
    assert expected_parts <= track_names


def test_drums_are_on_the_gm_percussion_channel():
    arrangement = _arrangement()
    files = export_arrangement_files(arrangement)
    mid = mido.MidiFile(file=io.BytesIO(files["drums.mid"]))
    channels = {msg.channel for track in mid.tracks for msg in track if not msg.is_meta}
    assert channels == {DRUM_CHANNEL}
