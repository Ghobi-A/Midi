import io

import mido
import pytest

from creative_audio_lab.export.midi_export import export_notes_to_bytes
from creative_audio_lab.midi_parser import (
    flatten_notes,
    parse_midi,
    parse_midi_info,
    polyphony_ratio,
    track_to_notes,
)
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


# ---------------------------------------------------------------------------
# Identity and metadata: duplicate names, per-channel notes, file info
# ---------------------------------------------------------------------------


def _file_with_tracks(*tracks):
    mid = mido.MidiFile(ticks_per_beat=480)
    for messages in tracks:
        track = mido.MidiTrack()
        for msg in messages:
            track.append(msg)
        mid.tracks.append(track)
    return mid


def test_duplicate_track_names_do_not_overwrite_each_other():
    def piano(pitch):
        return [
            mido.MetaMessage("track_name", name="piano", time=0),
            mido.Message("note_on", note=pitch, velocity=90, time=0),
            mido.Message("note_off", note=pitch, velocity=0, time=480),
        ]

    tracks = parse_midi(_file_with_tracks(piano(60), piano(72)))
    assert set(tracks) == {"piano", "piano#2"}
    assert tracks["piano"][0].pitch == 60
    assert tracks["piano#2"][0].pitch == 72


def test_same_pitch_on_two_channels_is_tracked_separately():
    # Channel 0 holds the note for 4 beats; channel 1 plays the same pitch
    # for 1 beat inside it. Keyed by pitch alone, the short note-off would
    # end the long note.
    messages = [
        mido.Message("note_on", note=60, velocity=90, channel=0, time=0),
        mido.Message("note_on", note=60, velocity=70, channel=1, time=0),
        mido.Message("note_off", note=60, velocity=0, channel=1, time=480),
        mido.Message("note_off", note=60, velocity=0, channel=0, time=1440),
    ]
    notes = track_to_notes(_file_with_tracks(messages).tracks[0], 480)
    assert sorted(note.duration for note in notes) == [1.0, 4.0]


def test_parse_midi_info_reports_signature_tempo_program_and_pedal():
    messages = [
        mido.MetaMessage("track_name", name="lead", time=0),
        mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0),
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(90), time=0),
        mido.Message("program_change", program=48, channel=0, time=0),
        mido.Message("control_change", control=64, value=127, channel=0, time=0),
        mido.Message("note_on", note=60, velocity=90, channel=0, time=0),
        mido.Message("note_off", note=60, velocity=0, channel=0, time=480),
    ]
    drums = [
        mido.MetaMessage("track_name", name="drums", time=0),
        mido.Message("note_on", note=36, velocity=100, channel=9, time=0),
        mido.Message("note_off", note=36, velocity=0, channel=9, time=120),
    ]
    info = parse_midi_info(_file_with_tracks(messages, drums))
    assert info.time_signatures == ("3/4",)
    assert info.primary_time_signature() == "3/4"
    assert info.tempos_bpm == (90.0,)
    assert info.uses_sustain_pedal is True
    lead, drum_track = info.tracks
    assert lead.programs == (48,) and lead.is_drum is False and lead.note_count == 1
    assert drum_track.is_drum is True


def test_primary_time_signature_defaults_to_four_four():
    info = parse_midi_info(_file_with_tracks([mido.Message("note_on", note=60, velocity=1, time=0)]))
    assert info.primary_time_signature() == "4/4"
    assert info.is_single_time_signature is True


def test_polyphony_ratio_separates_melody_from_chords():
    melody = [Note(start=float(i), pitch=60 + i, duration=1.0) for i in range(4)]
    chords = [Note(start=0.0, pitch=p, duration=1.0) for p in (60, 64, 67)]
    assert polyphony_ratio(melody) == 0.0
    assert polyphony_ratio(chords) == 1.0
    assert polyphony_ratio([]) == 0.0
