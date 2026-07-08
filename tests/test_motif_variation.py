import io

import mido

from creative_audio_lab.motif_variation import (
    add_passing_tones,
    augment_rhythm,
    call_and_response,
    compress_rhythm,
    export_variation,
    invert_contour,
    transpose,
)
from creative_audio_lab.music_theory import Note, scale_pitch_classes


def _motif():
    return [
        Note(start=0.0, pitch=60, duration=1.0, velocity=100),
        Note(start=1.0, pitch=64, duration=1.0, velocity=100),
        Note(start=2.0, pitch=67, duration=1.0, velocity=100),
    ]


def test_transpose_shifts_every_pitch():
    result = transpose(_motif(), 5)
    assert [n.pitch for n in result] == [65, 69, 72]


def test_invert_contour_mirrors_around_first_note():
    result = invert_contour(_motif())
    assert result[0].pitch == 60  # axis note is unchanged
    assert result[1].pitch == 56  # 2*60 - 64
    assert result[2].pitch == 53  # 2*60 - 67


def test_augment_rhythm_stretches_time():
    result = augment_rhythm(_motif(), factor=2.0)
    assert [n.start for n in result] == [0.0, 2.0, 4.0]
    assert [n.duration for n in result] == [2.0, 2.0, 2.0]


def test_compress_rhythm_shrinks_time():
    result = compress_rhythm(_motif(), factor=0.5)
    assert [n.start for n in result] == [0.0, 0.5, 1.0]
    assert [n.duration for n in result] == [0.5, 0.5, 0.5]


def test_add_passing_tones_inserts_extra_note_on_large_leap():
    scale = scale_pitch_classes("C", "major")
    motif = [
        Note(start=0.0, pitch=60, duration=1.0, velocity=100),
        Note(start=1.0, pitch=72, duration=1.0, velocity=100),  # octave leap
    ]
    result = add_passing_tones(motif, scale)
    assert len(result) == 3
    assert result[1].pitch != motif[0].pitch and result[1].pitch != motif[1].pitch


def test_call_and_response_appends_transposed_continuation():
    motif = _motif()
    result = call_and_response(motif, transpose_semitones=-5)
    assert len(result) == len(motif) * 2
    response = result[len(motif):]
    assert response[0].start == motif[-1].end()
    assert response[0].pitch == motif[0].pitch - 5


def test_export_variation_produces_valid_midi_bytes():
    data = export_variation(_motif(), name="variation", program=0, bpm=100.0)
    mid = mido.MidiFile(file=io.BytesIO(data))
    assert len(mid.tracks) == 2  # tempo track + motif track
