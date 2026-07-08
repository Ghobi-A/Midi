import pytest

from midi import (
    note_to_number,
    number_to_note,
    create_orchestral_midi,
)


def test_note_to_number():
    assert note_to_number('C4') == 60
    assert note_to_number('A4') == 69
    # Flats map to their sharp equivalents and input is case-insensitive
    assert note_to_number('Db4') == note_to_number('C#4')
    assert note_to_number('c4') == 60


def test_number_to_note():
    assert number_to_note(60) == 'C4'
    assert number_to_note(69) == 'A4'
    # Conversions handle sharps and flats
    assert number_to_note(note_to_number('Db4')) == 'C#4'
    assert number_to_note(note_to_number('C#4'), prefer_sharps=False) == 'Db4'
    assert number_to_note(note_to_number('db4')) == 'C#4'


def test_create_orchestral_midi():
    mido = pytest.importorskip("mido")
    layers = {
        "piano": [(0.0, 60, 1.0, 64)],
        "strings": [(0.5, 67, 1.5, 64)],
    }
    mid = create_orchestral_midi(layers)
    assert isinstance(mid, mido.MidiFile)
    # 1 tempo track + number of layers
    assert len(mid.tracks) == 1 + len(layers)


def test_program_change():
    pytest.importorskip("mido")
    layers = {
        "piano": ([(0.0, 60, 1.0, 64)], 1),
        "strings": [(0.5, 67, 1.5, 64)],
    }
    mid = create_orchestral_midi(layers)
    piano_track = next(t for t in mid.tracks if getattr(t, "name", "") == "piano")
    piano_program = next(msg for msg in piano_track if msg.type == "program_change")
    assert piano_program.program == 1
    assert piano_program.time == 0
    strings_track = next(
        t for t in mid.tracks if getattr(t, "name", "") == "strings"
    )
    assert all(msg.type != "program_change" for msg in strings_track)


if __name__ == '__main__':
    pytest.main([__file__])
