import pytest

from midi import (
    note_to_number,
    number_to_note,
    create_orchestral_midi,
)


def test_note_to_number():
    assert note_to_number('C4') == 60
    assert note_to_number('A4') == 69


def test_number_to_note():
    assert number_to_note(60) == 'C4'
    assert number_to_note(69) == 'A4'


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


if __name__ == '__main__':
    pytest.main([__file__])
