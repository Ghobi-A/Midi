import pytest

from midi import note_to_number, number_to_note


def test_note_to_number():
    assert note_to_number('C4') == 60
    assert note_to_number('A4') == 69


def test_number_to_note():
    assert number_to_note(60) == 'C4'
    assert number_to_note(69) == 'A4'


if __name__ == '__main__':
    pytest.main([__file__])
