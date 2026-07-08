from creative_audio_lab.music_theory import (
    chord_notes,
    diatonic_chord,
    midi_note,
    nearest_scale_tone,
    note_name_to_pitch_class,
    normalize_mode,
    scale_pitch_classes,
)


def test_note_name_to_pitch_class():
    assert note_name_to_pitch_class("C") == 0
    assert note_name_to_pitch_class("c#") == 1
    assert note_name_to_pitch_class("Db") == 1
    assert note_name_to_pitch_class("B") == 11


def test_midi_note():
    assert midi_note(0, 4) == 60  # C4
    assert midi_note(9, 4) == 69  # A4


def test_normalize_mode_aliases():
    assert normalize_mode("Minor") == "natural_minor"
    assert normalize_mode("harmonic minor") == "harmonic_minor"
    assert normalize_mode("Ionian") == "major"


def test_scale_pitch_classes_lengths_and_content():
    for mode in ("major", "natural_minor", "harmonic_minor", "dorian", "phrygian"):
        pcs = scale_pitch_classes("C", mode)
        assert len(pcs) == 7
        assert pcs[0] == 0
        assert len(set(pcs)) == 7


def test_diatonic_chord_qualities_major():
    assert diatonic_chord("C", "major", 1).quality == "maj"
    assert diatonic_chord("C", "major", 2).quality == "min"
    assert diatonic_chord("C", "major", 5).quality == "maj"
    assert diatonic_chord("C", "major", 7).quality == "dim"


def test_diatonic_chord_qualities_natural_minor():
    assert diatonic_chord("A", "natural_minor", 1).quality == "min"
    assert diatonic_chord("A", "natural_minor", 3).quality == "maj"  # relative major
    assert diatonic_chord("A", "natural_minor", 2).quality == "dim"


def test_diatonic_chord_qualities_harmonic_minor_has_raised_seventh():
    # Harmonic minor's V chord is major (raised 7th degree), unlike natural minor's v (minor).
    assert diatonic_chord("A", "harmonic_minor", 5).quality == "maj"
    assert diatonic_chord("A", "natural_minor", 5).quality == "min"


def test_chord_notes_are_a_triad():
    chord = diatonic_chord("C", "major", 1)
    notes = chord_notes(chord, octave=3)
    assert len(notes) == 3
    assert notes[0] % 12 == 0  # C


def test_nearest_scale_tone():
    scale = scale_pitch_classes("C", "major")
    assert nearest_scale_tone(60, scale) == 60  # C already in scale
    assert nearest_scale_tone(61, scale) in (60, 62)  # C# snaps to a neighbour
