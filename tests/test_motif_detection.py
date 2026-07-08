from creative_audio_lab.motif_detection import (
    detect_melodic_motifs,
    detect_motifs,
    detect_rhythmic_motifs,
    extract_contour,
    extract_intervals,
    extract_rhythm,
    find_repeated_ngrams,
)
from creative_audio_lab.music_theory import Note


def _make_repeating_notes():
    pitches = [60, 62, 64, 63, 65, 67, 66]
    durations = [0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5]
    notes = []
    cursor = 0.0
    for pitch, duration in zip(pitches, durations):
        notes.append(Note(start=cursor, pitch=pitch, duration=duration, velocity=90))
        cursor += duration
    return notes


def test_extract_intervals():
    notes = _make_repeating_notes()
    assert extract_intervals(notes) == [2, 2, -1, 2, 2, -1]


def test_extract_rhythm():
    notes = _make_repeating_notes()
    assert extract_rhythm(notes) == [0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5]


def test_extract_contour():
    notes = _make_repeating_notes()
    assert extract_contour(notes) == ["U", "U", "D", "U", "U", "D"]


def test_find_repeated_ngrams():
    sequence = [1, 2, 3, 1, 2, 3, 4]
    repeats = find_repeated_ngrams(sequence, 3)
    assert repeats == {(1, 2, 3): [0, 3]}


def test_detect_melodic_motifs_finds_known_repeat():
    notes = _make_repeating_notes()
    motifs = detect_melodic_motifs(notes, lengths=[3])
    matches = [m for m in motifs if m.intervals == (2, 2, -1)]
    assert matches
    assert matches[0].repetition_count == 2
    assert matches[0].length == 4


def test_detect_rhythmic_motifs_finds_known_repeat():
    notes = _make_repeating_notes()
    motifs = detect_rhythmic_motifs(notes, lengths=[3])
    matches = [m for m in motifs if m.durations == (0.5, 0.5, 1.0)]
    assert matches
    assert matches[0].repetition_count == 2


def test_detect_motifs_ranks_best_first():
    notes = _make_repeating_notes()
    motifs = detect_motifs(notes, lengths=[3])
    assert motifs
    scores = [m.score for m in motifs]
    assert scores == sorted(scores, reverse=True)


def test_no_motifs_in_non_repeating_sequence():
    pitches = [60, 67, 61, 74]
    durations = [0.5, 0.5, 1.0, 0.5]
    notes = []
    cursor = 0.0
    for pitch, duration in zip(pitches, durations):
        notes.append(Note(start=cursor, pitch=pitch, duration=duration, velocity=90))
        cursor += duration
    motifs = detect_motifs(notes, lengths=[3])
    assert motifs == []
