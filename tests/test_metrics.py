from creative_audio_lab.evaluation.metrics import (
    evaluate_arrangement,
    harmonic_fit_score,
    motif_retention_score,
    note_density,
    novelty_score,
    pitch_range,
    repetition_score,
    rhythmic_complexity_score,
    scale_adherence_score,
)
from creative_audio_lab.generators.arrangement import build_arrangement
from creative_audio_lab.generators.chords import ChordEvent
from creative_audio_lab.music_theory import Note, diatonic_chord
from creative_audio_lab.prompt_parser import parse_prompt


def _notes(pitches, duration=1.0):
    notes = []
    cursor = 0.0
    for pitch in pitches:
        notes.append(Note(start=cursor, pitch=pitch, duration=duration, velocity=90))
        cursor += duration
    return notes


def test_note_density():
    notes = _notes([60, 62, 64, 65, 67, 69, 71, 72])
    assert note_density(notes, bars=4) == 2.0
    assert note_density([], bars=0) == 0.0


def test_pitch_range():
    notes = _notes([60, 72, 65])
    assert pitch_range(notes) == (60, 72)
    assert pitch_range([]) == (0, 0)


def test_repetition_score_detects_repeating_pattern():
    repeating = _notes([60, 62, 64, 60, 62, 64, 60, 62, 64])
    non_repeating = _notes([60, 61, 63, 66, 70, 59, 61, 68, 72])
    assert repetition_score(repeating, n=3) > repetition_score(non_repeating, n=3)


def test_scale_adherence_score():
    in_scale = _notes([60, 62, 64, 65, 67])  # C major scale tones
    out_of_scale = _notes([61, 63, 66, 70])  # chromatic, mostly outside C major
    major_scale = {0, 2, 4, 5, 7, 9, 11}
    assert scale_adherence_score(in_scale, major_scale) == 1.0
    assert scale_adherence_score(out_of_scale, major_scale) < 1.0


def test_harmonic_fit_score_against_chord_tones():
    chord = diatonic_chord("C", "major", 1)  # C-E-G
    chord_events = [ChordEvent(start_beat=0.0, duration_beats=4.0, chord=chord)]
    on_chord = [Note(start=0.0, pitch=60, duration=1.0), Note(start=1.0, pitch=64, duration=1.0)]
    off_chord = [Note(start=0.0, pitch=61, duration=1.0), Note(start=1.0, pitch=66, duration=1.0)]
    assert harmonic_fit_score(on_chord, chord_events) == 1.0
    assert harmonic_fit_score(off_chord, chord_events) == 0.0


def test_rhythmic_complexity_score_zero_for_uniform_durations():
    notes = _notes([60, 62, 64, 65], duration=0.5)
    assert rhythmic_complexity_score(notes) == 0.0


def test_novelty_score_higher_for_varied_intervals():
    varied = _notes([60, 63, 58, 67, 65, 72])  # every interval distinct -> max entropy
    repetitive = _notes([60, 60, 60, 60, 60, 60])  # zero interval every step -> min entropy
    assert novelty_score(varied) > novelty_score(repetitive)


def test_motif_retention_score_identical_sequences_is_one():
    notes = _notes([60, 62, 64, 65, 67])
    assert motif_retention_score(notes, notes) == 1.0


def test_motif_retention_score_drops_for_unrelated_sequence():
    original = _notes([60, 62, 64, 65, 67])
    unrelated = _notes([60, 40, 90, 30, 100])
    assert motif_retention_score(original, unrelated) < 1.0


def test_evaluate_arrangement_returns_expected_keys():
    controls = parse_prompt("hopeful piano and strings theme for a fantasy village")
    arrangement = build_arrangement(controls)
    metrics = evaluate_arrangement(arrangement)
    expected_keys = {
        "note_density", "pitch_min", "pitch_max", "pitch_range",
        "repetition_score", "harmonic_fit_score", "rhythmic_complexity_score",
        "novelty_score", "scale_adherence_score",
    }
    assert expected_keys <= metrics.keys()
    assert metrics["scale_adherence_score"] == 1.0
