from creative_audio_lab.generators.chords import generate_chord_progression, generate_progression
from creative_audio_lab.music_theory import scale_pitch_classes
from creative_audio_lab.prompt_parser import GenerationControls


def test_progression_has_one_chord_per_bar():
    controls = GenerationControls(prompt="test", key="C", mode="major", bars=8, style="default")
    events = generate_progression(controls)
    assert len(events) == 8
    assert events[0].start_beat == 0.0
    assert events[1].start_beat == 4.0


def test_chords_stay_within_the_key():
    controls = GenerationControls(prompt="test", key="C", mode="major", bars=8, style="cinematic")
    events = generate_progression(controls)
    scale = set(scale_pitch_classes("C", "major"))
    for event in events:
        assert event.chord.root_pc in scale
        for pc in event.chord.pitch_classes():
            # Triads built from stacked scale thirds always land on scale tones.
            assert pc in scale


def test_render_chords_produces_three_notes_per_bar():
    controls = GenerationControls(prompt="test", key="C", mode="major", bars=4, style="default")
    _, notes = generate_chord_progression(controls)
    assert len(notes) == 4 * 3


def test_style_presets_use_different_progressions():
    controls_a = GenerationControls(prompt="test", key="C", mode="major", bars=4, style="cinematic")
    controls_b = GenerationControls(prompt="test", key="C", mode="major", bars=4, style="trap")
    events_a = generate_progression(controls_a)
    events_b = generate_progression(controls_b)
    degrees_a = [e.chord.degree for e in events_a]
    degrees_b = [e.chord.degree for e in events_b]
    assert degrees_a != degrees_b
