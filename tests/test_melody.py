from creative_audio_lab.generators.chords import generate_progression
from creative_audio_lab.generators.melody import INSTRUMENT_RANGES, generate_melody, lead_instrument
from creative_audio_lab.music_theory import scale_pitch_classes
from creative_audio_lab.prompt_parser import GenerationControls


def _controls(**overrides):
    defaults = dict(prompt="test", key="C", mode="major", bars=8, style="default", density="medium", instruments=["piano"])
    defaults.update(overrides)
    return GenerationControls(**defaults)


def test_melody_notes_are_scale_constrained():
    controls = _controls()
    chord_events = generate_progression(controls)
    melody = generate_melody(controls, chord_events)
    scale = set(scale_pitch_classes(controls.key, controls.mode))
    assert melody
    assert all(note.pitch % 12 in scale for note in melody)


def test_melody_respects_instrument_range():
    controls = _controls(instruments=["flute"])
    chord_events = generate_progression(controls)
    melody = generate_melody(controls, chord_events)
    low, high = INSTRUMENT_RANGES["flute"]
    assert all(low <= note.pitch <= high for note in melody)


def test_lead_instrument_prefers_flute_over_piano():
    controls = _controls(instruments=["piano", "flute"])
    assert lead_instrument(controls) == "flute"


def test_melody_is_deterministic_for_same_controls():
    controls = _controls()
    chord_events = generate_progression(controls)
    first = generate_melody(controls, chord_events)
    second = generate_melody(controls, chord_events)
    assert [(n.start, n.pitch, n.duration) for n in first] == [(n.start, n.pitch, n.duration) for n in second]


def test_higher_density_produces_more_notes():
    low_controls = _controls(density="low", bars=4)
    high_controls = _controls(density="high", bars=4)
    low_events = generate_progression(low_controls)
    high_events = generate_progression(high_controls)
    low_melody = generate_melody(low_controls, low_events)
    high_melody = generate_melody(high_controls, high_events)
    assert len(high_melody) > len(low_melody)


def test_empty_chord_events_produce_no_melody():
    controls = _controls()
    assert generate_melody(controls, []) == []
