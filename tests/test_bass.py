from creative_audio_lab.generators.bass import generate_bassline
from creative_audio_lab.generators.chords import generate_progression
from creative_audio_lab.prompt_parser import GenerationControls


def _controls(**overrides):
    defaults = dict(prompt="test", key="C", mode="major", bars=4, style="default", energy="medium", texture=[])
    defaults.update(overrides)
    return GenerationControls(**defaults)


def test_default_bass_follows_root_and_fifth():
    controls = _controls(style="default", energy="medium")
    events = generate_progression(controls)
    bass = generate_bassline(controls, events)
    # Two notes per bar (root, fifth) for the plain non-ostinato/non-sliding path.
    assert len(bass) == controls.bars * 2
    first_bar_notes = [n for n in bass if n.start < 4.0]
    root_pc = events[0].chord.root_pc
    fifth_pc = events[0].chord.pitch_classes()[2]
    assert first_bar_notes[0].pitch % 12 == root_pc
    assert first_bar_notes[1].pitch % 12 == fifth_pc


def test_cinematic_style_produces_ostinato_pattern():
    controls = _controls(style="cinematic")
    events = generate_progression(controls)
    bass = generate_bassline(controls, events)
    assert len(bass) == controls.bars * 8


def test_boss_battle_section_forces_ostinato_even_for_other_styles():
    controls = _controls(style="default", section="boss_battle")
    events = generate_progression(controls)
    bass = generate_bassline(controls, events)
    assert len(bass) == controls.bars * 8


def test_trap_style_produces_sliding_bass():
    controls = _controls(style="trap")
    events = generate_progression(controls)
    bass = generate_bassline(controls, events)
    assert len(bass) == controls.bars * 3  # root, short glide note, target


def test_high_energy_adds_a_walking_passing_note():
    controls = _controls(style="default", energy="high")
    events = generate_progression(controls)
    bass = generate_bassline(controls, events)
    assert len(bass) == controls.bars * 4
