from creative_audio_lab.generators.chords import generate_progression
from creative_audio_lab.generators.drums import (
    CLOSED_HAT,
    KICK,
    LOW_TOM,
    SNARE,
    generate_drum_pattern,
    select_template,
    wants_drums,
)
from creative_audio_lab.prompt_parser import GenerationControls


def _controls(**overrides):
    defaults = dict(prompt="test", key="C", mode="major", bars=4, style="default", energy="medium", instruments=[], section="theme")
    defaults.update(overrides)
    return GenerationControls(**defaults)


def test_wants_drums_true_for_trap_style():
    assert wants_drums(_controls(style="trap")) is True


def test_wants_drums_true_when_drums_in_instruments():
    assert wants_drums(_controls(instruments=["drums"])) is True


def test_wants_drums_true_for_boss_battle_section():
    assert wants_drums(_controls(section="boss_battle")) is True


def test_wants_drums_false_for_plain_ballad():
    assert wants_drums(_controls(style="ballad")) is False


def test_select_template_mapping():
    assert select_template(_controls(style="trap")) == "trap"
    assert select_template(_controls(style="cinematic")) == "cinematic"
    assert select_template(_controls(style="default", energy="high")) == "dance"
    assert select_template(_controls(style="default", energy="medium")) == "basic"


def test_generate_drum_pattern_uses_gm_note_numbers():
    controls = _controls(style="default", instruments=["drums"])
    events = generate_progression(controls)
    drums = generate_drum_pattern(controls, events)
    assert drums
    pitches = {note.pitch for note in drums}
    assert pitches <= {KICK, SNARE, CLOSED_HAT}


def test_cinematic_template_uses_low_tom_and_is_sparse():
    controls = _controls(style="cinematic", instruments=["drums"])
    events = generate_progression(controls)
    drums = generate_drum_pattern(controls, events)
    assert all(note.pitch == LOW_TOM for note in drums)
    assert len(drums) == controls.bars * 3


def test_no_drums_when_not_wanted():
    controls = _controls(style="ballad")
    events = generate_progression(controls)
    assert generate_drum_pattern(controls, events) == []
