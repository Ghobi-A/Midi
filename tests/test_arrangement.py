from creative_audio_lab.generators.arrangement import build_arrangement
from creative_audio_lab.prompt_parser import parse_prompt


def test_build_arrangement_produces_all_parts():
    controls = parse_prompt("dark orchestral boss battle theme, 140 BPM, brass ostinato, strings, piano arps")
    arrangement = build_arrangement(controls)
    assert set(arrangement.parts) == {"chords", "melody", "bass", "drums"}
    assert arrangement.parts["chords"]
    assert arrangement.parts["melody"]
    assert arrangement.parts["bass"]
    assert arrangement.parts["drums"]  # boss battle section implies drums
    assert arrangement.bpm == 140
    assert arrangement.bars == controls.bars


def test_build_arrangement_omits_drums_for_ballad():
    controls = parse_prompt("romantic piano loop in C minor, 90 BPM")
    arrangement = build_arrangement(controls)
    assert arrangement.parts["drums"] == []


def test_program_selection_prefers_requested_instruments():
    controls = parse_prompt("nostalgic JRPG town theme with piano and flute")
    arrangement = build_arrangement(controls)
    from creative_audio_lab.generators.arrangement import INSTRUMENT_PROGRAMS

    assert arrangement.programs["melody"] == INSTRUMENT_PROGRAMS["flute"]


def test_bass_program_uses_synth_bass_for_trap():
    from creative_audio_lab.generators.arrangement import BASS_PROGRAM_SYNTH

    controls = parse_prompt("trap melody with dark bells and sliding bass")
    arrangement = build_arrangement(controls)
    assert arrangement.programs["bass"] == BASS_PROGRAM_SYNTH
