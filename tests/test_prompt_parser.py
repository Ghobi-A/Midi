from creative_audio_lab.prompt_parser import parse_prompt


def test_infers_bpm_and_key_from_text():
    controls = parse_prompt("romantic piano loop in C minor, 90 BPM")
    assert controls.bpm == 90
    assert controls.key == "C"
    assert controls.mode == "natural_minor"
    assert controls.mood == "romantic"
    assert "piano" in controls.instruments


def test_infers_style_and_section_from_boss_battle():
    controls = parse_prompt("dark orchestral boss battle theme, 140 BPM, brass ostinato, strings, piano arps")
    assert controls.bpm == 140
    assert controls.mood == "dark"
    assert controls.style == "cinematic"
    assert controls.section == "battle"
    assert "ostinato" in controls.texture
    assert "arpeggio" in controls.texture
    assert set(controls.instruments) >= {"brass", "strings", "piano"}


def test_infers_trap_style_and_sliding_bass_texture():
    controls = parse_prompt("trap melody with dark bells and sliding bass")
    assert controls.style == "trap"
    assert "sliding_bass" in controls.texture
    assert "bells" in controls.instruments


def test_jrpg_town_theme_section():
    controls = parse_prompt("nostalgic JRPG town theme with piano and flute")
    assert controls.style == "jrpg"
    assert controls.section == "town"
    assert controls.mood == "nostalgic"


def test_explicit_overrides_win_over_inferred_text():
    controls = parse_prompt("dark trap beat, 140 BPM", key="G", bpm=100, style="ambient")
    assert controls.key == "G"
    assert controls.bpm == 100
    assert controls.style == "ambient"


def test_defaults_when_nothing_recognized():
    controls = parse_prompt("a piece of music")
    assert controls.style == "default"
    assert controls.mood == "neutral"
    assert controls.bars > 0
    assert controls.bpm > 0
