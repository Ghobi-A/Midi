from creative_audio_lab.generators.arrangement import build_arrangement
from creative_audio_lab.music_theory import Note
from creative_audio_lab.prompt_parser import parse_prompt
from creative_audio_lab.tokenization import (
    BAR_ADVANCE_VALUE,
    SymbolicTokenizer,
    Token,
    TokenType,
    TokenizerConfig,
    measure_round_trip,
    strings_to_tokens,
    token_from_string,
    tokens_to_strings,
)

import pytest

SIMPLE_PHRASE = [
    Note(start=0.0, pitch=60, duration=1.0, velocity=96),
    Note(start=1.0, pitch=62, duration=0.5, velocity=92),
    Note(start=1.5, pitch=64, duration=0.5, velocity=92),
    Note(start=2.0, pitch=67, duration=2.0, velocity=100),
    Note(start=4.0, pitch=65, duration=1.0, velocity=88),  # second bar
]


def test_encode_single_note_produces_expected_tokens():
    tokens = SymbolicTokenizer().encode([Note(start=0.0, pitch=60, duration=1.0, velocity=96)])
    assert [t.type for t in tokens] == [
        TokenType.BAR,
        TokenType.POSITION,
        TokenType.NOTE_ON,
        TokenType.VELOCITY,
        TokenType.DURATION,
    ]
    assert tokens[0].value == 0  # bar index
    assert tokens[1].value == 0  # position step within bar
    assert tokens[2].value == 60  # pitch
    assert tokens[3].value == 24  # velocity 96 // bin width 4
    assert tokens[4].value == 4  # 1 beat at 4 steps/beat


def test_encode_emits_metadata_header_tokens():
    tokenizer = SymbolicTokenizer()
    tokens = tokenizer.encode(SIMPLE_PHRASE, tempo=120, program=48, time_signature="4/4")
    assert tokens[0] == Token(TokenType.TEMPO, 120)
    assert tokens[1] == Token(TokenType.TIME_SIGNATURE, "4/4")
    assert tokens[2] == Token(TokenType.PROGRAM, 48)


def test_encode_emits_bar_token_on_bar_change():
    tokens = SymbolicTokenizer().encode(SIMPLE_PHRASE)
    bar_tokens = [t for t in tokens if t.type is TokenType.BAR]
    assert [t.value for t in bar_tokens] == [0, 1]


def test_decode_tokens_to_note():
    tokens = [
        Token(TokenType.BAR, 1),
        Token(TokenType.POSITION, 8),
        Token(TokenType.NOTE_ON, 72),
        Token(TokenType.VELOCITY, 25),
        Token(TokenType.DURATION, 2),
    ]
    decoded = SymbolicTokenizer().decode(tokens)
    assert decoded.notes == (Note(start=6.0, pitch=72, duration=0.5, velocity=100),)


def test_decode_uses_default_velocity_when_velocity_token_missing():
    tokens = [
        Token(TokenType.BAR, 0),
        Token(TokenType.POSITION, 0),
        Token(TokenType.NOTE_ON, 60),
        Token(TokenType.DURATION, 4),
    ]
    decoded = SymbolicTokenizer().decode(tokens)
    assert decoded.notes[0].velocity == 96


def test_decode_drops_note_on_without_duration():
    tokens = [
        Token(TokenType.BAR, 0),
        Token(TokenType.POSITION, 0),
        Token(TokenType.NOTE_ON, 60),
        Token(TokenType.NOTE_ON, 64),
        Token(TokenType.VELOCITY, 24),
        Token(TokenType.DURATION, 4),
    ]
    decoded = SymbolicTokenizer().decode(tokens)
    assert [note.pitch for note in decoded.notes] == [64]


def test_roundtrip_simple_phrase_is_exact():
    tokenizer = SymbolicTokenizer()
    tokens = tokenizer.encode(SIMPLE_PHRASE, tempo=100, program=0, time_signature="4/4")
    decoded = tokenizer.decode(tokens)
    assert list(decoded.notes) == SIMPLE_PHRASE
    assert decoded.tempo == 100
    assert decoded.program == 0
    assert decoded.time_signature == "4/4"


def test_roundtrip_generated_chords_part():
    controls = parse_prompt("romantic piano loop in C minor, 90 BPM")
    arrangement = build_arrangement(controls)
    chords = arrangement.parts["chords"]
    tokenizer = SymbolicTokenizer()
    decoded = tokenizer.decode(tokenizer.encode(chords, tempo=arrangement.bpm))
    assert sorted(decoded.notes, key=lambda n: (n.start, n.pitch)) == sorted(
        chords, key=lambda n: (n.start, n.pitch)
    )


def test_roundtrip_via_token_strings():
    tokenizer = SymbolicTokenizer()
    strings = tokenizer.encode_to_strings(SIMPLE_PHRASE, time_signature="4/4")
    assert all(isinstance(s, str) for s in strings)
    decoded = tokenizer.decode_from_strings(strings)
    assert list(decoded.notes) == SIMPLE_PHRASE


def test_token_string_forms_parse_back():
    tokens = [
        Token(TokenType.NOTE_ON, 60),
        Token(TokenType.TIME_SIGNATURE, "4/4"),
        Token(TokenType.BAR, 3),
    ]
    strings = tokens_to_strings(tokens)
    assert strings == ["NOTE_ON_60", "TIME_SIGNATURE_4/4", "BAR_3"]
    assert strings_to_tokens(strings) == tokens


def test_token_from_string_rejects_garbage():
    with pytest.raises(ValueError):
        token_from_string("NOT_A_TOKEN_TYPE_5")
    with pytest.raises(ValueError):
        token_from_string("nounderscore")


def test_duration_is_capped_at_max_duration():
    config = TokenizerConfig(max_duration_beats=2.0)
    tokens = SymbolicTokenizer(config).encode([Note(start=0.0, pitch=60, duration=16.0, velocity=96)])
    duration_token = next(t for t in tokens if t.type is TokenType.DURATION)
    assert duration_token.value == 8  # 2 beats * 4 steps/beat


# ---------------------------------------------------------------------------
# Relative (bar-advance) mode — the closed bar vocabulary for training
# ---------------------------------------------------------------------------


def test_relative_mode_emits_only_bar_advance_tokens():
    tokenizer = SymbolicTokenizer(TokenizerConfig(relative_bars=True))
    tokens = tokenizer.encode(SIMPLE_PHRASE)
    bar_tokens = [t for t in tokens if t.type is TokenType.BAR]
    assert [t.value for t in bar_tokens] == [BAR_ADVANCE_VALUE, BAR_ADVANCE_VALUE]


def test_relative_mode_keeps_bar_vocabulary_closed_on_long_pieces():
    # 300 bars of notes: absolute mode grows the vocab per bar, relative
    # mode never emits anything but BAR_NONE.
    long_piece = [Note(start=4.0 * bar, pitch=60, duration=1.0, velocity=96) for bar in range(300)]
    strings = SymbolicTokenizer(TokenizerConfig(relative_bars=True)).encode_to_strings(long_piece)
    bar_strings = {s for s in strings if s.startswith("BAR_")}
    assert bar_strings == {f"BAR_{BAR_ADVANCE_VALUE}"}


def test_relative_mode_preserves_empty_bars():
    # A note in bar 0 and the next in bar 3: two empty bars in between must
    # survive the round trip as repeated advance tokens.
    notes = [
        Note(start=0.0, pitch=60, duration=1.0, velocity=96),
        Note(start=12.0, pitch=64, duration=1.0, velocity=96),
    ]
    tokenizer = SymbolicTokenizer(TokenizerConfig(relative_bars=True))
    tokens = tokenizer.encode(notes)
    assert sum(1 for t in tokens if t.type is TokenType.BAR) == 4  # bars 0..3
    assert list(tokenizer.decode(tokens).notes) == notes


def test_relative_mode_roundtrip_is_exact_on_grid_aligned_notes():
    tokenizer = SymbolicTokenizer(TokenizerConfig(relative_bars=True))
    decoded = tokenizer.decode(tokenizer.encode(SIMPLE_PHRASE))
    assert list(decoded.notes) == SIMPLE_PHRASE


def test_bar_advance_token_string_form_parses_back():
    token = Token(TokenType.BAR, BAR_ADVANCE_VALUE)
    assert token.to_string() == "BAR_NONE"
    assert token_from_string("BAR_NONE") == token


def test_decode_handles_mixed_absolute_and_relative_bar_tokens():
    tokens = [
        Token(TokenType.BAR, 5),
        Token(TokenType.POSITION, 0),
        Token(TokenType.NOTE_ON, 60),
        Token(TokenType.VELOCITY, 24),
        Token(TokenType.DURATION, 4),
        Token(TokenType.BAR, BAR_ADVANCE_VALUE),  # advances to bar 6
        Token(TokenType.POSITION, 0),
        Token(TokenType.NOTE_ON, 62),
        Token(TokenType.VELOCITY, 24),
        Token(TokenType.DURATION, 4),
    ]
    decoded = SymbolicTokenizer().decode(tokens)
    assert [note.start for note in decoded.notes] == [20.0, 24.0]


# ---------------------------------------------------------------------------
# Round-trip measurement — the gate before tokenizing any real corpus
# ---------------------------------------------------------------------------


def test_measure_round_trip_timing_is_lossless_on_generated_output():
    # Generated melodies sit exactly on the 16th grid, so timing round-trips
    # losslessly — but their velocities come from rng.randint and get rebinned
    # by the default 32-bin config. measure_round_trip surfaced exactly that.
    controls = parse_prompt("hopeful jrpg town theme, flute, 120 BPM")
    arrangement = build_arrangement(controls)
    melody = arrangement.parts["melody"]

    report = measure_round_trip(melody)
    assert report.onset_moved == 0
    assert report.duration_changed == 0

    lossless_velocity = SymbolicTokenizer(TokenizerConfig(velocity_bins=128))
    exact_report = measure_round_trip(melody, lossless_velocity)
    assert exact_report.exact_fraction == 1.0


def test_measure_round_trip_counts_quantization_loss_on_off_grid_notes():
    off_grid = [
        Note(start=0.13, pitch=60, duration=0.9, velocity=97),  # off-grid onset/duration, odd velocity
        Note(start=1.0, pitch=62, duration=1.0, velocity=96),   # exactly on grid
    ]
    report = measure_round_trip(off_grid)
    assert report.total_notes == 2
    assert report.exact_notes == 1
    assert report.onset_moved == 1
    assert report.duration_changed == 1
    assert report.velocity_changed == 1
    assert report.exact_fraction == 0.5


def test_measure_round_trip_empty_input():
    report = measure_round_trip([])
    assert report.total_notes == 0
    assert report.exact_fraction == 1.0
