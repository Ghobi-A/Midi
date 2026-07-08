import pytest

from creative_audio_lab.models import DeterministicBackend
from creative_audio_lab.music_theory import Note
from creative_audio_lab.preview import build_piano_roll, summarize_arrangement, summarize_tracks
from creative_audio_lab.preview.midi_summary import summarize_notes

MELODY = [
    Note(start=0.0, pitch=60, duration=1.0, velocity=96),
    Note(start=1.0, pitch=64, duration=1.0, velocity=96),
    Note(start=2.0, pitch=67, duration=2.0, velocity=100),
    Note(start=4.0, pitch=72, duration=1.0, velocity=90),  # spills into bar 2
]


# ---------------------------------------------------------------------------
# MIDI summary
# ---------------------------------------------------------------------------


def test_summarize_notes_basic_stats():
    summary = summarize_notes("melody", MELODY, program=73)
    assert summary.note_count == 4
    assert (summary.pitch_min, summary.pitch_max) == (60, 72)
    assert summary.duration_beats == 5.0
    assert summary.duration_bars == 2.0  # 5 beats rounds up to 2 bars of 4/4
    assert summary.notes_per_bar == 2.0
    assert summary.program == 73
    assert summary.is_drum is False


def test_summarize_notes_empty_track():
    summary = summarize_notes("drums", [], is_drum=True)
    assert summary.note_count == 0
    assert summary.pitch_min is None and summary.pitch_max is None
    assert summary.duration_bars == 0.0
    assert summary.notes_per_bar == 0.0


def test_summarize_tracks_piece_level_stats():
    tracks = {"melody": MELODY, "drums": [Note(start=0.0, pitch=36, duration=0.25, velocity=110)]}
    summary = summarize_tracks(tracks, programs={"melody": 73}, bpm=120)
    assert summary.total_notes == 5
    assert summary.duration_bars == 2.0
    assert summary.has_drums is True
    assert summary.bpm == 120
    # Drum pitches are excluded from the pitched range.
    assert summary.pitch_range == (60, 72)


def test_summarize_arrangement_covers_all_parts():
    arrangement = DeterministicBackend().generate(
        "dark orchestral boss battle theme, 140 BPM, brass ostinato, strings, piano arps"
    )
    summary = summarize_arrangement(arrangement)
    assert {track.name for track in summary.tracks} == {"chords", "melody", "bass", "drums"}
    assert summary.total_notes > 0
    assert summary.has_drums is True
    assert summary.bpm == 140
    assert summary.duration_bars >= arrangement.bars
    assert summary.pitch_range is not None


# ---------------------------------------------------------------------------
# Piano roll
# ---------------------------------------------------------------------------


def test_piano_roll_grid_shape_and_content():
    roll = build_piano_roll(MELODY, steps_per_beat=4)
    assert roll.pitch_low == 60 and roll.pitch_high == 72
    assert roll.num_pitches == 13
    assert roll.num_steps == 20  # 5 beats * 4 steps
    assert roll.duration_beats == 5.0
    # First step: only middle C sounding, at its encoded velocity.
    assert roll.active_pitches(0) == [60]
    assert roll.grid[0][0] == 96
    # Step 8 (beat 2): the held G.
    assert roll.active_pitches(8) == [67]


def test_piano_roll_short_note_occupies_at_least_one_step():
    roll = build_piano_roll([Note(start=0.0, pitch=60, duration=0.01, velocity=80)], steps_per_beat=4)
    assert roll.num_steps >= 1
    assert roll.active_pitches(0) == [60]


def test_piano_roll_overlapping_notes_keep_loudest_velocity():
    notes = [
        Note(start=0.0, pitch=60, duration=1.0, velocity=50),
        Note(start=0.0, pitch=60, duration=0.5, velocity=110),
    ]
    roll = build_piano_roll(notes, steps_per_beat=4)
    assert roll.grid[0][0] == 110
    assert roll.grid[3][0] == 50  # after the loud note ends


def test_piano_roll_empty_notes_yield_empty_grid():
    roll = build_piano_roll([])
    assert roll.num_steps == 0
    assert roll.num_pitches == 0
    assert roll.grid == ()


def test_piano_roll_respects_explicit_pitch_bounds():
    roll = build_piano_roll(MELODY, pitch_low=60, pitch_high=67)
    assert roll.num_pitches == 8
    # The C5 (72) note is out of range and skipped; its time span still exists.
    assert roll.active_pitches(roll.num_steps - 1) == []


def test_piano_roll_rejects_invalid_config():
    with pytest.raises(ValueError):
        build_piano_roll(MELODY, steps_per_beat=0)
    with pytest.raises(ValueError):
        build_piano_roll(MELODY, pitch_low=80, pitch_high=60)
