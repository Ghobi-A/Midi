"""Tests for the optional MidiTok/symusic REMI adapter.

Split in two: the tests that assert the *absence* path (a clear error and
no heavy imports) run when the extras are missing, and the tests that
actually tokenize a MIDI file run when they are installed — which is what
the `symbolic-extras` CI job exists to exercise. Without the second group,
that job would install MidiTok and then test nothing about it.
"""

import pytest

from creative_audio_lab.tokenization.miditok_adapter import (
    MidiTokNotAvailableError,
    MidiTokRemiAdapter,
    is_miditok_available,
    require_miditok,
)

miditok_missing = not is_miditok_available()


@pytest.mark.skipif(not miditok_missing, reason="miditok/symusic are installed in this environment")
def test_adapter_raises_clear_error_when_miditok_missing():
    with pytest.raises(MidiTokNotAvailableError) as excinfo:
        MidiTokRemiAdapter()
    message = str(excinfo.value)
    assert "creative-audio-lab[symbolic]" in message
    assert "SymbolicTokenizer" in message  # points at the dependency-free alternative


@pytest.mark.skipif(not miditok_missing, reason="miditok/symusic are installed in this environment")
def test_require_miditok_raises_when_missing():
    with pytest.raises(MidiTokNotAvailableError):
        require_miditok()


def test_error_is_an_import_error_subclass():
    # Callers guarding with `except ImportError` keep working.
    assert issubclass(MidiTokNotAvailableError, ImportError)


def test_availability_probe_does_not_import_heavy_modules():
    import sys

    is_miditok_available()
    if miditok_missing:
        assert "miditok" not in sys.modules
        assert "symusic" not in sys.modules


# ---------------------------------------------------------------------------
# The installed path: these are the tests the symbolic-extras CI job runs.
# ---------------------------------------------------------------------------

requires_miditok = pytest.mark.skipif(
    miditok_missing, reason='needs the optional extras: pip install -e ".[symbolic]"'
)


@pytest.fixture
def midi_file(tmp_path):
    """A short monophonic MIDI file written by this project's own exporter."""
    from creative_audio_lab.export.midi_export import export_notes_to_bytes
    from creative_audio_lab.music_theory import Note

    notes = [
        Note(start=i * 0.5, pitch=60 + (i % 12), duration=0.5, velocity=88)
        for i in range(24)
    ]
    path = tmp_path / "melody.mid"
    path.write_bytes(export_notes_to_bytes(notes, name="melody", program=0, bpm=120.0))
    return path


@requires_miditok
def test_require_miditok_passes_when_installed():
    require_miditok()  # must not raise


@requires_miditok
def test_adapter_exposes_a_real_remi_vocabulary():
    assert MidiTokRemiAdapter().vocab_size > 0


@requires_miditok
def test_tokenize_file_returns_remi_token_strings(midi_file):
    tokens = MidiTokRemiAdapter().tokenize_file(midi_file)
    assert tokens
    assert all(isinstance(token, str) for token in tokens)
    # REMI anchors notes to bar/position and emits pitch/velocity/duration,
    # the same shape the internal SymbolicTokenizer produces.
    for prefix in ("Bar", "Position", "Pitch", "Velocity", "Duration"):
        assert any(token.startswith(prefix) for token in tokens), prefix


@requires_miditok
def test_tokenize_file_accepts_a_string_path(midi_file):
    adapter = MidiTokRemiAdapter()
    assert adapter.tokenize_file(str(midi_file)) == adapter.tokenize_file(midi_file)


@requires_miditok
def test_tokenizer_params_are_forwarded_to_miditok():
    default = MidiTokRemiAdapter()
    coarse = MidiTokRemiAdapter({"num_velocities": 8})
    assert coarse.vocab_size != default.vocab_size


@requires_miditok
def test_multitrack_files_are_flattened_track_by_track(tmp_path):
    import mido

    from creative_audio_lab.export.midi_export import export_notes_to_bytes
    from creative_audio_lab.music_theory import Note

    def line(pitch_base):
        return [
            Note(start=i * 0.5, pitch=pitch_base + (i % 5), duration=0.5, velocity=88)
            for i in range(16)
        ]

    single = tmp_path / "single.mid"
    single.write_bytes(export_notes_to_bytes(line(60), name="a", program=0, bpm=120.0))

    merged = mido.MidiFile(single)
    second = mido.MidiFile(
        file=__import__("io").BytesIO(
            export_notes_to_bytes(line(48), name="b", program=32, bpm=120.0)
        )
    )
    merged.tracks.extend(second.tracks)
    double = tmp_path / "double.mid"
    merged.save(double)

    adapter = MidiTokRemiAdapter()
    assert len(adapter.tokenize_file(double)) > len(adapter.tokenize_file(single))
