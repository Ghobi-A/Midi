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
