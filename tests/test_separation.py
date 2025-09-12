from unittest import mock

from separation import get_separator
from separation.demucs_wrapper import DemucsSeparator, separate as demucs_separate
from separation.spleeter_wrapper import (
    SpleeterSeparator,
    separate as spleeter_separate,
)


def test_factory_returns_correct_class():
    separator = get_separator("demucs")
    assert isinstance(separator, DemucsSeparator)


def test_demucs_wrapper_invokes_subprocess():
    with mock.patch("subprocess.run") as run:
        demucs_separate("input.wav", "out")
        run.assert_called_once()
        cmd = run.call_args[0][0]
        assert cmd[0] == "demucs"
        assert "-o" in cmd
        assert cmd[-1] == "input.wav"


def test_spleeter_wrapper_invokes_subprocess():
    with mock.patch("subprocess.run") as run:
        spleeter_separate("track.mp3", "out")
        run.assert_called_once()
        cmd = run.call_args[0][0]
        assert cmd[0] == "spleeter" and cmd[1] == "separate"
        assert "-o" in cmd
        assert cmd[-1] == "track.mp3"


def test_class_methods_call_subprocess():
    sep = SpleeterSeparator()
    with mock.patch("subprocess.run") as run:
        sep.separate("song.wav", "out")
        run.assert_called_once()
