"""Vocal track upscaling via a pre-trained DDSP model.

The main entry point is :func:`enhance` which takes an input audio file
and writes an enhanced version to the requested output path.  Model
loading and resampling logic are delegated to :mod:`upscaling.utils`.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf

from .utils import load_model, resample

_MODEL = None


def _get_model():
    """Lazily load the DDSP model used for vocal enhancement."""
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model("ddsp")
    return _MODEL


def enhance(input_path: str | Path, output_path: str | Path, *, target_sr: int = 44100) -> None:
    """Enhance a vocal recording.

    Parameters
    ----------
    input_path:
        Location of the source audio file.
    output_path:
        Path where the enhanced audio will be written.
    target_sr:
        Desired sample rate for the output.  ``44100`` Hz by default.
    """

    audio, sr = sf.read(str(input_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = resample(audio, sr, target_sr)
    model = _get_model()
    enhanced = model(audio, target_sr)
    sf.write(str(output_path), enhanced, target_sr)
