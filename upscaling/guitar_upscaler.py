"""Guitar track upscaling using a pre-trained RAVE model."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf

from .utils import load_model, resample

_MODEL = None


def _get_model():
    """Lazily load the RAVE model used for guitar enhancement."""
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model("rave")
    return _MODEL


def enhance(input_path: str | Path, output_path: str | Path, *, target_sr: int = 44100) -> None:
    """Enhance a guitar recording and write it to ``output_path``."""
    audio, sr = sf.read(str(input_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = resample(audio, sr, target_sr)
    model = _get_model()
    enhanced = model(audio, target_sr)
    sf.write(str(output_path), enhanced, target_sr)
