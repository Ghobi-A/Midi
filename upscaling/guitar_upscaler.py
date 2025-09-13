"""Guitar track upscaling utilities.

The current implementation uses the pre-trained `HDemucs` source-separation
network from :mod:`torchaudio` to extract and enhance the guitar stem from a
mix.  Input audio must be stereo at 48 kHz; internally it is resampled to the
model's native 44.1 kHz sample rate and the ``"other"`` stem is returned as the
enhanced track.

The module exposes a single :func:`enhance` function.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from .utils import load_model, model_sample_rate

_MODEL_NAME = "strumlift_v1"


def enhance(input_path: str | Path, output_path: str | Path) -> str:
    """Enhance a guitar recording.

    Parameters
    ----------
    input_path:
        Path to a **stereo 48 kHz** WAV file containing the guitar track.
    output_path:
        Destination path where the enhanced stereo audio will be written.  The
        result is also at 48 kHz.

    Returns
    -------
    str
        The ``output_path`` where the enhanced audio was saved.
    """

    model = load_model(_MODEL_NAME)
    waveform, sr = torchaudio.load(str(input_path))
    if sr != 48_000 or waveform.shape[0] != 2:
        raise ValueError("Guitar enhancer expects stereo 48 kHz audio")

    model_sr = model_sample_rate(_MODEL_NAME)
    resampled = torchaudio.functional.resample(waveform, sr, model_sr)

    with torch.no_grad():
        sources = model(resampled.unsqueeze(0))

    stem_index = model.sources.index("other")
    guitar = sources[0, stem_index]

    guitar = torchaudio.functional.resample(guitar, model_sr, sr)
    torchaudio.save(str(output_path), guitar, sr)
    return str(output_path)
