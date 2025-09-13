"""Vocal upscaling utilities.

The implementation uses the same pre-trained `HDemucs` model as the guitar
upscaler.  The network extracts the ``"vocals"`` stem from a mixture.  Inputs
must be mono 44.1 kHz WAV files; the model operates on stereo audio so the
single channel is duplicated prior to inference.

The main entry point is :func:`enhance`, which loads the model and writes the
processed audio to disk.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from .utils import load_model, model_sample_rate

_MODEL_NAME = "ultravox_v1"


def enhance(input_path: str | Path, output_path: str | Path) -> str:
    """Enhance a vocal recording.

    Parameters
    ----------
    input_path:
        Path to a **mono 44.1 kHz** WAV file containing the vocals.
    output_path:
        Destination path where the enhanced audio will be written.  The result
        is mono 44.1 kHz.

    Returns
    -------
    str
        The ``output_path`` where the enhanced audio was saved.
    """

    model = load_model(_MODEL_NAME)
    waveform, sr = torchaudio.load(str(input_path))
    if sr != 44_100 or waveform.shape[0] != 1:
        raise ValueError("Vocal enhancer expects mono 44.1 kHz audio")

    model_sr = model_sample_rate(_MODEL_NAME)
    if sr != model_sr:
        waveform = torchaudio.functional.resample(waveform, sr, model_sr)
        sr = model_sr

    # HDemucs expects stereo input; duplicate the channel for processing.
    stereo = waveform.repeat(2, 1)

    with torch.no_grad():
        sources = model(stereo.unsqueeze(0))

    stem_index = model.sources.index("vocals")
    vocals = sources[0, stem_index]

    mono = vocals.mean(dim=0, keepdim=True)
    torchaudio.save(str(output_path), mono, sr)
    return str(output_path)
