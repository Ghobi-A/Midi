"""Vocal upscaling utilities.

Model source: `UltraVox` – a fictional neural network for vocal enhancement
available from https://example.com/models/ultravox.  The model expects mono
44.1 kHz WAV files as input and outputs an enhanced version at the same sample
rate.

The main entry point is :func:`enhance`, which loads the model and writes the
processed audio to disk.  In this placeholder implementation the input file is
simply copied to the destination.
"""

from __future__ import annotations

from pathlib import Path
import shutil

from .utils import load_model

_MODEL_NAME = "ultravox_v1"


def enhance(input_path: str | Path, output_path: str | Path) -> str:
    """Enhance a vocal recording.

    Parameters
    ----------
    input_path:
        Path to a mono 44.1 kHz WAV file containing the vocals.
    output_path:
        Destination path where the enhanced audio will be written.

    Returns
    -------
    str
        The ``output_path`` where the enhanced audio was saved.
    """

    load_model(_MODEL_NAME)
    shutil.copyfile(input_path, output_path)
    return str(output_path)
