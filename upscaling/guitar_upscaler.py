"""Guitar track upscaling utilities.

Model source: `StrumLift` – an imaginary model trained on studio guitar
recordings hosted at https://example.com/models/strumlift.  The model expects
stereo 48 kHz WAV files as input and produces an enhanced track at the same
sample rate.

The module exposes a single :func:`enhance` function.  The current
implementation loads the model and simply copies the input to the output path.
"""

from __future__ import annotations

from pathlib import Path
import shutil

from .utils import load_model

_MODEL_NAME = "strumlift_v1"


def enhance(input_path: str | Path, output_path: str | Path) -> str:
    """Enhance a guitar recording.

    Parameters
    ----------
    input_path:
        Path to a stereo 48 kHz WAV file containing the guitar track.
    output_path:
        Destination path for the enhanced audio.

    Returns
    -------
    str
        The ``output_path`` where the enhanced audio was saved.
    """

    load_model(_MODEL_NAME)
    shutil.copyfile(input_path, output_path)
    return str(output_path)
