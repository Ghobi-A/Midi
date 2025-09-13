"""Basic loudness normalization and limiting for audio tracks."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf

try:  # pragma: no cover - optional dependency
    import pyloudnorm as pyln
except Exception:  # pragma: no cover
    pyln = None


def _integrated_loudness(audio: np.ndarray, rate: int) -> float:
    """Return integrated loudness of ``audio`` in LUFS.

    Parameters
    ----------
    audio:
        Audio samples with shape ``(n_samples, n_channels)`` or ``(n_samples,)``.
    rate:
        Sampling rate in Hz.
    """
    if pyln is None:
        # Fallback: use simple RMS-based approximation
        rms = np.sqrt(np.mean(np.square(audio)))
        return 20 * np.log10(rms + 1e-12)
    meter = pyln.Meter(rate)
    return meter.integrated_loudness(audio)


def master_track(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_lufs: float = -14.0,
) -> None:
    """Normalize ``input_path`` to ``target_lufs`` and apply peak limiting.

    Parameters
    ----------
    input_path:
        Path to the input audio file.
    output_path:
        Where the mastered audio will be written.
    target_lufs:
        Desired loudness level in LUFS. Defaults to -14, a common streaming target.
    """
    data, rate = sf.read(str(input_path))
    multi_channel = data.ndim > 1
    if multi_channel and pyln is not None:
        # pyloudnorm expects channels first
        audio = data.T
    else:
        audio = data
    loudness = _integrated_loudness(audio, rate)
    gain_db = target_lufs - loudness
    gain = 10 ** (gain_db / 20)
    normalized = audio * gain
    limited = np.clip(normalized, -1.0, 1.0)
    if multi_channel and pyln is not None:
        limited = limited.T
    sf.write(str(output_path), limited, rate)


def main(argv: list[str] | None = None) -> None:
    """Entry point for command line execution."""
    parser = argparse.ArgumentParser(description="Simple AI mastering")
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("output", help="Path for mastered output")
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-14.0,
        help="Target loudness in LUFS (default: -14)",
    )
    args = parser.parse_args(argv)
    master_track(args.input, args.output, args.target_lufs)
    print(f"Mastered file written to {args.output}")


__all__ = ["master_track", "main"]


if __name__ == "__main__":  # pragma: no cover
    main()
