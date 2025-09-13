from __future__ import annotations

"""Simple automatic equalization utilities."""

import numpy as np


def auto_eq(audio: np.ndarray, sr: int) -> np.ndarray:
    """Perform a crude inverse-EQ to flatten the spectrum of ``audio``.

    Parameters
    ----------
    audio:
        Mono audio signal as a 1-D numpy array.
    sr:
        Sampling rate in Hz.

    Returns
    -------
    numpy.ndarray
        Equalized audio of the same shape as ``audio``.
    """

    spectrum = np.fft.rfft(audio)
    magnitude = np.abs(spectrum)
    # Smooth the spectrum to obtain a basic target curve
    smooth = np.convolve(magnitude, np.ones(5) / 5, mode="same")
    target = np.mean(smooth)
    correction = target / (smooth + 1e-9)
    corrected = spectrum * correction
    return np.fft.irfft(corrected, n=audio.size).astype(audio.dtype)


def match_spectrum(source: np.ndarray, reference: np.ndarray, sr: int) -> np.ndarray:
    """Match the spectral envelope of ``source`` to ``reference``.

    This uses a simplistic FFT magnitude transfer and should be considered a
    placeholder for more advanced matching EQ algorithms.
    """

    spec_src = np.fft.rfft(source)
    spec_ref = np.fft.rfft(reference, n=source.size)
    gain = (np.abs(spec_ref) + 1e-9) / (np.abs(spec_src) + 1e-9)
    matched = spec_src * gain
    return np.fft.irfft(matched, n=source.size).astype(source.dtype)


__all__ = ["auto_eq", "match_spectrum"]
