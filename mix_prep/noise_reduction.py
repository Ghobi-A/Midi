from __future__ import annotations

"""Basic spectral noise reduction."""

import numpy as np


def spectral_gate(
    audio: np.ndarray,
    sr: int,
    noise_threshold_db: float = -40.0,
    reduction_db: float = 20.0,
) -> np.ndarray:
    """Attenuate frequency bins below ``noise_threshold_db``.

    Parameters
    ----------
    audio:
        Mono audio signal.
    sr:
        Sampling rate (present for API symmetry; not used).
    noise_threshold_db:
        Magnitude threshold in dB below which frequencies are considered noise.
    reduction_db:
        Amount of attenuation applied to noisy bins.
    """

    spectrum = np.fft.rfft(audio)
    magnitude = np.abs(spectrum)
    mag_db = 20 * np.log10(magnitude + 1e-9)
    mask = mag_db < noise_threshold_db
    spectrum[mask] *= 10 ** (-reduction_db / 20)
    cleaned = np.fft.irfft(spectrum, n=audio.size)
    return cleaned.astype(audio.dtype)


__all__ = ["spectral_gate"]
