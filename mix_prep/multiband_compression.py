from __future__ import annotations

"""Very lightweight multi-band compressor."""

import numpy as np

BANDS = [(20, 250), (250, 4000), (4000, None)]


def _bandpass(audio: np.ndarray, sr: int, low: float, high: float | None) -> np.ndarray:
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)
    mask = freqs >= low
    if high is not None:
        mask &= freqs < high
    filtered = np.zeros_like(spec)
    filtered[mask] = spec[mask]
    return np.fft.irfft(filtered, n=audio.size)


def _compress(band: np.ndarray, threshold_db: float, ratio: float) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(band))) + 1e-9
    rms_db = 20 * np.log10(rms)
    if rms_db <= threshold_db:
        return band
    gain_db = threshold_db + (rms_db - threshold_db) / ratio - rms_db
    gain = 10 ** (gain_db / 20)
    return band * gain


def multiband_compress(
    audio: np.ndarray,
    sr: int,
    thresholds: tuple[float, float, float] = (-20.0, -20.0, -20.0),
    ratios: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> np.ndarray:
    """Apply naive three-band compression to ``audio``.

    Parameters
    ----------
    audio:
        Mono audio signal.
    sr:
        Sample rate in Hz.
    thresholds:
        Thresholds in dB for low/mid/high bands.
    ratios:
        Compression ratios for the corresponding bands.
    """

    bands = []
    for (low, high), th, ra in zip(BANDS, thresholds, ratios):
        band = _bandpass(audio, sr, low, high)
        band = _compress(band, th, ra)
        bands.append(band)
    return np.sum(bands, axis=0).astype(audio.dtype)


__all__ = ["multiband_compress"]
