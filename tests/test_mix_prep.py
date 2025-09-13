import numpy as np

from mix_prep.auto_eq import auto_eq
from mix_prep.multiband_compression import multiband_compress
from mix_prep.stereo_imaging import stereo_widen


def _band_energy(audio: np.ndarray, sr: int, low: float, high: float) -> float:
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)
    mask = (freqs >= low) & (freqs < high)
    return np.mean(np.abs(spec[mask]))


def test_auto_eq_balances_spectrum():
    sr = 44100
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(sr)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(noise.size, 1.0 / sr)
    spec[freqs > 2000] *= 0.1  # dull high end
    dull = np.fft.irfft(spec, n=noise.size)
    processed = auto_eq(dull, sr)
    before = _band_energy(dull, sr, 4000, 8000) / _band_energy(dull, sr, 20, 2000)
    after = _band_energy(processed, sr, 4000, 8000) / _band_energy(processed, sr, 20, 2000)
    assert after > before


def test_multiband_compress_reduces_level():
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    compressed = multiband_compress(
        audio,
        sr,
        thresholds=(-30.0, -30.0, -30.0),
        ratios=(10.0, 1.0, 1.0),
    )
    orig_rms = np.sqrt(np.mean(audio**2))
    new_rms = np.sqrt(np.mean(compressed**2))
    assert new_rms < orig_rms


def test_stereo_widen_controls_side_energy():
    rng = np.random.default_rng(1)
    audio = rng.standard_normal((1000, 2)).astype(np.float32)
    mono = stereo_widen(audio, width=0.0)
    assert np.allclose(mono[:, 0], mono[:, 1])
    widened = stereo_widen(audio, width=2.0)
    orig_diff = np.mean((audio[:, 0] - audio[:, 1]) ** 2)
    new_diff = np.mean((widened[:, 0] - widened[:, 1]) ** 2)
    assert new_diff > orig_diff
