"""Tools for comparing stems against a reference mix.

This module provides helpers that analyse loudness and crude EQ curves for
multiple stems and a reference track. It is intentionally lightweight and is
meant as a starting point for human-in-the-loop mix workflows.
"""

from __future__ import annotations

from typing import Dict, List

import librosa
import numpy as np


def _rms_db(audio: np.ndarray) -> float:
    """Return the RMS loudness of ``audio`` in dBFS."""
    rms = np.sqrt(np.mean(np.square(audio)))
    return 20 * np.log10(rms + 1e-9)


def _eq_bands(audio: np.ndarray, sr: int) -> np.ndarray:
    """Calculate simple low/mid/high energy bands for ``audio``."""
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)
    bands = [(20, 250), (250, 4000), (4000, sr / 2)]
    energies = []
    for low, high in bands:
        idx = (freqs >= low) & (freqs < high)
        energies.append(float(np.mean(spectrum[idx]) if np.any(idx) else 0.0))
    return np.array(energies)


def compare_to_reference(reference_path: str, stem_paths: Dict[str, str]) -> Dict[str, List[str]]:
    """Compare stems to a reference track and return mix suggestions.

    Parameters
    ----------
    reference_path:
        Path to the reference mix.
    stem_paths:
        Mapping of stem name to audio file path.

    Returns
    -------
    dict
        A mapping of stem name to a list of textual suggestions.
    """

    ref_audio, sr = librosa.load(reference_path, sr=None, mono=True)
    ref_loudness = _rms_db(ref_audio)
    ref_eq = _eq_bands(ref_audio, sr)

    band_names = ["low", "mid", "high"]
    suggestions: Dict[str, List[str]] = {}

    for name, path in stem_paths.items():
        audio, _ = librosa.load(path, sr=sr, mono=True)
        loudness = _rms_db(audio)
        eq = _eq_bands(audio, sr)

        stem_suggestions: List[str] = []

        loud_diff = ref_loudness - loudness
        if abs(loud_diff) > 1.0:
            direction = "Increase" if loud_diff > 0 else "Decrease"
            stem_suggestions.append(
                f"{direction} loudness by approximately {abs(loud_diff):.1f} dB"
            )

        for band_name, ref_val, stem_val in zip(band_names, ref_eq, eq):
            diff = ref_val - stem_val
            threshold = max(ref_val, 1e-9) * 0.1
            if abs(diff) > threshold:
                action = "boost" if diff > 0 else "cut"
                stem_suggestions.append(f"{action} {band_name} frequencies")

        if not stem_suggestions:
            stem_suggestions.append("Stem is similar to reference.")
        suggestions[name] = stem_suggestions

    return suggestions
