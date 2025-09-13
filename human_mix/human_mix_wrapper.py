from __future__ import annotations

"""Helpers for human-guided mixing workflows.

This module connects the reference mix analysis utilities with reusable
effect-chain templates.  It intentionally implements only lightweight
processing suitable for examples and tests.
"""

from typing import Dict, List

import numpy as np

from .reference_mixer import compare_to_reference
from .effect_chain_templates import Effect, EffectChain, get_template


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def analyze(reference_path: str, stems: Dict[str, str]) -> Dict[str, List[str]]:
    """Return mix suggestions by comparing stems to a reference track."""

    return compare_to_reference(reference_path, stems)


def available_templates() -> List[str]:
    """List the names of built-in effect chain templates."""

    from .effect_chain_templates import TEMPLATES

    return sorted(TEMPLATES.keys())


def apply_template(audio: np.ndarray, sr: int, template: str) -> np.ndarray:
    """Apply a named effect-chain template to ``audio``."""

    chain = get_template(template)
    return apply_chain(audio, sr, chain)


def apply_chain(audio: np.ndarray, sr: int, chain: EffectChain) -> np.ndarray:
    """Apply a sequence of :class:`~Effect` instances to ``audio``."""

    processed = audio.copy()
    for effect in chain:
        processed = _apply_effect(processed, sr, effect)
    return processed


# ---------------------------------------------------------------------------
# Effect implementations (very naive)
# ---------------------------------------------------------------------------

def _apply_effect(audio: np.ndarray, sr: int, effect: Effect) -> np.ndarray:
    name = effect.name.lower()
    params = effect.params
    if name == "highpass":
        return _highpass(audio, sr, params.get("freq", 100))
    if name == "compressor":
        return _compressor(audio, params.get("ratio", 2.0))
    if name == "de-esser":
        return _de_esser(audio, sr, params.get("freq", 6000))
    if name == "eq":
        return _eq(audio, sr, params.get("freq", 1000), params.get("gain", 0.0))
    if name == "reverb":
        return _reverb(audio, sr, params.get("room_size", 0.3), params.get("wet", 0.3))
    return audio


def _highpass(audio: np.ndarray, sr: int, freq: float) -> np.ndarray:
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)
    spec[freqs < freq] = 0
    return np.fft.irfft(spec, n=audio.size).astype(audio.dtype)


def _compressor(audio: np.ndarray, ratio: float) -> np.ndarray:
    thr = np.sqrt(np.mean(np.square(audio))) + 1e-9
    out = audio.copy()
    above = np.abs(audio) > thr
    out[above] = np.sign(audio[above]) * (thr + (np.abs(audio[above]) - thr) / ratio)
    return out.astype(audio.dtype)


def _de_esser(audio: np.ndarray, sr: int, freq: float) -> np.ndarray:
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)
    spec[freqs > freq] *= 0.5
    return np.fft.irfft(spec, n=audio.size).astype(audio.dtype)


def _eq(audio: np.ndarray, sr: int, freq: float, gain: float) -> np.ndarray:
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sr)
    bandwidth = max(freq * 0.1, 1.0)
    band = (freqs >= freq - bandwidth) & (freqs <= freq + bandwidth)
    spec[band] *= 10 ** (gain / 20)
    return np.fft.irfft(spec, n=audio.size).astype(audio.dtype)


def _reverb(audio: np.ndarray, sr: int, room_size: float, wet: float) -> np.ndarray:
    delay = max(int(sr * room_size * 0.03), 1)
    wet_sig = np.zeros(audio.size + delay)
    wet_sig[delay:] = audio * wet
    dry_sig = np.zeros_like(wet_sig)
    dry_sig[: audio.size] = audio * (1 - wet)
    out = dry_sig
    out[: wet_sig.size] += wet_sig[: out.size]
    return out[: audio.size].astype(audio.dtype)


__all__ = ["analyze", "available_templates", "apply_template", "apply_chain"]
