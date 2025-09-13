from __future__ import annotations

"""Stereo imaging helpers."""

import numpy as np


def to_stereo(audio: np.ndarray) -> np.ndarray:
    """Ensure ``audio`` has two channels by duplicating if necessary."""
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=-1)
    return audio


def stereo_widen(audio: np.ndarray, width: float = 1.0) -> np.ndarray:
    """Adjust stereo width using mid/side processing.

    Parameters
    ----------
    audio:
        Stereo (``n_samples, 2``) or mono (``n_samples``) array.
    width:
        Side gain factor. ``1`` leaves the signal unchanged, ``>1`` widens,
        ``<1`` narrows.
    """

    stereo = to_stereo(audio)
    mid = (stereo[:, 0] + stereo[:, 1]) / 2.0
    side = (stereo[:, 0] - stereo[:, 1]) / 2.0
    side *= width
    left = mid + side
    right = mid - side
    return np.stack([left, right], axis=-1).astype(stereo.dtype)


__all__ = ["stereo_widen", "to_stereo"]
