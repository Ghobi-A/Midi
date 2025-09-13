"""Utilities for preparing tracks before mixing."""

from .auto_eq import auto_eq, match_spectrum
from .multiband_compression import multiband_compress
from .noise_reduction import spectral_gate
from .stereo_imaging import stereo_widen, to_stereo

__all__ = [
    "auto_eq",
    "match_spectrum",
    "multiband_compress",
    "spectral_gate",
    "stereo_widen",
    "to_stereo",
]
