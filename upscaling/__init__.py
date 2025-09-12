"""Audio upscaling helpers.

This package exposes simple wrappers around pre-trained models for
raising the fidelity of audio tracks.  The :mod:`voc_upscaler` and
:mod:`guitar_upscaler` modules each provide an :func:`enhance` function.
Shared utilities live in :mod:`upscaling.utils`.
"""
__all__ = ["voc_upscaler", "guitar_upscaler", "utils"]
