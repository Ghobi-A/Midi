"""Shared helpers for audio upscaling models.

This module centralises functionality used by the upscaling utilities such as
downloading and instantiating pre-trained neural networks.  The helpers wrap
``torchaudio`` pipelines so that individual upscalers only need to call
``load_model`` and work with a regular :class:`torch.nn.Module`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torchaudio

# Mapping from the public model names used by the modules to the corresponding
# ``torchaudio`` bundle.  Both the guitar and vocal upscalers currently share
# the same HDemucs model, but this dictionary makes it easy to swap out the
# backend in the future.
_MODEL_BUNDLES: dict[str, Any] = {
    "strumlift_v1": torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS,
    "ultravox_v1": torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS,
}


@lru_cache()
def load_model(name: str):
    """Load and cache an upscaling model by name.

    The first time a model is requested the corresponding weights are
    downloaded via the :mod:`torchaudio` pipeline.  Subsequent calls reuse the
    cached model instance.

    Parameters
    ----------
    name:
        Identifier of the model to load.  See :data:`_MODEL_BUNDLES`.

    Returns
    -------
    torch.nn.Module
        The instantiated model ready for inference.

    Raises
    ------
    ValueError
        If ``name`` does not correspond to a known model.
    """

    try:
        bundle = _MODEL_BUNDLES[name]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unknown model '{name}'") from exc

    return bundle.get_model().eval()


def model_sample_rate(name: str) -> int:
    """Return the sample rate expected by the given model."""

    try:
        bundle = _MODEL_BUNDLES[name]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unknown model '{name}'") from exc

    return bundle.sample_rate


def ensure_downloaded(name: str) -> None:
    """Ensure that the model weights for ``name`` are downloaded.

    This is a small convenience wrapper around :func:`load_model` that simply
    triggers the download without returning the model.  It mirrors the
    semantics of helpers in other parts of the codebase and is mainly provided
    so that tests can prefetch weights if desired.
    """

    load_model(name)

