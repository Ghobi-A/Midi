"""Shared helpers for audio upscaling models.

This module centralizes common functionality such as model loading.  The
actual models are placeholders; in a real project these functions would load
pre-trained neural networks from disk or download them on demand.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache()
def load_model(name: str) -> dict[str, Any]:
    """Load and cache an upscaling model by name.

    Parameters
    ----------
    name:
        Identifier of the model to load.

    Returns
    -------
    dict[str, Any]
        Placeholder object representing the loaded model.
    """

    # In this stub implementation we simply return a dictionary describing the
    # model.  Real implementations would perform I/O here.
    return {"model": name}
