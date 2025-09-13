"""Simple automatic EQ placeholder."""
from __future__ import annotations

from pathlib import Path
from typing import Union


def apply_auto_eq(stems_dir: Union[str, Path]) -> Path:
    """Pretend to apply automatic EQ to stems in ``stems_dir``.

    Parameters
    ----------
    stems_dir:
        Directory containing stem audio files.

    Returns
    -------
    Path
        The path to the directory after processing. Currently this is
        just ``stems_dir`` returned unchanged as this module is a
        lightweight placeholder.
    """

    return Path(stems_dir)


__all__ = ["apply_auto_eq"]
