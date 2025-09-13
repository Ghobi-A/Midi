"""Placeholder helpers for mix preparation steps.

The real implementation will organise and clean up separated stems before they
are sent to a mixing stage. For now, :func:`prepare_mix` simply returns the
directory containing the stems so that higher-level workflows can be wired up
without errors.
"""

from __future__ import annotations


def prepare_mix(stems_dir: str) -> str:
    """Prepare stems for mixing (stub).

    Parameters
    ----------
    stems_dir:
        Directory containing separated stems.

    Returns
    -------
    str
        The same ``stems_dir`` path for now.
    """

    # Future implementations might normalise levels, trim silence, etc.
    return stems_dir


__all__ = ["prepare_mix"]

