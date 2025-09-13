"""Audio source separation utilities."""

from . import demucs_wrapper, spleeter_wrapper


def get_separator(backend: str = "demucs"):
    """Return a separation function for the requested backend.

    Parameters
    ----------
    backend: str
        Name of the backend to use ("demucs" or "spleeter").
    """
    backend = backend.lower()
    if backend == "demucs":
        return demucs_wrapper.separate
    if backend == "spleeter":
        return spleeter_wrapper.separate
    raise ValueError(f"Unknown separation backend: {backend}")
