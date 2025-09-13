"""Audio source separation utilities.

This module exposes thin wrappers around popular source separation
backends.  Currently supported backends are:

* ``demucs`` – requires the :mod:`demucs` package.
* ``spleeter`` – requires the :mod:`spleeter` package.

Use :func:`get_separator` to obtain a callable for the desired backend.
"""

from . import demucs_wrapper, spleeter_wrapper


def get_separator(backend: str = "demucs"):
    """Return a separation function for the requested backend.

    Parameters
    ----------
    backend: str
        Name of the backend to use (``"demucs"`` or ``"spleeter"``).

    Returns
    -------
    Callable
        Function that performs separation on ``(input_path, output_dir)``.

    Notes
    -----
    The chosen backend must have its optional dependency installed.
    """
    backend = backend.lower()
    if backend == "demucs":
        return demucs_wrapper.separate
    if backend == "spleeter":
        return spleeter_wrapper.separate
    raise ValueError(f"Unknown separation backend: {backend}")
