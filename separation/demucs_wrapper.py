"""Demucs based source separation."""

from __future__ import annotations

from pathlib import Path


def separate(input_path: str | Path, output_dir: str | Path) -> Path:
    """Separate audio into stems using `demucs`.

    Parameters
    ----------
    input_path:
        Path to the audio file to separate.
    output_dir:
        Directory where the separated stems will be written.

    Returns
    -------
    pathlib.Path
        Directory containing the separated stems.

    Notes
    -----
    This function requires the optional :mod:`demucs` package.  A
    :class:`ImportError` will be raised if it is not available.
    """

    try:
        import demucs.separate as demucs_separate
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Demucs is required for this separation backend. "
            "Install it with `pip install demucs`."
        ) from exc

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run Demucs separation using its CLI entry point.
    # This mirrors running ``python -m demucs.separate``.
    demucs_separate.main(["--out", str(output_dir), str(input_path)])

    # Demucs creates ``<output_dir>/<model>/<track_name>``.  Return the
    # directory containing the final stems if possible.
    model_dirs = sorted(output_dir.iterdir())
    if model_dirs:
        stem_dir = model_dirs[0] / input_path.stem
        if stem_dir.exists():
            return stem_dir
    return output_dir

