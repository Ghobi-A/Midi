"""Spleeter based source separation."""

from __future__ import annotations

from pathlib import Path


def separate(input_path: str | Path, output_dir: str | Path) -> Path:
    """Separate audio into stems using `spleeter`.

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
    This function requires the optional :mod:`spleeter` package.  A
    :class:`ImportError` will be raised if it is not available.
    """

    try:
        from spleeter.separator import Separator
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Spleeter is required for this separation backend. "
            "Install it with `pip install spleeter`."
        ) from exc

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the default 2-stem configuration (vocals/accompaniment).
    separator = Separator("spleeter:2stems")
    separator.separate_to_file(str(input_path), str(output_dir))

    stem_dir = output_dir / input_path.stem
    return stem_dir if stem_dir.exists() else output_dir

