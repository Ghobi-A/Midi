"""Load MIDI files already present on local disk. Never downloads anything.

Intended as the ingestion step of the future ML pipeline described in
docs/ML_ROADMAP.md: point it at a directory of MIDI files you have already
sourced (and whose rights you've checked — see docs/DATASETS.md), and it
returns parsed note events ready for feature extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Union

from ..midi_parser import parse_midi_file
from ..music_theory import Note


def iter_midi_paths(directory: Union[str, Path]) -> Iterator[Path]:
    """Yield every ``.mid``/``.midi`` file under ``directory`` (recursively)."""
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory {root} does not exist. This loader never downloads data — "
            "point it at a local directory of MIDI files you've already sourced."
        )
    for pattern in ("*.mid", "*.midi"):
        yield from sorted(root.rglob(pattern))


def load_dataset_notes(directory: Union[str, Path]) -> Dict[str, Dict[str, List[Note]]]:
    """Parse every MIDI file under ``directory`` into ``{filename: {track_name: [Note, ...]}}``."""
    dataset: Dict[str, Dict[str, List[Note]]] = {}
    for path in iter_midi_paths(directory):
        dataset[path.name] = parse_midi_file(path)
    return dataset


__all__ = ["iter_midi_paths", "load_dataset_notes"]
