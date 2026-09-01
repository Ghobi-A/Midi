"""Load MIDI files already present on local disk. Never downloads anything.

Intended as the ingestion step of the ML pipeline described in
docs/ML_ROADMAP.md: point it at a directory of MIDI files you have already
sourced (and whose rights you've checked — see docs/DATASETS.md), and it
returns parsed note events ready for feature extraction.

Pieces are identified by their path *relative to the dataset root*, not by
filename: two files both called ``song.mid`` in different subdirectories
are two different compositions and must not overwrite each other. Each file
also carries a content hash, so a corpus built from it can be pinned in a
model artefact and re-verified later.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Union

from ..midi_parser import parse_midi_file
from ..music_theory import Note

#: Bytes read per chunk when hashing a file.
_HASH_CHUNK = 65536


@dataclass(frozen=True)
class DatasetFile:
    """One MIDI file in a dataset directory, with a stable identity."""

    #: Path relative to the dataset root, POSIX-style — the composition id.
    composition_id: str
    path: Path
    sha256: str


def file_sha256(path: Union[str, Path]) -> str:
    """SHA-256 of a file's contents, streamed so large files stay cheap."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_hash(hashes: Sequence[str]) -> str:
    """A single stable hash over a set of file hashes (order-independent)."""
    digest = hashlib.sha256()
    for value in sorted(hashes):
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def iter_midi_paths(directory: Union[str, Path]) -> Iterator[Path]:
    """Yield every ``.mid``/``.midi`` file under ``directory`` (recursively)."""
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory {root} does not exist. This loader never downloads data — "
            "point it at a local directory of MIDI files you've already sourced."
        )
    seen = set()
    for pattern in ("*.mid", "*.midi"):
        for path in sorted(root.rglob(pattern)):
            if path not in seen:
                seen.add(path)
                yield path


def iter_dataset_files(directory: Union[str, Path]) -> Iterator[DatasetFile]:
    """Yield every MIDI file under ``directory`` with its relative id and hash."""
    root = Path(directory)
    for path in iter_midi_paths(root):
        yield DatasetFile(
            composition_id=path.relative_to(root).as_posix(),
            path=path,
            sha256=file_sha256(path),
        )


def load_dataset_notes(directory: Union[str, Path]) -> Dict[str, Dict[str, List[Note]]]:
    """Parse every MIDI file under ``directory`` into ``{composition_id: {track: [Note]}}``.

    The key is the path relative to ``directory`` (e.g. ``"bach/song.mid"``),
    so identically-named files in different subdirectories both survive.
    """
    return {
        entry.composition_id: parse_midi_file(entry.path)
        for entry in iter_dataset_files(directory)
    }


__all__ = [
    "DatasetFile",
    "corpus_hash",
    "file_sha256",
    "iter_dataset_files",
    "iter_midi_paths",
    "load_dataset_notes",
]
