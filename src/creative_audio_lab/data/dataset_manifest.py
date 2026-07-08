"""Schema and recommended entries for future ML training datasets.

This module never downloads anything — see docs/DATASETS.md for licensing
and usage guidance on each recommended dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union


@dataclass(frozen=True)
class DatasetEntry:
    """Describes one dataset a future training pipeline could ingest."""

    name: str
    description: str
    url: str
    license_notes: str
    local_path: str | None = None


RECOMMENDED_DATASETS: List[DatasetEntry] = [
    DatasetEntry(
        name="MAESTRO",
        description="Expressive solo piano performances aligned to MIDI, from the International Piano-e-Competition.",
        url="https://magenta.tensorflow.org/datasets/maestro",
        license_notes="CC BY-NC-SA 4.0. Non-commercial use only unless a separate license is obtained.",
    ),
    DatasetEntry(
        name="POP909",
        description="909 pop songs with aligned melody, chords, and accompaniment arrangement structure.",
        url="https://github.com/music-x-lab/POP909-Dataset",
        license_notes="Research use; underlying songs are copyrighted commercial works — do not redistribute audio, "
        "and confirm terms before any commercial use of derived models.",
    ),
    DatasetEntry(
        name="GiantMIDI-Piano",
        description="Large-scale transcribed classical piano MIDI (~10k pieces) for symbolic modelling at scale.",
        url="https://github.com/bytedance/GiantMIDI-Piano",
        license_notes="Transcriptions of largely public-domain classical works; verify each source recording's rights "
        "before use, as transcription source audio may carry its own license.",
    ),
    DatasetEntry(
        name="Lakh MIDI Dataset",
        description="~176k MIDI files matched to Million Song Dataset entries; broad genre coverage.",
        url="https://colinraffel.com/projects/lmd/",
        license_notes="Mixed provenance and heavy duplication. Only usable with careful deduplication, and only for "
        "entries whose licensing has been individually checked — do not train on this in bulk without review.",
    ),
]


def load_manifest(path: Union[str, Path]) -> List[DatasetEntry]:
    """Load a local JSON manifest file (a list of dataset-entry objects).

    This never fetches data over the network — it only parses a manifest
    describing datasets the user has already made available locally.
    """
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest found at {manifest_path}. Datasets are never auto-downloaded; "
            "see docs/DATASETS.md for how to source and register one locally."
        )
    raw = json.loads(manifest_path.read_text())
    return [DatasetEntry(**entry) for entry in raw]


__all__ = ["DatasetEntry", "RECOMMENDED_DATASETS", "load_manifest"]
