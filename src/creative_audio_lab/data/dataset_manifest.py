"""Schema and recommended entries for future ML training datasets.

This module never downloads anything — see docs/DATASETS.md for licensing
and usage guidance on each recommended dataset, and
``examples/dataset_manifest.example.json`` for a manifest template using
placeholder paths.

Provenance/rights validation lives in
:mod:`creative_audio_lab.data.provenance` and
:mod:`creative_audio_lab.data.license_policy`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class DatasetEntry:
    """Describes one dataset a future training pipeline could ingest.

    Attributes
    ----------
    name:
        Human-readable dataset name.
    source_url:
        Where the dataset officially comes from. Required for a clean
        provenance record — "found on a forum" is not a source.
    license:
        License identifier or short statement (e.g. ``"CC BY-NC-SA 4.0"``,
        ``"research-only"``, ``"unknown"``).
    local_path:
        Where the user has placed the files locally, if at all. Nothing is
        ever downloaded automatically.
    commercial_use_allowed:
        ``True``/``False`` when the license makes it clear; ``None`` when
        unknown or unverified (validation warns on ``None``).
    attribution_required:
        Whether the license requires crediting the source; ``None`` = unknown.
    description:
        What the dataset contains and why it's useful.
    notes:
        Free-form rights/usage caveats (deduplication needs, per-file
        license checks, ...).
    tags:
        Free-form labels. Tags like ``"fan-archive"`` or
        ``"unclear-rights"`` mark a dataset as private reference/evaluation
        only — provenance validation flags them as unusable for training.
    """

    name: str
    source_url: str = ""
    license: str = ""
    local_path: Optional[str] = None
    commercial_use_allowed: Optional[bool] = None
    attribution_required: Optional[bool] = None
    description: str = ""
    notes: str = ""
    tags: Tuple[str, ...] = ()


_ENTRY_FIELDS = {f.name for f in fields(DatasetEntry)}


def entry_from_dict(raw: Dict[str, Any]) -> DatasetEntry:
    """Build a :class:`DatasetEntry` from a plain dict (e.g. one JSON manifest item).

    Raises
    ------
    ValueError
        If the dict is missing ``name`` or contains unknown keys, so typos
        in manifest files fail loudly instead of silently dropping rights
        metadata.
    """
    unknown = set(raw) - _ENTRY_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown dataset-manifest keys {sorted(unknown)}; expected a subset of {sorted(_ENTRY_FIELDS)}"
        )
    if "name" not in raw:
        raise ValueError("Every dataset-manifest entry needs a 'name'")
    data = dict(raw)
    if "tags" in data and data["tags"] is not None:
        data["tags"] = tuple(data["tags"])
    return DatasetEntry(**data)


RECOMMENDED_DATASETS: List[DatasetEntry] = [
    DatasetEntry(
        name="MAESTRO",
        source_url="https://magenta.tensorflow.org/datasets/maestro",
        license="CC BY-NC-SA 4.0",
        commercial_use_allowed=False,
        attribution_required=True,
        description="Expressive solo piano performances aligned to MIDI, from the International Piano-e-Competition.",
        notes="Non-commercial use only unless a separate license is obtained.",
        tags=("piano", "performance"),
    ),
    DatasetEntry(
        name="POP909",
        source_url="https://github.com/music-x-lab/POP909-Dataset",
        license="research-only",
        commercial_use_allowed=False,
        description="909 pop songs with aligned melody, chords, and accompaniment arrangement structure.",
        notes="Underlying songs are copyrighted commercial works — do not redistribute audio, "
        "and confirm terms before any commercial use of derived models.",
        tags=("pop", "melody-chord-split"),
    ),
    DatasetEntry(
        name="GiantMIDI-Piano",
        source_url="https://github.com/bytedance/GiantMIDI-Piano",
        license="unknown",
        description="Large-scale transcribed classical piano MIDI (~10k pieces) for symbolic modelling at scale.",
        notes="Transcriptions of largely public-domain classical works; verify each source recording's rights "
        "before use, as transcription source audio may carry its own license.",
        tags=("piano", "classical", "transcription"),
    ),
    DatasetEntry(
        name="Lakh MIDI Dataset",
        source_url="https://colinraffel.com/projects/lmd/",
        license="unknown",
        description="~176k MIDI files matched to Million Song Dataset entries; broad genre coverage.",
        notes="Mixed provenance and heavy duplication. Only usable with careful deduplication, and only for "
        "entries whose licensing has been individually checked — do not train on this in bulk without review.",
        tags=("multi-genre", "unclear-rights"),
    ),
    DatasetEntry(
        name="MidiCaps",
        source_url="https://huggingface.co/datasets/amaai-lab/MidiCaps",
        license="CC BY-SA 4.0 (captions/metadata)",
        description="~168k MIDI files paired with rich text captions — the natural fit for "
        "text-conditioned symbolic generation.",
        notes="Captions are CC BY-SA, but the MIDI comes from the Lakh MIDI Dataset, so Lakh's "
        "mixed-provenance caveats apply to the note content itself.",
        tags=("text-to-midi", "captions"),
    ),
    DatasetEntry(
        name="MetaScore (public subset)",
        source_url="https://github.com/salu133445/metascore",
        license="per-score (CC-licensed subset)",
        description="MuseScore-derived scores with rich metadata (genre, tags, user descriptions); "
        "the public subset is restricted to permissively licensed scores.",
        notes="Rights vary per score — filter to the CC-licensed public subset and record each score's license.",
        tags=("scores", "metadata"),
    ),
    DatasetEntry(
        name="Slakh2100",
        source_url="http://www.slakh.com/",
        license="CC BY 4.0",
        commercial_use_allowed=True,
        attribution_required=True,
        description="2100 multi-track MIDI arrangements (from Lakh) with professionally synthesized audio stems.",
        notes="Useful for multi-instrument arrangement modelling; the redistributed set is CC BY 4.0.",
        tags=("multi-track", "stems"),
    ),
    DatasetEntry(
        name="ComMU",
        source_url="https://github.com/POZAlabs/ComMU-code",
        license="CC BY-NC 4.0",
        commercial_use_allowed=False,
        attribution_required=True,
        description="~11k short MIDI samples written by professional composers, each labelled with 12 metadata "
        "fields (BPM, key, chord progression, instrument, track role, ...).",
        notes="Purpose-built for controllable/conditioned generation — very close to this project's "
        "GenerationControls framing. Non-commercial.",
        tags=("conditioned-generation", "metadata"),
    ),
    DatasetEntry(
        name="EMOPIA",
        source_url="https://annahung31.github.io/EMOPIA/",
        license="CC BY-NC-SA 4.0",
        commercial_use_allowed=False,
        attribution_required=True,
        description="~1k pop-piano clips with per-clip emotion (valence/arousal quadrant) labels.",
        notes="Clips are transcriptions of copyrighted pop piano covers — non-commercial, and underlying "
        "musical works remain copyrighted.",
        tags=("piano", "emotion-labels"),
    ),
    DatasetEntry(
        name="Aria-MIDI",
        source_url="https://github.com/EleutherAI/aria",
        license="verify at source",
        description="~1M piano MIDI transcriptions from public performance recordings; large-scale "
        "symbolic pretraining material.",
        notes="Transcription does not clear rights on the underlying recordings — verify the release "
        "license and treat commercial use as unresolved.",
        tags=("piano", "transcription", "large-scale"),
    ),
    DatasetEntry(
        name="music21 corpus",
        source_url="https://www.music21.org/music21docs/about/referenceCorpus.html",
        license="per-work (mostly public domain)",
        description="Curated corpus bundled with music21: Bach chorales, folk songs, and other largely "
        "public-domain scores in symbolic form.",
        notes="music21 itself is BSD-licensed; individual corpus works are mostly public domain but a few "
        "carry their own restrictions — check per work.",
        tags=("classical", "public-domain"),
    ),
]


def load_manifest(path: Union[str, Path]) -> List[DatasetEntry]:
    """Load a local JSON manifest file (a list of dataset-entry objects).

    This never fetches data over the network — it only parses a manifest
    describing datasets the user has already made available locally. Run the
    result through :func:`creative_audio_lab.data.provenance.validate_manifest`
    before using it for anything training-related.
    """
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest found at {manifest_path}. Datasets are never auto-downloaded; "
            "see docs/DATASETS.md for how to source and register one locally."
        )
    raw = json.loads(manifest_path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Manifest {manifest_path} must contain a JSON list of dataset entries")
    return [entry_from_dict(entry) for entry in raw]


__all__ = ["DatasetEntry", "RECOMMENDED_DATASETS", "entry_from_dict", "load_manifest"]
