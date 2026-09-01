"""Melody-model artefacts: a trained model plus the metadata that makes it a claim.

A bare model JSON (counts and order) says nothing about *what* it was
trained on, how the held-out split was drawn, or which tokenizer settings
produced its vocabulary. Once numbers from a model are reported anywhere,
those facts have to travel with the file. The artefact envelope written
here wraps any melody model with:

- ``model_kind`` and orders,
- corpus provenance: manifest path, dataset names/licences, the provenance
  report, and a content hash over every training file,
- the composition-level split (which piece ids went to train/val/test) and
  the seed that produced it,
- tokenizer configuration,
- package version, git commit, and creation time,
- corpus statistics and held-out evaluation metrics.

:func:`load_any_model` also accepts bare :class:`NGramModel` /
:class:`NoteEventModel` JSON so older files keep working (with no metadata).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from . import ngram_model, note_event_model
from .ngram_model import NGramModel
from .note_event_model import MelodyModel, NoteEventModel

ARTEFACT_FORMAT = "creative-audio-lab.melody-model/1"

#: Metadata keys every artefact written by the training CLI carries.
REQUIRED_METADATA_KEYS = (
    "model_kind",
    "orders",
    "corpus",
    "split",
    "tokenizer",
    "seeds",
    "software",
    "created_at",
    "stats",
    "metrics",
)


def model_kind(model: MelodyModel) -> str:
    return "factorised" if isinstance(model, NoteEventModel) else "flat"


def model_orders(model: MelodyModel) -> Dict[str, int]:
    if isinstance(model, NoteEventModel):
        return model.orders
    return {"flat": model.order}


def package_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("creative-audio-lab")
    except Exception:  # pragma: no cover - not installed as a distribution
        return None


def git_commit(repo_root: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Best-effort ``git rev-parse HEAD``; ``None`` outside a checkout."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def software_metadata() -> Dict[str, Optional[str]]:
    return {"package_version": package_version(), "git_commit": git_commit()}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def build_artefact(model: MelodyModel, metadata: Dict[str, Any]) -> dict:
    """Wrap ``model`` and ``metadata`` in the artefact envelope (JSON-ready)."""
    meta = dict(_jsonable(metadata))
    meta.setdefault("model_kind", model_kind(model))
    meta.setdefault("orders", model_orders(model))
    meta.setdefault("software", software_metadata())
    meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    return {"format": ARTEFACT_FORMAT, "model": model.to_dict(), "metadata": meta}


def save_artefact(model: MelodyModel, metadata: Dict[str, Any], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_artefact(model, metadata), indent=1), encoding="utf-8")
    return path


def model_from_dict(data: dict) -> MelodyModel:
    """Rebuild whichever bare model ``data`` encodes."""
    fmt = data.get("format")
    if fmt == note_event_model.MODEL_FORMAT:
        return NoteEventModel.from_dict(data)
    if fmt in (ngram_model.MODEL_FORMAT, ngram_model.LEGACY_MODEL_FORMAT):
        return NGramModel.from_dict(data)
    raise ValueError(f"Unsupported model format: {fmt!r}")


def load_any_model(path: Union[str, Path]) -> Tuple[MelodyModel, Optional[dict]]:
    """Load an artefact envelope *or* a bare model JSON.

    Returns ``(model, metadata)``; ``metadata`` is ``None`` for bare files.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") == ARTEFACT_FORMAT:
        return model_from_dict(data["model"]), dict(data.get("metadata", {}))
    return model_from_dict(data), None


def load_artefact(path: Union[str, Path]) -> Tuple[MelodyModel, dict]:
    """Load an artefact envelope; raises ``ValueError`` for bare model files."""
    model, metadata = load_any_model(path)
    if metadata is None:
        raise ValueError(f"{path} is a bare model file, not a training artefact")
    return model, metadata


__all__ = [
    "ARTEFACT_FORMAT",
    "REQUIRED_METADATA_KEYS",
    "build_artefact",
    "save_artefact",
    "load_artefact",
    "load_any_model",
    "model_from_dict",
    "model_kind",
    "model_orders",
    "software_metadata",
    "git_commit",
    "package_version",
]
