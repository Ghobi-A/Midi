"""Generation backends (Stage 1.5 ML readiness + the Stage 2 n-gram baseline).

Defines the :class:`~creative_audio_lab.models.base.GenerationBackend`
interface, the default :class:`DeterministicBackend` wrapping the existing
parser + generators, the Stage 2 :class:`NgramMelodyBackend` (a small
statistical n-gram melody-continuation baseline, bootstrap-trained on
synthetic data — see :mod:`.ngram_training`), and placeholder adapters
marking where heavier learned models (Text2midi-style, MIDI-LLM-style) will
plug in later. Nothing here downloads data or requires ML dependencies.
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import BackendInfo, GenerationBackend
from .deterministic_backend import DeterministicBackend
from .midi_llm_adapter import MidiLlmAdapter
from .ngram_backend import NgramMelodyBackend
from .ngram_model import NGramModel
from .note_event_model import NoteEventModel
from .text2midi_adapter import Text2MidiAdapter

DEFAULT_BACKEND_NAME = DeterministicBackend.name

#: Registry of every known backend, keyed by its stable name.
BACKEND_REGISTRY: Dict[str, Type[GenerationBackend]] = {
    DeterministicBackend.name: DeterministicBackend,
    NgramMelodyBackend.name: NgramMelodyBackend,
    Text2MidiAdapter.name: Text2MidiAdapter,
    MidiLlmAdapter.name: MidiLlmAdapter,
}


def get_backend(name: str = DEFAULT_BACKEND_NAME, **kwargs) -> GenerationBackend:
    """Instantiate the backend registered under ``name``.

    Keyword arguments are forwarded to the backend constructor (e.g.
    ``get_backend("ngram_melody", model_path="model.json")``).

    Raises
    ------
    KeyError
        If ``name`` is not a registered backend, listing the valid names.
    """
    try:
        backend_cls = BACKEND_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown backend {name!r}; registered backends: {sorted(BACKEND_REGISTRY)}"
        ) from None
    return backend_cls(**kwargs)


def list_backends() -> List[BackendInfo]:
    """Return :class:`BackendInfo` metadata for every registered backend."""
    return [backend_cls().info() for backend_cls in BACKEND_REGISTRY.values()]


__all__ = [
    "BackendInfo",
    "GenerationBackend",
    "DeterministicBackend",
    "NgramMelodyBackend",
    "NGramModel",
    "NoteEventModel",
    "Text2MidiAdapter",
    "MidiLlmAdapter",
    "BACKEND_REGISTRY",
    "DEFAULT_BACKEND_NAME",
    "get_backend",
    "list_backends",
]
