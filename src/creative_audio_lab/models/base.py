"""The common interface every generation backend implements.

A *backend* turns a text prompt (plus optional user overrides) into an
:class:`~creative_audio_lab.generators.arrangement.Arrangement`. The
deterministic rule-based pipeline is the first and default implementation;
future learned models (Text2midi-style text-conditioned generation, or a
MIDI LLM) plug in behind the same interface so the app, export, and
evaluation layers never need to know which one produced the notes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

from ..generators.arrangement import Arrangement


@dataclass(frozen=True)
class BackendInfo:
    """Static metadata describing a backend, for UIs and registries."""

    name: str
    display_name: str
    description: str
    available: bool
    requires: Tuple[str, ...]


class GenerationBackend(ABC):
    """Abstract base class for prompt-to-arrangement generation backends.

    Subclasses set the class attributes below and implement
    :meth:`generate`. Backends whose runtime requirements are missing (e.g.
    a model checkpoint or an optional extra) should report
    ``is_available == False`` and raise a helpful error from
    :meth:`generate` instead of failing at import time.
    """

    #: Stable registry key (e.g. ``"deterministic"``).
    name: ClassVar[str] = "abstract"
    #: Human-readable name for UI display.
    display_name: ClassVar[str] = "Abstract backend"
    #: One-or-two-sentence description of how the backend generates music.
    description: ClassVar[str] = ""
    #: pip extras / external projects the backend needs (informational).
    requires: ClassVar[Tuple[str, ...]] = ()

    @property
    def is_available(self) -> bool:
        """Whether the backend can actually generate right now."""
        return True

    def info(self) -> BackendInfo:
        """Return static metadata plus current availability."""
        return BackendInfo(
            name=self.name,
            display_name=self.display_name,
            description=self.description,
            available=self.is_available,
            requires=self.requires,
        )

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        key: Optional[str] = None,
        mode: Optional[str] = None,
        bpm: Optional[int] = None,
        bars: Optional[int] = None,
        style: Optional[str] = None,
        energy: Optional[str] = None,
        density: Optional[str] = None,
        instruments: Optional[List[str]] = None,
    ) -> Arrangement:
        """Generate a full arrangement from ``prompt``.

        Keyword overrides mirror the Streamlit sidebar controls: any that
        are ``None`` are inferred from the prompt by the backend.
        """


__all__ = ["BackendInfo", "GenerationBackend"]
