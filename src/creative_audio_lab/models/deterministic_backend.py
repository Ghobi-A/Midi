"""The default backend: the deterministic parser + rule-based generators.

This is a thin wrapper — all real behaviour still lives in
:mod:`creative_audio_lab.prompt_parser` and
:mod:`creative_audio_lab.generators`. Wrapping it behind
:class:`~creative_audio_lab.models.base.GenerationBackend` is what lets a
future learned backend swap in without touching the app.
"""

from __future__ import annotations

from typing import List, Optional

from ..generators.arrangement import Arrangement, build_arrangement
from ..prompt_parser import parse_prompt
from .base import GenerationBackend


class DeterministicBackend(GenerationBackend):
    """Rule-based prompt-to-MIDI generation (the current default)."""

    name = "deterministic"
    display_name = "Deterministic"
    description = (
        "Keyword-based prompt parsing plus rule-based chord/melody/bass/drum "
        "generators. Fully explainable, reproducible per prompt, and requires "
        "no trained model."
    )
    requires = ()

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
        """Parse ``prompt`` into controls (honouring overrides) and build the arrangement."""
        controls = parse_prompt(
            prompt,
            key=key,
            mode=mode,
            bpm=bpm,
            bars=bars,
            style=style,
            energy=energy,
            density=density,
            instruments=instruments,
        )
        return build_arrangement(controls)


__all__ = ["DeterministicBackend"]
