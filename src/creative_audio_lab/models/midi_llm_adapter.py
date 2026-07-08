"""Placeholder adapter for a MIDI-LLM style backend (LLM over MIDI tokens).

The "MIDI-LLM" family of approaches adapts a pretrained text LLM to emit
symbolic music tokens (an expanded vocabulary over a base model, or plain
text formats like ABC/MIDI-text). It is a *candidate future integration*,
not a dependency — nothing here imports any external model repo, and no
checkpoint ships with this project.

Like :class:`~creative_audio_lab.models.text2midi_adapter.Text2MidiAdapter`,
this adapter exists to pin down the integration surface early: implement
:meth:`MidiLlmAdapter.generate` and the rest of the app works unchanged.
"""

from __future__ import annotations

from typing import List, Optional

from ..generators.arrangement import Arrangement
from .base import GenerationBackend

_NOT_IMPLEMENTED_GUIDANCE = (
    "MidiLlmAdapter is a scaffolding placeholder — no MIDI-capable LLM is "
    "integrated or shipped yet.\n\n"
    "To integrate one later:\n"
    "  1. Install the ML extras: pip install \"creative-audio-lab[ml]\" (torch, "
    "transformers) and pick a MIDI-token LLM (e.g. a model with a vocabulary "
    "expanded for symbolic music tokens).\n"
    "  2. Map the prompt (and any GenerationControls overrides) into the model's "
    "conditioning format.\n"
    "  3. Sample a token sequence and decode it into Note events via "
    "creative_audio_lab.tokenization (SymbolicTokenizer for the internal REMI-style "
    "vocabulary, MidiTokRemiAdapter for MidiTok formats).\n"
    "  4. Assemble the notes into an Arrangement (parts + General MIDI programs) "
    "so export and evaluation work unchanged.\n\n"
    "Until then, use the default deterministic backend: "
    "creative_audio_lab.models.get_backend(\"deterministic\")."
)


class MidiLlmAdapter(GenerationBackend):
    """Future adapter for LLM-based symbolic generation (not yet implemented)."""

    name = "midi_llm"
    display_name = "MIDI-LLM (future adapter)"
    description = (
        "Planned adapter for a text LLM adapted to emit symbolic music tokens. "
        "Scaffolded only — no model is trained, bundled, or downloaded."
    )
    requires = ("ml extra (torch, transformers)", "MIDI-token LLM weights (external)")

    @property
    def is_available(self) -> bool:
        """Always ``False``: this adapter is scaffolding, not an integration."""
        return False

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
        """Not implemented — raises with step-by-step integration guidance."""
        raise NotImplementedError(_NOT_IMPLEMENTED_GUIDANCE)


__all__ = ["MidiLlmAdapter"]
