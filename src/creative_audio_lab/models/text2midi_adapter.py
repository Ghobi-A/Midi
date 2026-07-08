"""Placeholder adapter for a Text2midi-style text-conditioned symbolic model.

`Text2midi <https://github.com/AMAAI-Lab/Text2midi>`_ (AMAAI Lab) generates
MIDI token sequences from free-text captions with an LLM-encoder +
transformer-decoder architecture, trained on caption-paired MIDI such as
MidiCaps. It is a *candidate future integration*, not a dependency — nothing
here imports it, and no checkpoint ships with this project.

The adapter exists so the integration surface is designed now: the app
already selects backends through
:class:`~creative_audio_lab.models.base.GenerationBackend`, so a working
implementation of :meth:`Text2MidiAdapter.generate` is all that's missing.
"""

from __future__ import annotations

from typing import List, Optional

from ..generators.arrangement import Arrangement
from .base import GenerationBackend

_NOT_IMPLEMENTED_GUIDANCE = (
    "Text2MidiAdapter is a scaffolding placeholder — no Text2midi model is "
    "integrated or shipped yet.\n\n"
    "To integrate one later:\n"
    "  1. Install the ML extras: pip install \"creative-audio-lab[ml]\" (torch, "
    "transformers) and obtain the Text2midi weights per the upstream repo's license.\n"
    "  2. Encode the prompt and sample a symbolic token sequence from the model.\n"
    "  3. Decode tokens into Note events — creative_audio_lab.tokenization."
    "SymbolicTokenizer.decode already handles the REMI-style vocabulary, and "
    "MidiTokRemiAdapter covers MidiTok-formatted output.\n"
    "  4. Assemble the notes into an Arrangement (parts + General MIDI programs) "
    "so export and evaluation work unchanged.\n\n"
    "Until then, use the default deterministic backend: "
    "creative_audio_lab.models.get_backend(\"deterministic\")."
)


class Text2MidiAdapter(GenerationBackend):
    """Future adapter for caption-conditioned symbolic generation (not yet implemented)."""

    name = "text2midi"
    display_name = "Text2midi (future adapter)"
    description = (
        "Planned adapter for a Text2midi-style caption-conditioned transformer "
        "that generates MIDI token sequences from free text. Scaffolded only — "
        "no model is trained, bundled, or downloaded."
    )
    requires = ("ml extra (torch, transformers)", "Text2midi weights (external)")

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


__all__ = ["Text2MidiAdapter"]
