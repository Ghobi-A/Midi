"""Creative Audio Lab: a prompt-conditioned symbolic music generator.

The main feature, Prompt-to-MIDI Generator, turns a natural-language prompt
into a DAW-ready MIDI arrangement using deterministic, rule-based
generators (see :mod:`creative_audio_lab.generators`). MIDI Motif Lab
(:mod:`creative_audio_lab.motif_detection`, :mod:`.motif_variation`) adds
motif analysis and variation on top of any generated or imported melody.
"""

from .generators.arrangement import Arrangement, build_arrangement
from .music_theory import Note
from .prompt_parser import GenerationControls, parse_prompt

__all__ = [
    "Arrangement",
    "build_arrangement",
    "Note",
    "GenerationControls",
    "parse_prompt",
]
