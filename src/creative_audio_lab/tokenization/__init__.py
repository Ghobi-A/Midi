"""Symbolic tokenization layer (Stage 1.5 ML readiness).

Converts the package's canonical :class:`~creative_audio_lab.music_theory.Note`
events into flat symbolic token sequences (and back), the representation a
future text-conditioned symbolic model would train on.

Two paths are provided:

- :class:`SymbolicTokenizer` — a dependency-free, REMI-style internal
  tokenizer that works with only the core install.
- :class:`~creative_audio_lab.tokenization.miditok_adapter.MidiTokRemiAdapter`
  — an optional wrapper around `MidiTok <https://github.com/Natooz/MidiTok>`_
  (requires ``pip install "creative-audio-lab[symbolic]"``) for tokenizing
  MIDI files with the battle-tested REMI implementation.
"""

from .symbolic_tokenizer import (
    DecodedSequence,
    SymbolicTokenizer,
    TokenizerConfig,
    beats_per_bar_for,
)
from .token_types import Token, TokenType, strings_to_tokens, token_from_string, tokens_to_strings

__all__ = [
    "DecodedSequence",
    "SymbolicTokenizer",
    "TokenizerConfig",
    "beats_per_bar_for",
    "Token",
    "TokenType",
    "token_from_string",
    "tokens_to_strings",
    "strings_to_tokens",
]
