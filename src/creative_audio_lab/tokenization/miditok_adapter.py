"""Optional MidiTok/symusic adapter for tokenizing MIDI files with REMI.

MidiTok and symusic are **not** core dependencies. Install them with::

    pip install "creative-audio-lab[symbolic]"

Everything in this module degrades gracefully when they are absent:
:func:`is_miditok_available` reports availability, and constructing
:class:`MidiTokRemiAdapter` without them raises
:class:`MidiTokNotAvailableError` with installation guidance instead of a
bare ``ImportError`` deep inside a third-party import.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_INSTALL_HINT = (
    "MidiTok-based tokenization requires the optional 'miditok' and 'symusic' "
    "packages, which are not installed. Install them with:\n\n"
    '    pip install "creative-audio-lab[symbolic]"\n\n'
    "The core prompt-to-MIDI app does not need them — the dependency-free "
    "creative_audio_lab.tokenization.SymbolicTokenizer covers note-event "
    "tokenization without any extras."
)


class MidiTokNotAvailableError(ImportError):
    """Raised when MidiTok/symusic functionality is requested but not installed."""


def is_miditok_available() -> bool:
    """Return ``True`` if both ``miditok`` and ``symusic`` are importable."""
    return (
        importlib.util.find_spec("miditok") is not None
        and importlib.util.find_spec("symusic") is not None
    )


def require_miditok() -> None:
    """Raise :class:`MidiTokNotAvailableError` with install guidance if MidiTok is missing."""
    if not is_miditok_available():
        raise MidiTokNotAvailableError(_INSTALL_HINT)


class MidiTokRemiAdapter:
    """Thin wrapper around MidiTok's REMI tokenizer for whole MIDI files.

    This exists so future dataset preprocessing can use the reference REMI
    implementation (via MidiTok + symusic) while the rest of the codebase
    keeps depending only on the internal
    :class:`~creative_audio_lab.tokenization.SymbolicTokenizer`.
    """

    def __init__(self, tokenizer_params: Optional[Dict[str, Any]] = None) -> None:
        """Create a REMI tokenizer.

        Parameters
        ----------
        tokenizer_params:
            Optional keyword arguments forwarded to
            ``miditok.TokenizerConfig`` (beat resolution, velocity bins, ...).

        Raises
        ------
        MidiTokNotAvailableError
            If ``miditok``/``symusic`` are not installed.
        """
        require_miditok()
        import miditok

        config = miditok.TokenizerConfig(**(tokenizer_params or {}))
        self._tokenizer = miditok.REMI(config)

    @property
    def vocab_size(self) -> int:
        """Size of the underlying REMI vocabulary."""
        return len(self._tokenizer)

    def tokenize_file(self, path: Union[str, Path]) -> List[str]:
        """Tokenize a MIDI file on disk into a flat list of REMI token strings.

        Multi-track files are flattened track by track, in file order.
        """
        sequences = self._tokenizer(Path(path))
        if not isinstance(sequences, list):
            sequences = [sequences]
        tokens: List[str] = []
        for sequence in sequences:
            tokens.extend(sequence.tokens)
        return tokens


__all__ = [
    "MidiTokNotAvailableError",
    "MidiTokRemiAdapter",
    "is_miditok_available",
    "require_miditok",
]
