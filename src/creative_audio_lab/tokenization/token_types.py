"""Token vocabulary for the internal symbolic tokenizer.

A token is a ``(type, value)`` pair with a canonical string form of
``TYPE_value`` (e.g. ``NOTE_ON_60``, ``POSITION_8``, ``TIME_SIGNATURE_4/4``),
mirroring the convention used by MidiTok's REMI tokenizer so sequences stay
readable and easy to port to an external tokenizer later.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence, Union


class TokenType(str, Enum):
    """The token kinds emitted by :class:`~creative_audio_lab.tokenization.SymbolicTokenizer`."""

    BAR = "BAR"
    POSITION = "POSITION"
    PROGRAM = "PROGRAM"
    NOTE_ON = "NOTE_ON"
    DURATION = "DURATION"
    VELOCITY = "VELOCITY"
    TEMPO = "TEMPO"
    TIME_SIGNATURE = "TIME_SIGNATURE"


@dataclass(frozen=True)
class Token:
    """One symbolic token: a :class:`TokenType` plus its value.

    Values are ``int`` for every type except :attr:`TokenType.TIME_SIGNATURE`,
    which uses a string like ``"4/4"``.
    """

    type: TokenType
    value: Union[int, str]

    def to_string(self) -> str:
        """Render the canonical ``TYPE_value`` string form (e.g. ``"NOTE_ON_60"``)."""
        return f"{self.type.value}_{self.value}"


def token_from_string(text: str) -> Token:
    """Parse a ``TYPE_value`` string produced by :meth:`Token.to_string` back into a :class:`Token`.

    Raises
    ------
    ValueError
        If ``text`` does not match any known token type.
    """
    head, sep, raw_value = text.rpartition("_")
    if not sep:
        raise ValueError(f"Not a valid token string: {text!r}")
    try:
        token_type = TokenType(head)
    except ValueError:
        raise ValueError(f"Unknown token type in {text!r}") from None
    value: Union[int, str]
    try:
        value = int(raw_value)
    except ValueError:
        value = raw_value
    return Token(type=token_type, value=value)


def tokens_to_strings(tokens: Sequence[Token]) -> List[str]:
    """Convert a token sequence to its canonical string forms."""
    return [token.to_string() for token in tokens]


def strings_to_tokens(strings: Sequence[str]) -> List[Token]:
    """Parse a sequence of ``TYPE_value`` strings back into :class:`Token` objects."""
    return [token_from_string(text) for text in strings]


__all__ = [
    "TokenType",
    "Token",
    "token_from_string",
    "tokens_to_strings",
    "strings_to_tokens",
]
