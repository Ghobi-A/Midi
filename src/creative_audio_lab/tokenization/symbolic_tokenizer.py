"""Dependency-free REMI-style tokenizer over canonical :class:`Note` events.

The encoding follows the REMI convention (bar/position anchoring rather than
time-shift deltas): an optional metadata header (``TEMPO``,
``TIME_SIGNATURE``, ``PROGRAM``) followed, per note, by

    ``BAR_<index>`` (when the bar changes) → ``POSITION_<step>`` →
    ``NOTE_ON_<pitch>`` → ``VELOCITY_<bin>`` → ``DURATION_<steps>``

Timing is quantized to ``positions_per_beat`` steps per beat (default 4, a
16th-note grid) and velocity to ``velocity_bins`` bins (default 32, i.e.
bin width 4). Notes that already sit on the grid — which everything the
deterministic generators emit does — round-trip with exact timing;
velocities only round-trip exactly when they're bin-aligned (or at the
lossless ``velocity_bins=128``), and the deterministic melody generator's
randomized velocities are not. :func:`measure_round_trip` quantifies
exactly this kind of loss for any note set and config.

Bar tokens come in two modes (``TokenizerConfig.relative_bars``):

- absolute (default): ``BAR_<index>`` carries the absolute bar number.
  Convenient for inspection, but the vocabulary grows with piece length —
  a 300-bar corpus piece emits ``BAR_299``, a token a model trained on
  shorter pieces has never seen.
- relative: one ``BAR_NONE`` bar-advance token per bar boundary crossed
  (the REMI convention), so the bar vocabulary is closed regardless of
  piece length. This is the mode any training corpus must use — see
  docs/STAGE3_PLAN.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..music_theory import Note
from .token_types import Token, TokenType, strings_to_tokens, tokens_to_strings

DEFAULT_VELOCITY = 96

#: The value carried by a bar-advance token (``BAR_NONE``) in relative mode.
BAR_ADVANCE_VALUE = "NONE"


@dataclass(frozen=True)
class TokenizerConfig:
    """Quantization settings for :class:`SymbolicTokenizer`.

    Attributes
    ----------
    beats_per_bar:
        Bar length in beats (4.0 = the 4/4 the generators assume).
    positions_per_beat:
        Grid resolution: steps per beat for POSITION and DURATION tokens.
    velocity_bins:
        Number of velocity bins over the 0-127 MIDI range. 128 is lossless;
        the default 32 (bin width 4) round-trips velocities that are
        multiples of 4.
    max_duration_beats:
        Durations longer than this are truncated so the DURATION vocabulary
        stays bounded.
    relative_bars:
        When ``True``, bar changes are encoded as one ``BAR_NONE`` advance
        token per bar crossed instead of an absolute ``BAR_<index>``, keeping
        the bar vocabulary closed for training. Decoding accepts both forms
        regardless of this setting.
    """

    beats_per_bar: float = 4.0
    positions_per_beat: int = 4
    velocity_bins: int = 32
    max_duration_beats: float = 8.0
    relative_bars: bool = False

    @property
    def steps_per_bar(self) -> int:
        return round(self.beats_per_bar * self.positions_per_beat)

    @property
    def velocity_bin_width(self) -> int:
        return max(128 // self.velocity_bins, 1)


@dataclass(frozen=True)
class DecodedSequence:
    """The result of decoding a token sequence back into note events."""

    notes: Tuple[Note, ...]
    tempo: Optional[int] = None
    program: Optional[int] = None
    time_signature: Optional[str] = None


class SymbolicTokenizer:
    """Encode :class:`Note` events to symbolic tokens and decode them back.

    Works with the core install only — no MidiTok, symusic, or torch
    required. For tokenizing arbitrary MIDI *files* with the reference REMI
    implementation, see
    :class:`~creative_audio_lab.tokenization.miditok_adapter.MidiTokRemiAdapter`.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None) -> None:
        self.config = config or TokenizerConfig()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        notes: Sequence[Note],
        *,
        tempo: Optional[float] = None,
        program: Optional[int] = None,
        time_signature: Optional[str] = None,
    ) -> List[Token]:
        """Encode note events (plus optional metadata) into a flat token list.

        Notes are sorted by ``(start, pitch)`` and quantized to the config's
        position grid; simultaneous notes (chords) share one BAR/POSITION
        anchor.
        """
        cfg = self.config
        tokens: List[Token] = []
        if tempo is not None:
            tokens.append(Token(TokenType.TEMPO, round(tempo)))
        if time_signature is not None:
            tokens.append(Token(TokenType.TIME_SIGNATURE, time_signature))
        if program is not None:
            tokens.append(Token(TokenType.PROGRAM, int(program)))

        max_duration_steps = max(round(cfg.max_duration_beats * cfg.positions_per_beat), 1)
        current_bar: Optional[int] = None
        current_position: Optional[int] = None

        for note in sorted(notes, key=lambda n: (n.start, n.pitch)):
            total_steps = round(note.start * cfg.positions_per_beat)
            bar, position = divmod(total_steps, cfg.steps_per_bar)
            if bar != current_bar:
                if cfg.relative_bars:
                    previous = -1 if current_bar is None else current_bar
                    tokens.extend(Token(TokenType.BAR, BAR_ADVANCE_VALUE) for _ in range(bar - previous))
                else:
                    tokens.append(Token(TokenType.BAR, bar))
                current_bar = bar
                current_position = None
            if position != current_position:
                tokens.append(Token(TokenType.POSITION, position))
                current_position = position

            duration_steps = min(max(round(note.duration * cfg.positions_per_beat), 1), max_duration_steps)
            velocity_bin = min(max(note.velocity, 0), 127) // cfg.velocity_bin_width
            tokens.append(Token(TokenType.NOTE_ON, note.pitch))
            tokens.append(Token(TokenType.VELOCITY, velocity_bin))
            tokens.append(Token(TokenType.DURATION, duration_steps))

        return tokens

    def encode_to_strings(
        self,
        notes: Sequence[Note],
        *,
        tempo: Optional[float] = None,
        program: Optional[int] = None,
        time_signature: Optional[str] = None,
    ) -> List[str]:
        """Like :meth:`encode`, but returns canonical ``TYPE_value`` strings."""
        return tokens_to_strings(self.encode(notes, tempo=tempo, program=program, time_signature=time_signature))

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, tokens: Sequence[Token]) -> DecodedSequence:
        """Decode a token sequence back into note events where possible.

        A note is emitted for each ``NOTE_ON`` once its ``DURATION`` token
        arrives; a missing ``VELOCITY`` falls back to
        :data:`DEFAULT_VELOCITY`, and a ``NOTE_ON`` never followed by a
        ``DURATION`` is dropped rather than raising.
        """
        cfg = self.config
        notes: List[Note] = []
        tempo: Optional[int] = None
        program: Optional[int] = None
        time_signature: Optional[str] = None

        current_bar = 0
        bar_token_seen = False
        current_position = 0
        pending_pitch: Optional[int] = None
        pending_velocity: Optional[int] = None

        for token in tokens:
            if token.type is TokenType.TEMPO:
                tempo = int(token.value)
            elif token.type is TokenType.TIME_SIGNATURE:
                time_signature = str(token.value)
            elif token.type is TokenType.PROGRAM:
                program = int(token.value)
            elif token.type is TokenType.BAR:
                if token.value == BAR_ADVANCE_VALUE:
                    # Bar-advance: the first one lands on bar 0.
                    current_bar = current_bar + 1 if bar_token_seen else 0
                else:
                    current_bar = int(token.value)
                bar_token_seen = True
                current_position = 0
            elif token.type is TokenType.POSITION:
                current_position = int(token.value)
            elif token.type is TokenType.NOTE_ON:
                pending_pitch = int(token.value)
                pending_velocity = None
            elif token.type is TokenType.VELOCITY:
                pending_velocity = int(token.value) * cfg.velocity_bin_width
            elif token.type is TokenType.DURATION and pending_pitch is not None:
                start_steps = current_bar * cfg.steps_per_bar + current_position
                notes.append(
                    Note(
                        start=start_steps / cfg.positions_per_beat,
                        pitch=pending_pitch,
                        duration=int(token.value) / cfg.positions_per_beat,
                        velocity=pending_velocity if pending_velocity is not None else DEFAULT_VELOCITY,
                    )
                )
                pending_pitch = None
                pending_velocity = None

        return DecodedSequence(
            notes=tuple(notes),
            tempo=tempo,
            program=program,
            time_signature=time_signature,
        )

    def decode_from_strings(self, strings: Sequence[str]) -> DecodedSequence:
        """Like :meth:`decode`, but accepts canonical ``TYPE_value`` strings."""
        return self.decode(strings_to_tokens(strings))


@dataclass(frozen=True)
class RoundTripReport:
    """Measured encode→decode loss for a set of notes under one config.

    The tokenizer never drops notes, so loss shows up as quantization:
    onsets snapped to the position grid, durations snapped/truncated, and
    velocities binned. docs/STAGE3_PLAN.md makes an acceptable-loss
    threshold on the actual corpus a go/no-go gate before training — this
    is the measurement behind that gate.
    """

    total_notes: int
    exact_notes: int
    onset_moved: int
    duration_changed: int
    velocity_changed: int

    @property
    def exact_fraction(self) -> float:
        """Fraction of notes that survived the round trip unchanged."""
        if self.total_notes == 0:
            return 1.0
        return self.exact_notes / self.total_notes


def measure_round_trip(notes: Sequence[Note], tokenizer: Optional[SymbolicTokenizer] = None) -> RoundTripReport:
    """Encode then decode ``notes`` and count what quantization changed."""
    tok = tokenizer or SymbolicTokenizer()
    decoded = tok.decode(tok.encode(notes)).notes
    original = sorted(notes, key=lambda n: (n.start, n.pitch))
    roundtripped = sorted(decoded, key=lambda n: (n.start, n.pitch))

    exact = onset_moved = duration_changed = velocity_changed = 0
    for before, after in zip(original, roundtripped):
        moved = not math.isclose(before.start, after.start, rel_tol=1e-9, abs_tol=1e-9)
        stretched = not math.isclose(before.duration, after.duration, rel_tol=1e-9, abs_tol=1e-9)
        rebinned = before.velocity != after.velocity
        onset_moved += moved
        duration_changed += stretched
        velocity_changed += rebinned
        exact += not (moved or stretched or rebinned)

    return RoundTripReport(
        total_notes=len(original),
        exact_notes=exact,
        onset_moved=onset_moved,
        duration_changed=duration_changed,
        velocity_changed=velocity_changed,
    )


__all__ = [
    "TokenizerConfig",
    "DecodedSequence",
    "SymbolicTokenizer",
    "RoundTripReport",
    "measure_round_trip",
    "DEFAULT_VELOCITY",
    "BAR_ADVANCE_VALUE",
]
