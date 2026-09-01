"""Factorised note-event n-gram model for melodies.

Why this exists
---------------
The first Stage 2 baseline fitted a single :class:`NGramModel` over the
interleaved ``NOTE_ON → VELOCITY → DURATION`` token stream. With the
default order of 3, the context available when predicting a ``NOTE_ON``
token is ``(VELOCITY_prev, DURATION_prev)`` — the previous *pitch* has
already fallen out of the window, so that model never learned
pitch-to-pitch (melodic) transitions at all. It learned pitch given
rhythm/dynamics, velocity given pitch, and duration given pitch and
velocity.

This module models a note as a composite event whose three attributes are
predicted from *their own* history:

- ``P(pitch_i | pitch_{i-1}, pitch_{i-2}, ...)`` — a true melodic n-gram,
- ``P(duration_i | duration_{i-1}, ...)`` — a rhythm n-gram,
- ``P(velocity_i | velocity_{i-1}, ...)`` — a dynamics n-gram.

The per-note log-likelihood is the sum of the three, which makes the model
directly comparable with the flat interleaved model on held-out data (see
:func:`evaluate_flat_model`): both report **bits per note**, and both
report the pitch component separately, which is the number that actually
measures melodic predictability. Cross-attribute dependencies (e.g.
duration given pitch) are deliberately dropped in this first correction;
they are a documented follow-up, not an accident.

Everything here is dependency-free and JSON-serialisable, like the
underlying :class:`NGramModel`.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..tokenization.token_types import TokenType
from .ngram_model import EOS, NGramModel, TokenFilter

MODEL_FORMAT = "creative-audio-lab.note-event-ngram/1"

PITCH_PREFIX = TokenType.NOTE_ON.value + "_"
VELOCITY_PREFIX = TokenType.VELOCITY.value + "_"
DURATION_PREFIX = TokenType.DURATION.value + "_"

DEFAULT_PITCH_ORDER = 3
DEFAULT_DURATION_ORDER = 3
DEFAULT_VELOCITY_ORDER = 2

#: The three attribute streams of a note-event sequence, in this order.
STREAM_NAMES = ("pitch", "velocity", "duration")

NoteTriple = Tuple[str, str, str]


def split_note_stream(stream: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """Split an interleaved ``NOTE_ON/VELOCITY/DURATION`` stream into three streams.

    The input is the format produced by
    :func:`creative_audio_lab.models.ngram_training.melody_token_stream`.
    Any incomplete trailing note is dropped; a stream that is not made of
    well-formed triples raises ``ValueError``.
    """
    pitches: List[str] = []
    velocities: List[str] = []
    durations: List[str] = []
    for i in range(0, len(stream) - len(stream) % 3, 3):
        pitch, velocity, duration = stream[i], stream[i + 1], stream[i + 2]
        if not (
            pitch.startswith(PITCH_PREFIX)
            and velocity.startswith(VELOCITY_PREFIX)
            and duration.startswith(DURATION_PREFIX)
        ):
            raise ValueError(
                f"Expected NOTE_ON/VELOCITY/DURATION triple at index {i}, got "
                f"{stream[i:i + 3]!r}"
            )
        pitches.append(pitch)
        velocities.append(velocity)
        durations.append(duration)
    return pitches, velocities, durations


def _bits(nats: float) -> float:
    return nats / math.log(2)


def _summarise(
    note_count: int, pitch_nats: float, velocity_nats: float, duration_nats: float,
    pitch_oov: float,
) -> Dict[str, float]:
    if note_count == 0:
        nan = float("nan")
        return {
            "notes": 0, "bits_per_note": nan, "pitch_bits_per_note": nan,
            "velocity_bits_per_note": nan, "duration_bits_per_note": nan,
            "perplexity_per_note": nan, "pitch_perplexity": nan, "pitch_oov_rate": nan,
        }
    pitch_bits = _bits(pitch_nats) / note_count
    velocity_bits = _bits(velocity_nats) / note_count
    duration_bits = _bits(duration_nats) / note_count
    total = pitch_bits + velocity_bits + duration_bits
    return {
        "notes": note_count,
        "bits_per_note": total,
        "pitch_bits_per_note": pitch_bits,
        "velocity_bits_per_note": velocity_bits,
        "duration_bits_per_note": duration_bits,
        "perplexity_per_note": 2.0 ** total,
        "pitch_perplexity": 2.0 ** pitch_bits,
        "pitch_oov_rate": pitch_oov,
    }


class NoteEventModel:
    """Three independent n-gram sub-models over pitch, velocity, and duration streams."""

    def __init__(
        self,
        pitch_order: int = DEFAULT_PITCH_ORDER,
        duration_order: int = DEFAULT_DURATION_ORDER,
        velocity_order: int = DEFAULT_VELOCITY_ORDER,
    ) -> None:
        self.pitch = NGramModel(order=pitch_order)
        self.velocity = NGramModel(order=velocity_order)
        self.duration = NGramModel(order=duration_order)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @property
    def orders(self) -> Dict[str, int]:
        return {
            "pitch": self.pitch.order,
            "velocity": self.velocity.order,
            "duration": self.duration.order,
        }

    @property
    def total_notes(self) -> int:
        """Number of training notes (pitch targets, EOS excluded)."""
        return max(self.pitch.total_tokens - self._pieces, 0)

    _pieces = 0

    def fit(self, sequences: Iterable[Sequence[str]]) -> "NoteEventModel":
        """Fit all three sub-models on interleaved note-token ``sequences``."""
        for stream in sequences:
            pitches, velocities, durations = split_note_stream(stream)
            if not pitches:
                continue
            self._pieces += 1
            self.pitch.fit([pitches])
            self.velocity.fit([velocities])
            self.duration.fit([durations])
        return self

    @property
    def vocabulary(self) -> List[str]:
        """Union of the three sub-vocabularies (EOS excluded)."""
        tokens = set(self.pitch.vocabulary) | set(self.velocity.vocabulary) | set(self.duration.vocabulary)
        tokens.discard(EOS)
        return sorted(tokens)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_note(
        self,
        pitch_context: Sequence[str],
        velocity_context: Sequence[str],
        duration_context: Sequence[str],
        *,
        rng: random.Random,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        allowed_pitch: Optional[TokenFilter] = None,
        allowed_duration: Optional[TokenFilter] = None,
    ) -> Optional[NoteTriple]:
        """Sample one ``(pitch, velocity, duration)`` token triple, or ``None``.

        Each attribute is drawn from its own sub-model given that
        attribute's history. Boundary tokens are never returned: the
        filters below always exclude :data:`EOS`.
        """
        def not_eos(prefix: str, extra: Optional[TokenFilter]) -> TokenFilter:
            return lambda token: token.startswith(prefix) and (extra is None or extra(token))

        pitch = self.pitch.sample_next(
            pitch_context, temperature=temperature, top_k=top_k,
            allowed=not_eos(PITCH_PREFIX, allowed_pitch), rng=rng,
        )
        if pitch is None:
            return None
        velocity = self.velocity.sample_next(
            velocity_context, temperature=temperature, top_k=top_k,
            allowed=not_eos(VELOCITY_PREFIX, None), rng=rng,
        )
        if velocity is None:
            return None
        duration = self.duration.sample_next(
            duration_context, temperature=temperature, top_k=top_k,
            allowed=not_eos(DURATION_PREFIX, allowed_duration), rng=rng,
        )
        if duration is None:
            return None
        return pitch, velocity, duration

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def pitch_prob(self, pitch_token: str, pitch_context: Sequence[str]) -> float:
        """Smoothed ``P(next pitch | previous pitches)`` — the melodic-dependency probe."""
        return self.pitch.prob(pitch_token, pitch_context)

    def evaluate(self, sequences: Sequence[Sequence[str]]) -> Dict[str, float]:
        """Held-out bits per note, with the pitch/velocity/duration breakdown.

        Boundary (EOS) predictions are excluded so the numbers are comparable
        with :func:`evaluate_flat_model`; each reported quantity is the
        average over the *notes* of ``sequences``.
        """
        notes = 0
        pitch_nats = velocity_nats = duration_nats = 0.0
        pitch_streams: List[List[str]] = []
        for stream in sequences:
            pitches, velocities, durations = split_note_stream(stream)
            if not pitches:
                continue
            notes += len(pitches)
            pitch_streams.append(pitches)
            pitch_nats -= sum(self.pitch.token_log_probs(pitches)[: len(pitches)])
            velocity_nats -= sum(self.velocity.token_log_probs(velocities)[: len(velocities)])
            duration_nats -= sum(self.duration.token_log_probs(durations)[: len(durations)])
        oov = self.pitch.oov_rate(pitch_streams) if pitch_streams else float("nan")
        return _summarise(notes, pitch_nats, velocity_nats, duration_nats, oov)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": MODEL_FORMAT,
            "pieces": self._pieces,
            "pitch": self.pitch.to_dict(),
            "velocity": self.velocity.to_dict(),
            "duration": self.duration.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NoteEventModel":
        if data.get("format") != MODEL_FORMAT:
            raise ValueError(f"Unsupported model format: {data.get('format')!r}")
        model = cls()
        model.pitch = NGramModel.from_dict(data["pitch"])
        model.velocity = NGramModel.from_dict(data["velocity"])
        model.duration = NGramModel.from_dict(data["duration"])
        model._pieces = int(data.get("pieces", 0))
        return model

    def save_json(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "NoteEventModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def evaluate_flat_model(model: NGramModel, sequences: Sequence[Sequence[str]]) -> Dict[str, float]:
    """Score a flat interleaved :class:`NGramModel` in the same per-note units.

    The flat model's token log-probabilities are grouped by the token type
    they predict, so its ``pitch_bits_per_note`` is exactly the cost of the
    ``NOTE_ON`` tokens given whatever context the flat model had — which is
    what exposes the order-3 context bug.
    """
    notes = 0
    pitch_nats = velocity_nats = duration_nats = 0.0
    pitch_streams: List[List[str]] = []
    for stream in sequences:
        pitches, _, _ = split_note_stream(stream)
        if not pitches:
            continue
        notes += len(pitches)
        pitch_streams.append(pitches)
        log_probs = model.token_log_probs(stream)
        for token, log_prob in zip(stream, log_probs):
            if token.startswith(PITCH_PREFIX):
                pitch_nats -= log_prob
            elif token.startswith(VELOCITY_PREFIX):
                velocity_nats -= log_prob
            elif token.startswith(DURATION_PREFIX):
                duration_nats -= log_prob
    oov = model.oov_rate(pitch_streams) if pitch_streams else float("nan")
    return _summarise(notes, pitch_nats, velocity_nats, duration_nats, oov)


MelodyModel = Union[NGramModel, NoteEventModel]


def evaluate_melody_model(model: MelodyModel, sequences: Sequence[Sequence[str]]) -> Dict[str, float]:
    """Dispatch to :meth:`NoteEventModel.evaluate` or :func:`evaluate_flat_model`."""
    if isinstance(model, NoteEventModel):
        return model.evaluate(sequences)
    return evaluate_flat_model(model, sequences)


__all__ = [
    "MODEL_FORMAT",
    "DEFAULT_PITCH_ORDER",
    "DEFAULT_DURATION_ORDER",
    "DEFAULT_VELOCITY_ORDER",
    "STREAM_NAMES",
    "MelodyModel",
    "NoteEventModel",
    "split_note_stream",
    "evaluate_flat_model",
    "evaluate_melody_model",
]
