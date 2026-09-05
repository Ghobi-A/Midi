"""Stage 2 backend: N-gram melody continuation over the deterministic scaffold.

This is the project's first *statistical* generation backend, and it is
deliberately narrow: the deterministic pipeline still parses the prompt and
builds the chords, bass, drums, structure, and instrumentation — only the
melody is replaced. The first bar of the deterministic melody is kept as a
seed motif, and the rest of the line is *continued* by sampling from a
trained :class:`~creative_audio_lab.models.ngram_model.NGramModel` over
symbolic melody tokens, then decoded back into notes through the existing
:class:`~creative_audio_lab.tokenization.SymbolicTokenizer`.

By default the model is trained on first use from a synthetic bootstrap
corpus (melodies the deterministic backend itself generates — see
:mod:`creative_audio_lab.models.ngram_training`), so the backend is always
runnable without downloads, but its "learning" is bounded by that corpus.
Pass ``model`` or ``model_path`` to use a model trained on your own licensed
token sequences instead (``model_path`` accepts the training artefact
written by ``scripts/train_ngram_melody.py`` or a bare model JSON). This is
not a neural text-to-MIDI model and does not claim to be.

The default model kind is the factorised
:class:`~creative_audio_lab.models.note_event_model.NoteEventModel`, whose
pitch sub-model conditions on previous *pitches*. The original flat
interleaved model is still selectable (``model_kind="flat"``) for
comparison; at its default order of 3 it predicted pitch from the previous
velocity and duration only.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..generators.arrangement import Arrangement, build_arrangement
from ..generators.chords import BEATS_PER_BAR
from ..generators.melody import INSTRUMENT_RANGES, lead_instrument
from ..music_theory import Note, note_name_to_pitch_class
from ..prompt_parser import parse_prompt
from ..tokenization import SymbolicTokenizer
from .artefact import load_any_model
from .base import GenerationBackend
from .ngram_model import NGramModel
from .ngram_training import (
    DEFAULT_MODEL_KIND,
    melody_token_stream,
    train_bootstrap_model,
    transpose_notes,
)
from .note_event_model import MelodyModel, NoteEventModel, split_note_stream

#: Beats of deterministic melody kept as the seed motif before continuation.
SEED_BEATS = float(BEATS_PER_BAR)

# Bootstrap models are cached per (kind, order) so repeated backend
# instantiations (e.g. one per Streamlit "Generate" click) don't retrain.
_BOOTSTRAP_MODELS: Dict[Tuple[str, Optional[int]], MelodyModel] = {}


def _bootstrap_model(kind: str, order: Optional[int]) -> MelodyModel:
    key = (kind, order)
    if key not in _BOOTSTRAP_MODELS:
        _BOOTSTRAP_MODELS[key] = train_bootstrap_model(order=order, kind=kind)
    return _BOOTSTRAP_MODELS[key]


def _stable_seed(*parts: object) -> int:
    """Deterministic cross-process sampling seed from the resolved controls."""
    digest = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _token_value(token: str) -> int:
    return int(token.rpartition("_")[2])


class NgramMelodyBackend(GenerationBackend):
    """Deterministic arrangement scaffolding with an N-gram-continued melody."""

    name = "ngram_melody"
    display_name = "Statistical"
    description = (
        "Stage 2 statistical baseline: the deterministic pipeline still builds "
        "chords, bass, drums, and structure, but the melody continues a seed "
        "motif by sampling from a trained n-gram model over symbolic tokens. "
        "The bundled demo model is bootstrap-trained on synthetic melodies from "
        "the deterministic backend itself — a machinery demo, not evidence of "
        "musical generalisation."
    )
    requires = ()

    def __init__(
        self,
        model: Optional[MelodyModel] = None,
        model_path: Optional[Union[str, Path]] = None,
        order: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        model_kind: str = DEFAULT_MODEL_KIND,
        sampling_seed: Optional[int] = None,
        seed_beats: float = SEED_BEATS,
    ) -> None:
        self._model = model
        self._model_path = Path(model_path) if model_path is not None else None
        self._order = order
        self._model_kind = model_kind
        self._temperature = temperature
        self._top_k = top_k
        self._sampling_seed = sampling_seed
        self._seed_beats = seed_beats
        self._artefact_metadata: Optional[dict] = None

    def _resolve_model(self) -> MelodyModel:
        if self._model is None:
            if self._model_path is not None:
                self._model, self._artefact_metadata = load_any_model(self._model_path)
            else:
                self._model = _bootstrap_model(self._model_kind, self._order)
        return self._model

    @property
    def model(self) -> MelodyModel:
        """The resolved melody model (loads or bootstrap-trains on first access)."""
        return self._resolve_model()

    @property
    def artefact_metadata(self) -> Optional[dict]:
        """Training metadata of the loaded artefact, or ``None`` for bootstrap/bare models."""
        self._resolve_model()
        return self._artefact_metadata

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
        """Build the deterministic arrangement, then swap in the n-gram melody."""
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
        base = build_arrangement(controls)
        base_melody = base.parts.get("melody", [])
        if not base_melody:
            return base

        melody = self._generate_melody(base, base_melody)
        parts = dict(base.parts)
        parts["melody"] = melody
        return Arrangement(
            controls=base.controls,
            chord_events=base.chord_events,
            parts=parts,
            programs=base.programs,
            bpm=base.bpm,
            bars=base.bars,
            ticks_per_beat=base.ticks_per_beat,
        )

    # ------------------------------------------------------------------
    # Melody continuation
    # ------------------------------------------------------------------

    def _generate_melody(self, base: Arrangement, base_melody: List[Note]) -> List[Note]:
        controls = base.controls
        tokenizer = SymbolicTokenizer()
        model = self._resolve_model()
        root_pc = note_name_to_pitch_class(controls.key)
        low_pitch, high_pitch = INSTRUMENT_RANGES.get(
            lead_instrument(controls), INSTRUMENT_RANGES["default"]
        )

        seed_notes = [note for note in base_melody if note.start < self._seed_beats]
        if not seed_notes:
            seed_notes = base_melody[:1]

        # Sampling context is the seed motif transposed to the C root the
        # corpus was normalized to (see ngram_training).
        context = melody_token_stream(transpose_notes(seed_notes, -root_pc), tokenizer)
        seed = self._sampling_seed
        if seed is None:
            seed = _stable_seed(controls.prompt, controls.key, controls.mode, controls.bpm,
                                controls.style, controls.density, "ngram-melody")
        rng = random.Random(seed)

        steps_per_beat = tokenizer.config.positions_per_beat
        total_steps = round(base.bars * BEATS_PER_BAR * steps_per_beat)
        cursor = round(max(note.end() for note in seed_notes) * steps_per_beat)
        events = self._sample_events(model, context, rng, cursor, total_steps,
                                     low_pitch - root_pc, high_pitch - root_pc)

        # Rebuild a full REMI-style token stream — seed motif in the target
        # key, then the sampled continuation with BAR/POSITION anchors derived
        # from the running cursor — and decode it through the shared tokenizer.
        strings = tokenizer.encode_to_strings(seed_notes)
        steps_per_bar = tokenizer.config.steps_per_bar
        current_bar = max(round(note.start * steps_per_beat) for note in seed_notes) // steps_per_bar
        for step, pitch, velocity_bin, duration_steps in events:
            bar, position = divmod(step, steps_per_bar)
            if bar != current_bar:
                strings.append(f"BAR_{bar}")
                current_bar = bar
            strings.append(f"POSITION_{position}")
            strings.append(f"NOTE_ON_{pitch + root_pc}")
            strings.append(f"VELOCITY_{velocity_bin}")
            strings.append(f"DURATION_{duration_steps}")
        return list(tokenizer.decode_from_strings(strings).notes)

    def _sample_events(
        self,
        model: MelodyModel,
        context: List[str],
        rng: random.Random,
        cursor: int,
        total_steps: int,
        low: int,
        high: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Sample (step, pitch, velocity_bin, duration_steps) events until the bars are full."""
        if isinstance(model, NoteEventModel):
            return self._sample_factorised(model, context, rng, cursor, total_steps, low, high)
        return self._sample_flat(model, context, rng, cursor, total_steps, low, high)

    def _sample_factorised(
        self,
        model: NoteEventModel,
        context: List[str],
        rng: random.Random,
        cursor: int,
        total_steps: int,
        low: int,
        high: int,
    ) -> List[Tuple[int, int, int, int]]:
        pitches, velocities, durations = split_note_stream(context)

        def in_range(token: str) -> bool:
            return low <= _token_value(token) <= high

        def positive(token: str) -> bool:
            return _token_value(token) >= 1

        def sample(allowed_pitch):
            return model.sample_note(
                pitches, velocities, durations, rng=rng,
                temperature=self._temperature, top_k=self._top_k,
                allowed_pitch=allowed_pitch, allowed_duration=positive,
            )

        events: List[Tuple[int, int, int, int]] = []
        while cursor < total_steps:
            # Prefer pitches inside the lead instrument's range; if no
            # backoff level has one, fall back to any observed pitch.
            triple = sample(in_range) or sample(None)
            if triple is None:
                break  # untrained/degenerate model: keep whatever we have
            pitch_token, velocity_token, duration_token = triple
            duration = min(_token_value(duration_token), total_steps - cursor)
            events.append((cursor, _token_value(pitch_token), _token_value(velocity_token), duration))
            pitches.append(pitch_token)
            velocities.append(velocity_token)
            durations.append(f"DURATION_{duration}")
            cursor += duration
        return events

    def _sample_flat(
        self,
        model: NGramModel,
        context: List[str],
        rng: random.Random,
        cursor: int,
        total_steps: int,
        low: int,
        high: int,
    ) -> List[Tuple[int, int, int, int]]:
        def sample(ctx: List[str], allowed) -> Optional[str]:
            return model.sample_next(
                ctx, temperature=self._temperature, top_k=self._top_k,
                allowed=allowed, rng=rng,
            )

        def in_range(token: str) -> bool:
            return token.startswith("NOTE_ON_") and low <= _token_value(token) <= high

        def is_note_on(token: str) -> bool:
            return token.startswith("NOTE_ON_")

        def is_velocity(token: str) -> bool:
            return token.startswith("VELOCITY_")

        def is_duration(token: str) -> bool:
            return token.startswith("DURATION_") and _token_value(token) >= 1

        events: List[Tuple[int, int, int, int]] = []
        while cursor < total_steps:
            # Prefer pitches inside the lead instrument's range; if no
            # backoff level has one, fall back to any observed pitch.
            pitch_token = sample(context, in_range) or sample(context, is_note_on)
            if pitch_token is None:
                break  # untrained/degenerate model: keep whatever we have
            velocity_token = sample(context + [pitch_token], is_velocity)
            if velocity_token is None:
                break
            duration_token = sample(context + [pitch_token, velocity_token], is_duration)
            if duration_token is None:
                break

            duration = min(_token_value(duration_token), total_steps - cursor)
            events.append((cursor, _token_value(pitch_token), _token_value(velocity_token), duration))
            context.extend([pitch_token, velocity_token, f"DURATION_{duration}"])
            cursor += duration
        return events


__all__ = ["NgramMelodyBackend", "SEED_BEATS"]
