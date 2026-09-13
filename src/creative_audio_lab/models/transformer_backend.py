"""Transformer melody continuation over the unchanged deterministic arrangement."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Union

import torch

from ..generators.arrangement import Arrangement, build_arrangement
from ..generators.chords import BEATS_PER_BAR
from ..generators.melody import INSTRUMENT_RANGES, lead_instrument
from ..music_theory import Note, note_name_to_pitch_class
from ..prompt_parser import parse_prompt
from .base import GenerationBackend
from .neural_events import EOS, VALUE_OFFSET, NeuralNoteEvent
from .transformer_training import load_transformer_artefact


def _stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4], "big")


def _sample(logits, generator, temperature, top_k, top_p, allowed=None):
    scores = logits.float() / temperature
    if allowed is not None:
        valid = torch.zeros_like(scores, dtype=torch.bool)
        valid[allowed] = True
        scores = scores.masked_fill(~valid, float("-inf"))
    if top_k is not None:
        cutoff = torch.topk(scores, min(top_k, scores.numel())).values[-1]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))
    if top_p is not None:
        ordered, indices = torch.sort(scores, descending=True)
        cumulative = torch.softmax(ordered, -1).cumsum(-1)
        remove = cumulative - torch.softmax(ordered, -1) > top_p
        ordered[remove] = float("-inf")
        scores = torch.full_like(scores, float("-inf")).scatter(0, indices, ordered)
    return int(torch.multinomial(torch.softmax(scores, -1), 1, generator=generator))


class TransformerMelodyBackend(GenerationBackend):
    """Replace only melody notes using a safely loaded trained neural artefact."""

    name = "transformer_melody"
    display_name = "Neural"
    description = "Small causal Transformer trained on symbolic melody note events."
    requires = ("torch",)

    def __init__(self, model_path: Union[str, Path], *, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None,
                 sampling_seed: Optional[int] = None, seed_beats: float = 4.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        self.model, self.codec, self.metadata = load_transformer_artefact(model_path)
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.sampling_seed, self.seed_beats = sampling_seed, seed_beats

    def generate(self, prompt: str, *, key=None, mode=None, bpm=None, bars=None, style=None,
                 energy=None, density=None, instruments=None) -> Arrangement:
        controls = parse_prompt(prompt, key=key, mode=mode, bpm=bpm, bars=bars, style=style,
                                energy=energy, density=density, instruments=instruments)
        base = build_arrangement(controls)
        original = base.parts.get("melody", [])
        if not original:
            return base
        seed_notes = [note for note in original if note.start < self.seed_beats] or original[:1]
        root = note_name_to_pitch_class(controls.key)
        normalised = [Note(n.start, n.pitch - root, n.duration, n.velocity) for n in seed_notes]
        context = self.codec.encode_notes(normalised)
        context = context[:-1]  # BOS + real seed; EOS is only a target.
        seed = self.sampling_seed if self.sampling_seed is not None else _stable_seed(
            controls.prompt, controls.key, controls.mode, controls.bpm, "transformer-melody")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        cursor = max(note.end() for note in seed_notes)
        end = base.bars * BEATS_PER_BAR
        low, high = INSTRUMENT_RANGES.get(lead_instrument(controls), INSTRUMENT_RANGES["default"])
        generated = list(seed_notes)
        self.model.eval()
        with torch.no_grad():
            while cursor < end:
                visible = context[-self.model.config.context_length:]
                inputs = {name: torch.tensor([[getattr(event, name) for event in visible]])
                          for name in ("pitch", "duration", "velocity")}
                outputs = self.model(**inputs)
                allowed_pitch = [EOS] + [pitch + VALUE_OFFSET for pitch in range(
                    max(0, low - root), min(127, high - root) + 1)]
                pitch = _sample(outputs[0][0, -1], generator, self.temperature, self.top_k,
                                self.top_p, allowed_pitch)
                if pitch == EOS:
                    break
                duration = _sample(outputs[1][0, -1], generator, self.temperature, self.top_k,
                                   self.top_p, range(VALUE_OFFSET, self.codec.duration_size))
                velocity = _sample(outputs[2][0, -1], generator, self.temperature, self.top_k,
                                   self.top_p, range(VALUE_OFFSET, self.codec.velocity_size))
                event = NeuralNoteEvent(pitch, duration, velocity)
                decoded = self.codec.decode_event(event, cursor)
                duration_beats = min(decoded.duration, end - cursor)
                generated.append(Note(cursor, decoded.pitch + root, duration_beats, decoded.velocity))
                cursor += duration_beats
                context.append(event)
        parts = dict(base.parts)
        parts["melody"] = generated
        return Arrangement(base.controls, base.chord_events, parts, base.programs, base.bpm,
                           base.bars, base.ticks_per_beat)
