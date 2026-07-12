"""Stage 2: an n-gram melody-continuation baseline backend.

This is roadmap step 7 (see docs/ML_ROADMAP.md) and the first gate of
docs/STAGE3_PLAN.md: a statistical model over the internal tokenizer's
vocabulary, "trained" on melodies sampled from the deterministic generator
itself. Its purpose is *not* musical quality — synthetic-on-synthetic
statistics cannot exceed their source. It exists to exercise the full
tokenize → train → sample → decode → evaluate loop at near-zero compute,
and to produce the held-out NLL number any future neural model must beat.

Everything here is core-install only: no torch, no external corpus, and
nothing downloaded. Training happens lazily in-process on first use (a few
dozen deterministic arrangements), so constructing the backend — which the
app does on every rerun to list backends — stays cheap.
"""

from __future__ import annotations

import math
import random
import zlib
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from ..generators.arrangement import Arrangement, build_arrangement
from ..music_theory import Note
from ..prompt_parser import GenerationControls, parse_prompt
from ..tokenization import BAR_ADVANCE_VALUE, SymbolicTokenizer, TokenizerConfig
from .base import GenerationBackend

#: Prompts used to bootstrap the n-gram statistics from the deterministic
#: generator. Deliberately disjoint from
#: :data:`creative_audio_lab.evaluation.prompts.HELD_OUT_PROMPTS` — a test
#: enforces this — so the held-out comparison stays honest.
BOOTSTRAP_PROMPTS: Tuple[str, ...] = (
    "dark cinematic theme in C minor, 120 BPM",
    "epic orchestral theme, brass and strings, 100 BPM",
    "sad cinematic piano theme in E minor",
    "hopeful orchestral theme in F major, 16 bars",
    "trap melody in G minor, 140 BPM, bells",
    "aggressive drill melody, 144 BPM",
    "ambient piano in A dorian, 75 BPM, sparse",
    "mysterious ambient loop, strings, 80 BPM",
    "jrpg fantasy theme in Bb major, flute, 126 BPM",
    "happy rpg town theme, 118 BPM",
    "romantic ballad in D major, piano, 88 BPM",
    "nostalgic piano ballad in Ab major, 92 BPM",
    "tense cinematic theme in B minor, 138 BPM, dense",
    "dark trap loop in Eb minor, busy",
    "calm ambient theme in G dorian, minimal",
    "energetic orchestral battle music, 145 BPM",
    "sad ballad in F# minor, strings and piano",
    "mysterious jrpg theme in D minor, 110 BPM",
)

_BOS = "<s>"
_EOS = "</s>"

#: Token config for all Stage 2 training material: relative bar-advance
#: tokens keep the vocabulary closed (the mandatory pre-training fix from
#: docs/STAGE3_PLAN.md, section 3).
TRAINING_TOKENIZER_CONFIG = TokenizerConfig(relative_bars=True)

_BAR_TOKEN = f"BAR_{BAR_ADVANCE_VALUE}"


class TokenNgramModel:
    """A plain order-``n`` n-gram model over token strings with backoff.

    Sampling draws from the longest context with observed counts (no
    smoothing, so only tokens actually seen after that context can be
    sampled). Log-likelihood uses Laplace smoothing at the backoff level so
    unseen continuations get finite, comparable probabilities. This is a
    baseline, not a calibrated language model — see the module docstring.
    """

    def __init__(self, order: int = 4) -> None:
        if order < 2:
            raise ValueError("order must be at least 2 (context of at least one token)")
        self.order = order
        self._counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self._vocab: set = set()

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def _padded(self, sequence: Sequence[str]) -> List[str]:
        return [_BOS] * (self.order - 1) + list(sequence) + [_EOS]

    def fit(self, sequences: Sequence[Sequence[str]]) -> None:
        """Accumulate n-gram counts (all orders up to ``self.order``) from token sequences."""
        for sequence in sequences:
            padded = self._padded(sequence)
            self._vocab.update(padded)
            for i in range(self.order - 1, len(padded)):
                token = padded[i]
                for context_len in range(self.order):
                    context = tuple(padded[i - context_len : i])
                    self._counts[context][token] += 1

    def _backoff_counts(self, context: Sequence[str]) -> Tuple[Tuple[str, ...], Counter]:
        """Return the longest suffix of ``context`` with observed counts (down to the unigram)."""
        context = tuple(context)[-(self.order - 1) :]
        for start in range(len(context) + 1):
            suffix = context[start:]
            if self._counts.get(suffix):
                return suffix, self._counts[suffix]
        return (), self._counts[()]

    def sample_next(self, context: Sequence[str], rng: random.Random) -> str:
        """Sample the next token from the longest matching context."""
        if not self._vocab:
            raise RuntimeError("TokenNgramModel.sample_next called before fit()")
        _, counts = self._backoff_counts(context)
        tokens = list(counts.keys())
        weights = list(counts.values())
        return rng.choices(tokens, weights=weights, k=1)[0]

    def log_probability(self, token: str, context: Sequence[str]) -> float:
        """Natural-log probability of ``token`` after ``context`` (Laplace-smoothed backoff)."""
        _, counts = self._backoff_counts(context)
        total = sum(counts.values())
        vocab = max(self.vocab_size, 1)
        return math.log((counts.get(token, 0) + 1) / (total + vocab))

    def avg_negative_log_likelihood(self, sequences: Sequence[Sequence[str]]) -> float:
        """Average per-token negative log-likelihood (nats) over ``sequences``.

        This is the Stage 2 baseline number a future trained model has to
        beat on the same held-out token sequences.
        """
        total_nll = 0.0
        total_tokens = 0
        for sequence in sequences:
            padded = self._padded(sequence)
            for i in range(self.order - 1, len(padded)):
                context = tuple(padded[i - (self.order - 1) : i])
                total_nll -= self.log_probability(padded[i], context)
                total_tokens += 1
        if total_tokens == 0:
            raise ValueError("No tokens to score")
        return total_nll / total_tokens


def _rng_seed(controls: GenerationControls) -> int:
    # crc32 rather than hash(): stable across processes, so fixed-prompt
    # comparisons are reproducible run to run.
    payload = f"{controls.prompt}|{controls.key}|{controls.mode}|{controls.bpm}|{controls.style}|ngram"
    return zlib.crc32(payload.encode("utf-8"))


class NgramMelodyBackend(GenerationBackend):
    """Melody continuation from n-gram statistics over synthetic bootstrap data.

    ``generate`` builds the deterministic arrangement first, then replaces
    the melody: the deterministic melody's opening bar seeds the n-gram
    model, which samples the continuation token by token until the bar
    budget is spent. Chords, bass, and drums stay deterministic — the same
    scoping the Stage 3a transformer pilot will use — so
    ``harmonic_fit_score`` remains meaningful. If sampling yields fewer
    than :data:`MIN_SAMPLED_NOTES` decodable notes, the deterministic
    melody is kept as-is rather than shipping a degenerate part.
    """

    name = "ngram"
    display_name = "N-gram melody baseline (Stage 2)"
    description = (
        "Continues the deterministic melody's opening bar by sampling from "
        "n-gram statistics over tokenized deterministic output. A pipeline "
        "and evaluation baseline, not a quality upgrade: it cannot know more "
        "than the rule-based generator it bootstraps from."
    )
    requires = ()

    MIN_SAMPLED_NOTES = 4
    MAX_SAMPLED_TOKENS = 2048

    def __init__(
        self,
        order: int = 4,
        bootstrap_prompts: Sequence[str] = BOOTSTRAP_PROMPTS,
        seed: int = 0,
    ) -> None:
        self._order = order
        self._bootstrap_prompts = tuple(bootstrap_prompts)
        self._seed = seed
        self._tokenizer = SymbolicTokenizer(TRAINING_TOKENIZER_CONFIG)
        self._model: Optional[TokenNgramModel] = None

    # ------------------------------------------------------------------
    # Training (lazy, in-process, synthetic bootstrap only)
    # ------------------------------------------------------------------

    def _melody_tokens(self, arrangement: Arrangement) -> List[str]:
        return self._tokenizer.encode_to_strings(arrangement.parts.get("melody", []))

    def bootstrap_sequences(self) -> List[List[str]]:
        """Tokenized melodies of every bootstrap prompt's deterministic arrangement."""
        sequences = []
        for prompt in self._bootstrap_prompts:
            arrangement = build_arrangement(parse_prompt(prompt))
            tokens = self._melody_tokens(arrangement)
            if tokens:
                sequences.append(tokens)
        return sequences

    def _ensure_trained(self) -> TokenNgramModel:
        if self._model is None:
            model = TokenNgramModel(order=self._order)
            model.fit(self.bootstrap_sequences())
            self._model = model
        return self._model

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _split_seed_bar(self, tokens: List[str]) -> List[str]:
        """Return the tokens of the melody's first bar (through the second bar-advance)."""
        bar_count = 0
        for index, token in enumerate(tokens):
            if token == _BAR_TOKEN:
                bar_count += 1
                if bar_count == 2:
                    return tokens[:index]
        return tokens

    def _sample_continuation(self, seed_tokens: List[str], total_bars: int, rng: random.Random) -> List[str]:
        model = self._ensure_trained()
        sampled = list(seed_tokens)
        bars_started = sampled.count(_BAR_TOKEN)
        while len(sampled) < self.MAX_SAMPLED_TOKENS:
            token = model.sample_next(sampled, rng)
            if token == _EOS:
                break
            if token == _BAR_TOKEN:
                bars_started += 1
                if bars_started > total_bars:
                    break
            sampled.append(token)
        return sampled

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
        """Build the deterministic arrangement, then swap in the sampled melody."""
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
        arrangement = build_arrangement(controls)

        deterministic_tokens = self._melody_tokens(arrangement)
        if deterministic_tokens:
            rng = random.Random(self._seed ^ _rng_seed(controls))
            seed_tokens = self._split_seed_bar(deterministic_tokens)
            sampled = self._sample_continuation(seed_tokens, arrangement.bars, rng)
            decoded = self._tokenizer.decode_from_strings(sampled).notes
            total_beats = arrangement.bars * self._tokenizer.config.beats_per_bar
            # Sampled POSITION tokens can land out of order within a bar, so
            # re-sort into chronological order before handing downstream.
            melody = sorted(
                (note for note in decoded if note.start < total_beats),
                key=lambda note: (note.start, note.pitch),
            )
            if len(melody) >= self.MIN_SAMPLED_NOTES:
                arrangement.parts["melody"] = self._clip_to_length(melody, total_beats)
        return arrangement

    @staticmethod
    def _clip_to_length(notes: List[Note], total_beats: float) -> List[Note]:
        """Trim any note that sustains past the end of the arrangement."""
        return [
            note
            if note.start + note.duration <= total_beats
            else Note(start=note.start, pitch=note.pitch, duration=total_beats - note.start, velocity=note.velocity)
            for note in notes
        ]


__all__ = [
    "BOOTSTRAP_PROMPTS",
    "TRAINING_TOKENIZER_CONFIG",
    "TokenNgramModel",
    "NgramMelodyBackend",
]
