"""A small, dependency-free N-gram model over symbolic token strings.

This is the Stage 2 statistical baseline described in ``docs/ML_ROADMAP.md``
(item 7: "Markov / n-gram continuation baseline"). It counts token
transitions of every context length up to ``order - 1``, backs off to
shorter contexts when a context was never seen, and samples continuations
with temperature and optional top-k truncation. Sampling is deterministic
when given a seed, and models round-trip through plain JSON — no torch,
sklearn, or any other heavy dependency is involved (or allowed) here.

Besides sampling, the model assigns *smoothed* probabilities to tokens
(Witten-Bell interpolation, which needs no tuned hyper-parameters and
reserves probability mass for out-of-vocabulary tokens), so held-out
sequences can be scored with :meth:`NGramModel.cross_entropy` and
:meth:`NGramModel.perplexity`. Sequences are padded with ``BOS``/``EOS``
boundary tokens when ``boundaries=True`` (the default), so the model learns
how pieces start and end instead of treating every piece as one unbroken
stream.

The model is generic over hashable string tokens; nothing in this module
knows about music. The melody-specific corpus building and decoding live in
:mod:`creative_audio_lab.models.ngram_training`,
:mod:`creative_audio_lab.models.note_event_model`, and
:mod:`creative_audio_lab.models.ngram_backend`.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

#: Version tag written into saved models so future format changes can be detected.
MODEL_FORMAT = "creative-audio-lab.ngram/2"
#: Older format (no boundary tokens) that :meth:`NGramModel.from_dict` still accepts.
LEGACY_MODEL_FORMAT = "creative-audio-lab.ngram/1"

#: Sequence-boundary tokens. BOS only ever appears in contexts; EOS is predicted.
BOS = "<BOS>"
EOS = "<EOS>"

#: Predicate deciding whether a candidate next token may be sampled.
TokenFilter = Callable[[str], bool]


class NGramModel:
    """Count-based N-gram model with backoff sampling and Witten-Bell scoring.

    Parameters
    ----------
    order:
        Maximum n-gram order. ``order=3`` (the default) predicts the next
        token from up to the two preceding tokens.
    boundaries:
        When ``True`` every fitted or scored sequence is padded with
        ``order - 1`` :data:`BOS` tokens and one trailing :data:`EOS`.
    """

    def __init__(self, order: int = 3, boundaries: bool = True) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.boundaries = boundaries
        # counts[context_tuple] -> Counter of next tokens, for every context
        # length from 0 (unigram) to order - 1.
        self._counts: Dict[Tuple[str, ...], Counter] = {}
        self._total_tokens = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def pad(self, sequence: Sequence[str]) -> List[str]:
        """Return ``sequence`` with boundary tokens added (no-op if disabled)."""
        tokens = list(sequence)
        if not self.boundaries:
            return tokens
        return [BOS] * (self.order - 1) + tokens + [EOS]

    def fit(self, sequences: Iterable[Sequence[str]]) -> "NGramModel":
        """Accumulate n-gram counts from token sequences (additive across calls)."""
        for sequence in sequences:
            tokens = self.pad(sequence)
            start = self.order - 1 if self.boundaries else 0
            for i in range(start, len(tokens)):
                token = tokens[i]
                self._total_tokens += 1
                for ctx_len in range(min(i, self.order - 1) + 1):
                    context = tuple(tokens[i - ctx_len : i])
                    self._counts.setdefault(context, Counter())[token] += 1
        return self

    @property
    def total_tokens(self) -> int:
        """Number of training tokens seen so far (EOS tokens included when enabled)."""
        return self._total_tokens

    @property
    def vocabulary(self) -> List[str]:
        """Every token observed as a prediction target during fitting, sorted."""
        return sorted(self._counts.get((), Counter()))

    def context_counts(self, context: Sequence[str]) -> Dict[str, int]:
        """Raw next-token counts for an exact ``context`` (no backoff)."""
        return dict(self._counts.get(tuple(context), Counter()))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _candidates(
        self, context: Sequence[str], allowed: Optional[TokenFilter]
    ) -> List[Tuple[str, int]]:
        """Backoff lookup: longest known context suffix with allowed continuations."""
        tokens = list(context)
        for ctx_len in range(min(len(tokens), self.order - 1), -1, -1):
            suffix = tuple(tokens[len(tokens) - ctx_len :])
            counter = self._counts.get(suffix)
            if not counter:
                continue
            candidates = [
                (token, count)
                for token, count in counter.items()
                if token != BOS and (allowed is None or allowed(token))
            ]
            if candidates:
                return candidates
        return []

    def sample_next(
        self,
        context: Sequence[str],
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        allowed: Optional[TokenFilter] = None,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
    ) -> Optional[str]:
        """Sample the next token after ``context``, backing off through shorter contexts.

        ``temperature`` reshapes count weights as ``count ** (1 / temperature)``
        (1.0 = proportional to counts; lower is greedier). ``top_k`` keeps only
        the k highest-count candidates before weighting; ``top_k=1`` is argmax.
        ``allowed`` filters candidates (e.g. to enforce a token grammar).
        Returns ``None`` only when no observed token passes the filter at any
        backoff level. :data:`EOS` can be returned when boundaries are on;
        :data:`BOS` never is.
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0 (use top_k=1 for argmax)")
        candidates = self._candidates(context, allowed)
        if not candidates:
            return None
        # Sort by (count desc, token) so top-k truncation and tie-handling are
        # reproducible regardless of dict insertion order.
        candidates.sort(key=lambda item: (-item[1], item[0]))
        if top_k is not None:
            candidates = candidates[: max(top_k, 1)]
        chooser = rng if rng is not None else random.Random(seed)
        weights = [count ** (1.0 / temperature) for _, count in candidates]
        return chooser.choices([token for token, _ in candidates], weights=weights)[0]

    def sample_sequence(
        self,
        seed_context: Sequence[str] = (),
        *,
        length: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        allowed: Optional[TokenFilter] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Sample up to ``length`` tokens continuing ``seed_context`` (context not included).

        With boundaries enabled an empty ``seed_context`` starts from the
        BOS padding (so the first token follows the learned piece-start
        distribution) and sampling stops early when :data:`EOS` is drawn.
        """
        rng = random.Random(seed)
        context = list(seed_context)
        if self.boundaries and not context:
            context = [BOS] * (self.order - 1)
        generated: List[str] = []
        for _ in range(length):
            token = self.sample_next(
                context, temperature=temperature, top_k=top_k, allowed=allowed, rng=rng
            )
            if token is None or token == EOS:
                break
            generated.append(token)
            context.append(token)
        return generated

    # ------------------------------------------------------------------
    # Smoothed probabilities (Witten-Bell interpolation)
    # ------------------------------------------------------------------

    def prob(self, token: str, context: Sequence[str] = ()) -> float:
        """Smoothed ``P(token | context)`` — strictly positive for any token.

        Witten-Bell interpolation: for a context ``h`` seen ``c(h)`` times
        with ``T(h)`` distinct continuations,
        ``P(w|h) = (c(h,w) + T(h) * P(w|h')) / (c(h) + T(h))`` where ``h'``
        drops the oldest token. The recursion bottoms out at the unigram
        interpolated with a uniform distribution over ``|V| + 1`` symbols,
        the extra symbol being "any out-of-vocabulary token" — so unseen
        tokens get a small but finite probability and the distribution over
        (vocabulary ∪ {OOV}) sums to one for every context.
        """
        tokens = list(context)
        suffix = tuple(tokens[max(len(tokens) - (self.order - 1), 0) :])
        return self._prob_recursive(token, suffix)

    def _prob_recursive(self, token: str, context: Tuple[str, ...]) -> float:
        counter = self._counts.get(context)
        if not context:
            vocab_size = len(self._counts.get((), ()))
            uniform = 1.0 / (vocab_size + 1)
            if not counter:
                return uniform
            total = sum(counter.values())
            types = len(counter)
            return (counter.get(token, 0) + types * uniform) / (total + types)
        lower = self._prob_recursive(token, context[1:])
        if not counter:
            return lower
        total = sum(counter.values())
        types = len(counter)
        return (counter.get(token, 0) + types * lower) / (total + types)

    def log_prob(self, token: str, context: Sequence[str] = ()) -> float:
        """Natural log of :meth:`prob`."""
        return math.log(self.prob(token, context))

    def token_log_probs(self, sequence: Sequence[str]) -> List[float]:
        """Per-target log-probabilities over ``sequence`` (padded with boundaries).

        One entry per token of ``sequence``, plus a final entry for
        :data:`EOS` when boundaries are enabled.
        """
        tokens = self.pad(sequence)
        start = self.order - 1 if self.boundaries else 0
        return [
            self.log_prob(tokens[i], tokens[max(i - (self.order - 1), 0) : i])
            for i in range(start, len(tokens))
        ]

    def sequence_log_prob(self, sequence: Sequence[str]) -> float:
        """Sum of token log-probabilities over ``sequence`` (padded with boundaries)."""
        return sum(self.token_log_probs(sequence))

    def scored_token_count(self, sequences: Iterable[Sequence[str]]) -> int:
        """Number of prediction targets in ``sequences`` (EOS included when enabled)."""
        return sum(len(seq) + (1 if self.boundaries else 0) for seq in sequences)

    def cross_entropy(self, sequences: Sequence[Sequence[str]]) -> float:
        """Average bits per token needed to encode ``sequences`` under the model."""
        count = self.scored_token_count(sequences)
        if count == 0:
            return float("nan")
        nats = -sum(self.sequence_log_prob(seq) for seq in sequences)
        return nats / count / math.log(2)

    def perplexity(self, sequences: Sequence[Sequence[str]]) -> float:
        """``2 ** cross_entropy`` — the effective branching factor on ``sequences``."""
        return 2.0 ** self.cross_entropy(sequences)

    def oov_rate(self, sequences: Iterable[Sequence[str]]) -> float:
        """Fraction of tokens in ``sequences`` never seen as a target during fitting."""
        vocab = set(self._counts.get((), ()))
        total = 0
        unseen = 0
        for seq in sequences:
            for token in seq:
                total += 1
                if token not in vocab:
                    unseen += 1
        return unseen / total if total else float("nan")

    # ------------------------------------------------------------------
    # JSON serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serializable snapshot of the model."""
        return {
            "format": MODEL_FORMAT,
            "order": self.order,
            "boundaries": self.boundaries,
            "total_tokens": self._total_tokens,
            # Contexts are stored as lists (not joined strings) so arbitrary
            # token vocabularies — including tokens containing separators —
            # round-trip safely.
            "counts": [
                [list(context), dict(counter)]
                for context, counter in sorted(self._counts.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NGramModel":
        """Rebuild a model from :meth:`to_dict` output (legacy ``/1`` files load too)."""
        fmt = data.get("format")
        if fmt == MODEL_FORMAT:
            boundaries = bool(data.get("boundaries", True))
        elif fmt == LEGACY_MODEL_FORMAT:
            boundaries = False
        else:
            raise ValueError(f"Unsupported model format: {fmt!r}")
        model = cls(order=int(data["order"]), boundaries=boundaries)
        model._total_tokens = int(data.get("total_tokens", 0))
        for context, counter in data["counts"]:
            model._counts[tuple(context)] = Counter(
                {str(token): int(count) for token, count in counter.items()}
            )
        return model

    def save_json(self, path: Union[str, Path]) -> Path:
        """Write the model to ``path`` as JSON and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "NGramModel":
        """Load a model previously written by :meth:`save_json`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = ["MODEL_FORMAT", "LEGACY_MODEL_FORMAT", "BOS", "EOS", "NGramModel", "TokenFilter"]
