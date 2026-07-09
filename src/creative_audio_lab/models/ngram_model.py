"""A small, dependency-free N-gram model over symbolic token strings.

This is the Stage 2 statistical baseline described in ``docs/ML_ROADMAP.md``
(item 7: "Markov / n-gram continuation baseline"). It counts token
transitions of every context length up to ``order - 1``, backs off to
shorter contexts when a context was never seen, and samples continuations
with temperature and optional top-k truncation. Sampling is deterministic
when given a seed, and models round-trip through plain JSON — no torch,
sklearn, or any other heavy dependency is involved (or allowed) here.

The model is generic over hashable string tokens; nothing in this module
knows about music. The melody-specific corpus building and decoding live in
:mod:`creative_audio_lab.models.ngram_training` and
:mod:`creative_audio_lab.models.ngram_backend`.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

#: Version tag written into saved models so future format changes can be detected.
MODEL_FORMAT = "creative-audio-lab.ngram/1"

#: Predicate deciding whether a candidate next token may be sampled.
TokenFilter = Callable[[str], bool]


class NGramModel:
    """Count-based N-gram model with backoff, temperature, and top-k sampling.

    Parameters
    ----------
    order:
        Maximum n-gram order. ``order=3`` (the default) predicts the next
        token from up to the two preceding tokens.
    """

    def __init__(self, order: int = 3) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        # counts[context_tuple] -> Counter of next tokens, for every context
        # length from 0 (unigram) to order - 1.
        self._counts: Dict[Tuple[str, ...], Counter] = {}
        self._total_tokens = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, sequences: Iterable[Sequence[str]]) -> "NGramModel":
        """Accumulate n-gram counts from token sequences (additive across calls)."""
        for sequence in sequences:
            tokens = list(sequence)
            for i, token in enumerate(tokens):
                self._total_tokens += 1
                for ctx_len in range(min(i, self.order - 1) + 1):
                    context = tuple(tokens[i - ctx_len : i])
                    self._counts.setdefault(context, Counter())[token] += 1
        return self

    @property
    def total_tokens(self) -> int:
        """Number of training tokens seen so far."""
        return self._total_tokens

    @property
    def vocabulary(self) -> List[str]:
        """Every token observed during fitting, sorted for determinism."""
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
                if allowed is None or allowed(token)
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
        backoff level.
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
        """Sample ``length`` tokens continuing ``seed_context`` (context not included)."""
        rng = random.Random(seed)
        context = list(seed_context)
        generated: List[str] = []
        for _ in range(length):
            token = self.sample_next(
                context, temperature=temperature, top_k=top_k, allowed=allowed, rng=rng
            )
            if token is None:
                break
            generated.append(token)
            context.append(token)
        return generated

    # ------------------------------------------------------------------
    # JSON serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serializable snapshot of the model."""
        return {
            "format": MODEL_FORMAT,
            "order": self.order,
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
        """Rebuild a model from :meth:`to_dict` output."""
        if data.get("format") != MODEL_FORMAT:
            raise ValueError(f"Unsupported model format: {data.get('format')!r}")
        model = cls(order=int(data["order"]))
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


__all__ = ["MODEL_FORMAT", "NGramModel", "TokenFilter"]
