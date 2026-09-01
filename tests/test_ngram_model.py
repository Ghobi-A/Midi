"""Unit tests for the Stage 2 n-gram model: fit, backoff, sampling, JSON I/O."""

import json
import random

import pytest

import math

from creative_audio_lab.models.ngram_model import (
    BOS,
    EOS,
    LEGACY_MODEL_FORMAT,
    MODEL_FORMAT,
    NGramModel,
)

# A tiny corpus with unambiguous trigram structure: after ("a", "b") the only
# continuation ever seen is "c"; after ("b", "c") it's "d".
CORPUS = [
    ["a", "b", "c", "d"],
    ["a", "b", "c", "d"],
    ["x", "b", "c", "d"],
]


@pytest.fixture
def model():
    return NGramModel(order=3).fit(CORPUS)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def test_order_must_be_positive():
    with pytest.raises(ValueError):
        NGramModel(order=0)


def test_fit_counts_and_vocabulary(model):
    # 12 corpus tokens + one EOS per sequence; BOS is context-only.
    assert model.total_tokens == 15
    assert model.vocabulary == [EOS, "a", "b", "c", "d", "x"]
    assert model.context_counts([BOS, BOS]) == {"a": 2, "x": 1}
    assert model.context_counts(["c", "d"]) == {EOS: 3}
    assert model.context_counts(["a", "b"]) == {"c": 2}
    assert model.context_counts(["b"]) == {"c": 3}
    assert model.context_counts([])["b"] == 3


def test_fit_is_additive(model):
    model.fit([["a", "b", "z"]])
    assert model.context_counts(["a", "b"]) == {"c": 2, "z": 1}


# ---------------------------------------------------------------------------
# Sampling and backoff
# ---------------------------------------------------------------------------


def test_sample_follows_seen_trigram_context(model):
    # ("a", "b") has exactly one observed continuation, so any sample returns it.
    assert model.sample_next(["a", "b"], seed=0) == "c"


def test_backoff_to_shorter_context(model):
    # ("never", "b") was never seen, but ("b",) was: backoff must find "c".
    assert model.sample_next(["never", "b"], seed=0) == "c"


def test_backoff_to_unigram_for_fully_unseen_context(model):
    # Nothing about this context was ever seen; the unigram distribution is used.
    token = model.sample_next(["nope", "nada"], seed=0)
    assert token in model.vocabulary


def test_unfitted_model_returns_none():
    assert NGramModel(order=3).sample_next(["a"], seed=0) is None


def test_allowed_filter_restricts_candidates(model):
    token = model.sample_next([], allowed=lambda t: t in ("a", "x"), seed=0)
    assert token in ("a", "x")


def test_allowed_filter_with_no_survivors_returns_none(model):
    assert model.sample_next([], allowed=lambda t: False, seed=0) is None


def test_top_k_one_is_argmax(model):
    # EOS, "b", "c", "d" are the most frequent unigrams (3 each); ties break by token.
    assert model.sample_next([], top_k=1, seed=123) == EOS
    assert model.sample_next([], top_k=1, seed=123, allowed=lambda t: t != EOS) == "b"


def test_temperature_must_be_positive(model):
    with pytest.raises(ValueError):
        model.sample_next([], temperature=0.0)


def test_deterministic_sampling_with_fixed_seed(model):
    first = model.sample_sequence(["a"], length=20, seed=42, allowed=lambda t: t != EOS)
    second = model.sample_sequence(["a"], length=20, seed=42, allowed=lambda t: t != EOS)
    assert first == second
    assert len(first) == 20
    assert all(token in model.vocabulary for token in first)


def test_different_seeds_can_differ(model):
    samples = {tuple(model.sample_sequence([], length=10, seed=s)) for s in range(20)}
    assert len(samples) > 1


def test_sample_next_accepts_external_rng(model):
    rng = random.Random(7)
    token = model.sample_next([], rng=rng)
    assert token in model.vocabulary


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def test_json_roundtrip_preserves_counts_and_sampling(model, tmp_path):
    path = model.save_json(tmp_path / "model.json")
    restored = NGramModel.load_json(path)

    assert restored.order == model.order
    assert restored.total_tokens == model.total_tokens
    assert restored.vocabulary == model.vocabulary
    assert restored.context_counts(["a", "b"]) == model.context_counts(["a", "b"])
    assert restored.sample_sequence(["a"], length=30, seed=9) == model.sample_sequence(
        ["a"], length=30, seed=9
    )


def test_saved_file_is_plain_json(model, tmp_path):
    path = model.save_json(tmp_path / "model.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == MODEL_FORMAT
    assert data["order"] == 3


def test_legacy_format_loads_without_boundaries():
    legacy = {"format": LEGACY_MODEL_FORMAT, "order": 2, "counts": [[[], {"a": 1}], [["a"], {"a": 1}]]}
    model = NGramModel.from_dict(legacy)
    assert model.boundaries is False
    assert model.vocabulary == ["a"]


def test_from_dict_rejects_unknown_format():
    with pytest.raises(ValueError, match="format"):
        NGramModel.from_dict({"format": "something-else", "order": 3, "counts": []})


# ---------------------------------------------------------------------------
# Boundaries and smoothed probabilities
# ---------------------------------------------------------------------------


def test_bos_is_never_sampled_and_eos_ends_sequences(model):
    for seed in range(50):
        assert model.sample_next([BOS, BOS], seed=seed) != BOS
    # Every corpus sequence ends "c d <EOS>", so sampling from BOS stops by itself.
    generated = model.sample_sequence(length=100, seed=3)
    assert generated[-2:] == ["c", "d"]
    assert EOS not in generated


def test_boundaries_can_be_disabled():
    plain = NGramModel(order=3, boundaries=False).fit(CORPUS)
    assert plain.total_tokens == 12
    assert EOS not in plain.vocabulary


def _distribution_sums_to_one(model, context):
    vocab = model.vocabulary
    total = sum(model.prob(token, context) for token in vocab)
    total += model.prob("<never-seen>", context)  # the single OOV symbol
    return total


def test_smoothed_distribution_sums_to_one(model):
    for context in ([], ["a"], ["a", "b"], ["never", "seen"], ["x", "b"]):
        assert math.isclose(_distribution_sums_to_one(model, context), 1.0, rel_tol=1e-9)


def test_oov_token_has_finite_log_prob(model):
    assert math.isfinite(model.log_prob("<never-seen>", ["a", "b"]))
    assert model.prob("<never-seen>", ["a", "b"]) < model.prob("c", ["a", "b"])


def test_seen_continuation_dominates(model):
    assert model.prob("c", ["a", "b"]) > 0.8


def test_training_data_has_lower_perplexity_than_shuffled_copy(model):
    shuffled = [list(reversed(seq)) for seq in CORPUS]
    assert model.perplexity(CORPUS) < model.perplexity(shuffled)
    assert model.cross_entropy(CORPUS) > 0.0


def test_oov_rate(model):
    assert model.oov_rate(CORPUS) == 0.0
    assert model.oov_rate([["a", "zzz"]]) == 0.5


def test_unfitted_model_is_uniform():
    empty = NGramModel(order=2)
    assert empty.prob("anything", ["ctx"]) == 1.0
