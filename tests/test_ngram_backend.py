import random

import pytest

from creative_audio_lab.evaluation.comparison import compare_backends
from creative_audio_lab.evaluation.prompts import HELD_OUT_PROMPTS
from creative_audio_lab.generators.arrangement import Arrangement, build_arrangement
from creative_audio_lab.models import DeterministicBackend, NgramMelodyBackend
from creative_audio_lab.models.ngram_backend import (
    BOOTSTRAP_PROMPTS,
    TRAINING_TOKENIZER_CONFIG,
    TokenNgramModel,
)
from creative_audio_lab.prompt_parser import parse_prompt

PROMPT = "sad orchestral theme in A minor, strings and piano, sparse"


@pytest.fixture(scope="module")
def backend():
    # Module-scoped so the bootstrap corpus is generated and fit only once.
    return NgramMelodyBackend()


# ---------------------------------------------------------------------------
# The n-gram model itself
# ---------------------------------------------------------------------------


def test_model_requires_fit_before_sampling():
    with pytest.raises(RuntimeError):
        TokenNgramModel().sample_next(["BAR_NONE"], random.Random(0))


def test_model_rejects_order_below_two():
    with pytest.raises(ValueError):
        TokenNgramModel(order=1)


def test_model_learns_a_deterministic_continuation():
    model = TokenNgramModel(order=3)
    model.fit([["a", "b", "c", "a", "b", "c", "a", "b", "c"]])
    # After context (a, b) the only continuation ever seen is c.
    assert model.sample_next(["a", "b"], random.Random(0)) == "c"


def test_model_nll_is_lower_on_training_material_than_on_unrelated_tokens():
    model = TokenNgramModel(order=3)
    training = [["a", "b", "c", "a", "b", "c"]]
    model.fit(training)
    unrelated = [["x", "y", "z", "x", "y", "z"]]
    assert model.avg_negative_log_likelihood(training) < model.avg_negative_log_likelihood(unrelated)


def test_model_nll_rejects_empty_input():
    model = TokenNgramModel()
    model.fit([["a", "b"]])
    with pytest.raises(ValueError):
        model.avg_negative_log_likelihood([])


# ---------------------------------------------------------------------------
# Bootstrap corpus hygiene
# ---------------------------------------------------------------------------


def test_bootstrap_prompts_are_disjoint_from_held_out_prompts():
    assert not set(BOOTSTRAP_PROMPTS) & set(HELD_OUT_PROMPTS)


def test_training_config_uses_the_closed_bar_vocabulary():
    assert TRAINING_TOKENIZER_CONFIG.relative_bars is True


def test_bootstrap_sequences_are_nonempty_relative_token_streams(backend):
    sequences = backend.bootstrap_sequences()
    assert len(sequences) == len(BOOTSTRAP_PROMPTS)
    for sequence in sequences:
        assert sequence, "every bootstrap prompt should tokenize to a nonempty melody"
        bar_tokens = {token for token in sequence if token.startswith("BAR_")}
        assert bar_tokens == {"BAR_NONE"}


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


def test_generate_returns_full_arrangement_with_sampled_melody(backend):
    arrangement = backend.generate(PROMPT)
    assert isinstance(arrangement, Arrangement)
    melody = arrangement.parts["melody"]
    assert len(melody) >= backend.MIN_SAMPLED_NOTES
    total_beats = arrangement.bars * 4.0
    assert all(0 <= note.pitch <= 127 for note in melody)
    assert all(note.start < total_beats for note in melody)
    assert all(note.start + note.duration <= total_beats + 1e-9 for note in melody)


def test_generate_keeps_deterministic_accompaniment(backend):
    arrangement = backend.generate(PROMPT)
    deterministic = build_arrangement(parse_prompt(PROMPT))
    for part in ("chords", "bass", "drums"):
        assert arrangement.parts[part] == deterministic.parts[part]
    assert arrangement.programs == deterministic.programs


def test_generate_is_reproducible_for_the_same_prompt_and_seed():
    a = NgramMelodyBackend(seed=7).generate(PROMPT)
    b = NgramMelodyBackend(seed=7).generate(PROMPT)
    assert a.parts["melody"] == b.parts["melody"]


def test_generate_honours_overrides(backend):
    arrangement = backend.generate(PROMPT, bpm=97, bars=4, key="D")
    assert arrangement.bpm == 97
    assert arrangement.bars == 4
    assert arrangement.controls.key == "D"


def test_backend_reports_available_without_training():
    info = NgramMelodyBackend().info()
    assert info.available is True
    assert info.requires == ()


def test_held_out_nll_is_finite_and_positive(backend):
    model = backend._ensure_trained()
    tokenizer_sequences = [
        backend._melody_tokens(build_arrangement(parse_prompt(prompt)))
        for prompt in HELD_OUT_PROMPTS[:3]
    ]
    nll = model.avg_negative_log_likelihood(tokenizer_sequences)
    assert 0.0 < nll < float("inf")


# ---------------------------------------------------------------------------
# The comparison harness
# ---------------------------------------------------------------------------


def test_compare_backends_scores_both_backends_on_the_same_prompts(backend):
    prompts = list(HELD_OUT_PROMPTS[:2])
    results = compare_backends([DeterministicBackend(), backend], prompts)
    assert set(results) == {"deterministic", "ngram"}
    for metrics in results.values():
        assert {"scale_adherence", "density_in_band", "harmonic_fit_score"} <= metrics.keys()
        assert all(value == value for value in metrics.values())  # no NaNs


def test_compare_backends_rejects_empty_prompt_list():
    with pytest.raises(ValueError):
        compare_backends([DeterministicBackend()], [])
