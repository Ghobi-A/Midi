"""Tests for the factorised note-event model, including the melodic-dependency regression.

The regression corpus makes the next pitch a deterministic function of the
previous pitch while every velocity and duration is identical. A model
whose pitch context contains the previous pitch predicts it perfectly; the
original flat order-3 model — whose NOTE_ON context was
``(VELOCITY_prev, DURATION_prev)`` — cannot tell the cycles apart.
"""

import math

import pytest

from creative_audio_lab.models import NgramMelodyBackend
from creative_audio_lab.models.artefact import load_any_model, save_artefact
from creative_audio_lab.models.ngram_model import NGramModel
from creative_audio_lab.models.ngram_training import (
    build_bootstrap_corpus,
    train_melody_model,
)
from creative_audio_lab.models.note_event_model import (
    NoteEventModel,
    evaluate_flat_model,
    evaluate_melody_model,
    split_note_stream,
)

# Two pitch cycles that share no pitches: 60→64→67→60... and 62→65→69→62...
CYCLES = ((60, 64, 67), (62, 65, 69))


def _cycle_stream(cycle, length=30, start=0):
    stream = []
    for i in range(length):
        pitch = cycle[(start + i) % len(cycle)]
        stream += [f"NOTE_ON_{pitch}", "VELOCITY_24", "DURATION_4"]
    return stream


@pytest.fixture(scope="module")
def dependency_corpus():
    return [_cycle_stream(cycle, start=s) for cycle in CYCLES for s in range(3)]


@pytest.fixture(scope="module")
def held_out():
    # Unseen phase offsets/lengths of the same cycles.
    return [_cycle_stream(cycle, length=17, start=1) for cycle in CYCLES]


def test_split_note_stream_roundtrip():
    stream = ["NOTE_ON_60", "VELOCITY_24", "DURATION_4", "NOTE_ON_62", "VELOCITY_20", "DURATION_2"]
    pitches, velocities, durations = split_note_stream(stream)
    assert pitches == ["NOTE_ON_60", "NOTE_ON_62"]
    assert velocities == ["VELOCITY_24", "VELOCITY_20"]
    assert durations == ["DURATION_4", "DURATION_2"]


def test_split_note_stream_rejects_malformed():
    with pytest.raises(ValueError):
        split_note_stream(["VELOCITY_1", "NOTE_ON_60", "DURATION_4"])


# ---------------------------------------------------------------------------
# The regression: does the previous pitch change the next-pitch distribution?
# ---------------------------------------------------------------------------


def test_factorised_model_learns_pitch_to_pitch_transitions(dependency_corpus, held_out):
    model = train_melody_model(dependency_corpus, "factorised", order=3)
    assert isinstance(model, NoteEventModel)
    # After 60 the corpus only ever continues to 64; after 62, only to 65.
    assert model.pitch_prob("NOTE_ON_64", ["NOTE_ON_60"]) > 0.9
    assert model.pitch_prob("NOTE_ON_65", ["NOTE_ON_62"]) > 0.9
    assert model.pitch_prob("NOTE_ON_65", ["NOTE_ON_60"]) < 0.05
    metrics = model.evaluate(held_out)
    assert metrics["pitch_bits_per_note"] < 0.5
    assert metrics["pitch_oov_rate"] == 0.0


def test_flat_order3_model_does_not_see_the_previous_pitch(dependency_corpus, held_out):
    flat = train_melody_model(dependency_corpus, "flat", order=3)
    assert isinstance(flat, NGramModel)
    context = ["NOTE_ON_60", "VELOCITY_24", "DURATION_4"]
    p_correct = flat.prob("NOTE_ON_64", context)
    p_wrong_cycle = flat.prob("NOTE_ON_65", context)
    # Same (VELOCITY, DURATION) context for every note ⇒ pitch is unconditional.
    assert math.isclose(p_correct, p_wrong_cycle, rel_tol=1e-6)
    flat_metrics = evaluate_flat_model(flat, held_out)
    factorised_metrics = train_melody_model(dependency_corpus, "factorised").evaluate(held_out)
    assert flat_metrics["pitch_bits_per_note"] > 2.0  # ≈ log2(6) for six equiprobable pitches
    assert factorised_metrics["pitch_bits_per_note"] < flat_metrics["pitch_bits_per_note"] - 1.5


def test_flat_order4_model_recovers_the_dependency(dependency_corpus, held_out):
    # The user's "minimum correction": with order 4 the previous pitch is in context.
    flat4 = train_melody_model(dependency_corpus, "flat", order=4)
    assert evaluate_flat_model(flat4, held_out)["pitch_bits_per_note"] < 0.5


# ---------------------------------------------------------------------------
# Evaluation helpers, sampling, serialisation
# ---------------------------------------------------------------------------


def test_evaluate_reports_comparable_units(dependency_corpus, held_out):
    model = train_melody_model(dependency_corpus, "factorised")
    metrics = evaluate_melody_model(model, held_out)
    assert metrics["notes"] == sum(len(s) // 3 for s in held_out)
    total = (
        metrics["pitch_bits_per_note"]
        + metrics["velocity_bits_per_note"]
        + metrics["duration_bits_per_note"]
    )
    assert math.isclose(metrics["bits_per_note"], total)
    assert math.isclose(metrics["perplexity_per_note"], 2 ** total)
    for key in ("pitch_bits_per_note", "velocity_bits_per_note", "duration_bits_per_note"):
        assert metrics[key] >= 0.0


def test_evaluate_on_empty_is_nan():
    metrics = NoteEventModel().evaluate([])
    assert metrics["notes"] == 0 and math.isnan(metrics["bits_per_note"])


def test_sample_note_respects_filters(dependency_corpus):
    import random

    model = train_melody_model(dependency_corpus, "factorised")
    rng = random.Random(0)
    for _ in range(30):
        triple = model.sample_note(
            ["NOTE_ON_60"], ["VELOCITY_24"], ["DURATION_4"], rng=rng,
            allowed_pitch=lambda t: int(t.rpartition("_")[2]) >= 65,
        )
        assert triple is not None
        pitch, velocity, duration = triple
        assert int(pitch.rpartition("_")[2]) >= 65
        assert velocity.startswith("VELOCITY_") and duration.startswith("DURATION_")


def test_sample_note_never_returns_boundary_tokens(dependency_corpus):
    import random

    model = train_melody_model(dependency_corpus, "factorised")
    rng = random.Random(1)
    # Contexts at the end of every training piece are followed by EOS only.
    for _ in range(50):
        triple = model.sample_note(["NOTE_ON_67", "NOTE_ON_60"], ["VELOCITY_24"], ["DURATION_4"], rng=rng)
        assert triple is not None
        assert all(not token.startswith("<") for token in triple)


def test_json_roundtrip(dependency_corpus, tmp_path):
    model = train_melody_model(dependency_corpus, "factorised", order=4, duration_order=2)
    path = model.save_json(tmp_path / "note_event.json")
    restored = NoteEventModel.load_json(path)
    assert restored.orders == {"pitch": 4, "velocity": 2, "duration": 2}
    assert restored.total_notes == model.total_notes
    assert restored.pitch_prob("NOTE_ON_64", ["NOTE_ON_60"]) == model.pitch_prob("NOTE_ON_64", ["NOTE_ON_60"])


def test_train_melody_model_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        train_melody_model([], "quantum")


# ---------------------------------------------------------------------------
# Backend integration with both kinds and artefacts
# ---------------------------------------------------------------------------

PROMPT = "nostalgic jrpg town theme with piano and flute"


@pytest.mark.parametrize("kind", ["factorised", "flat"])
def test_backend_generates_with_each_model_kind(kind):
    result = NgramMelodyBackend(model_kind=kind).generate(PROMPT, bars=4)
    melody = result.parts["melody"]
    assert melody
    assert all(note.end() <= 16.0 + 1e-9 for note in melody)


def test_backend_default_is_factorised():
    backend = NgramMelodyBackend()
    assert isinstance(backend.model, NoteEventModel)
    assert backend.artefact_metadata is None


def test_backend_loads_artefact_and_exposes_metadata(tmp_path):
    corpus = build_bootstrap_corpus(prompts=("romantic piano ballad",), keys=("C",), bpms=(None,))
    model = train_melody_model(corpus, "factorised")
    path = save_artefact(model, {"corpus": {"source": "test"}}, tmp_path / "artefact.json")
    loaded, metadata = load_any_model(path)
    assert isinstance(loaded, NoteEventModel)
    assert metadata["corpus"] == {"source": "test"}
    assert metadata["model_kind"] == "factorised"
    backend = NgramMelodyBackend(model_path=path)
    assert backend.generate(PROMPT, bars=4).parts["melody"]
    assert backend.artefact_metadata["orders"]["pitch"] == 3


def test_backend_loads_bare_flat_model(tmp_path):
    corpus = build_bootstrap_corpus(prompts=("romantic piano ballad",), keys=("C",), bpms=(None,))
    path = train_melody_model(corpus, "flat").save_json(tmp_path / "flat.json")
    backend = NgramMelodyBackend(model_path=path)
    assert isinstance(backend.model, NGramModel)
    assert backend.artefact_metadata is None
