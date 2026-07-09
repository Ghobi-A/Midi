"""Tests for the Stage 2 NgramMelodyBackend and its training utilities."""

import io

import mido
import pytest

from creative_audio_lab.export import export_arrangement_files
from creative_audio_lab.generators.arrangement import Arrangement
from creative_audio_lab.models import DeterministicBackend, NgramMelodyBackend, get_backend
from creative_audio_lab.models.ngram_training import (
    build_bootstrap_corpus,
    melody_token_stream,
    train_ngram_model,
    transpose_notes,
)
from creative_audio_lab.music_theory import Note, scale_pitch_classes

PROMPT = "dark orchestral boss battle theme, 140 BPM, brass ostinato, strings, piano arps"


@pytest.fixture(scope="module")
def arrangement():
    return NgramMelodyBackend().generate(PROMPT)


@pytest.fixture(scope="module")
def deterministic_arrangement():
    return DeterministicBackend().generate(PROMPT)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def test_transpose_notes_shifts_and_clamps():
    notes = [Note(start=0.0, pitch=60, duration=1.0), Note(start=1.0, pitch=1, duration=1.0)]
    down = transpose_notes(notes, -5)
    assert [n.pitch for n in down] == [55, 0]
    assert [n.start for n in down] == [0.0, 1.0]


def test_melody_token_stream_is_note_triples_only():
    notes = [Note(start=0.0, pitch=60, duration=1.0, velocity=96)]
    stream = melody_token_stream(notes)
    assert stream == ["NOTE_ON_60", "VELOCITY_24", "DURATION_4"]


def test_bootstrap_corpus_is_nonempty_token_sequences():
    corpus = build_bootstrap_corpus(
        prompts=("trap melody with dark bells",), keys=("C", "G"), bpms=(None,)
    )
    assert len(corpus) == 2
    for sequence in corpus:
        assert sequence  # non-empty
        assert all(
            token.split("_")[0] in ("NOTE", "VELOCITY", "DURATION") for token in sequence
        )


def test_train_ngram_model_from_corpus():
    corpus = build_bootstrap_corpus(prompts=("calm ambient piano",), keys=("C",), bpms=(None,))
    model = train_ngram_model(corpus, order=3)
    assert model.total_tokens == len(corpus[0])
    assert model.sample_next([], seed=0) in model.vocabulary


# ---------------------------------------------------------------------------
# Backend behaviour
# ---------------------------------------------------------------------------


def test_backend_returns_valid_arrangement(arrangement):
    assert isinstance(arrangement, Arrangement)
    assert set(arrangement.parts) == {"chords", "melody", "bass", "drums"}
    melody = arrangement.parts["melody"]
    assert melody
    total_beats = arrangement.bars * 4.0
    assert all(0.0 <= note.start < total_beats for note in melody)
    assert all(note.end() <= total_beats + 1e-9 for note in melody)
    assert all(0 <= note.pitch <= 127 for note in melody)
    assert all(1 <= note.velocity <= 127 for note in melody)


def test_backend_keeps_deterministic_scaffolding(arrangement, deterministic_arrangement):
    for part in ("chords", "bass", "drums"):
        assert arrangement.parts[part] == deterministic_arrangement.parts[part]
    assert arrangement.programs == deterministic_arrangement.programs
    assert arrangement.bpm == deterministic_arrangement.bpm
    assert arrangement.bars == deterministic_arrangement.bars
    assert arrangement.chord_events == deterministic_arrangement.chord_events


def test_backend_melody_differs_from_deterministic(arrangement, deterministic_arrangement):
    assert arrangement.parts["melody"] != deterministic_arrangement.parts["melody"]


def test_backend_is_reproducible_per_prompt(arrangement):
    again = NgramMelodyBackend().generate(PROMPT)
    assert again.parts["melody"] == arrangement.parts["melody"]


def test_backend_honours_overrides():
    result = NgramMelodyBackend().generate(PROMPT, bpm=97, bars=4, key="D")
    assert result.bpm == 97
    assert result.bars == 4
    assert result.controls.key == "D"
    assert all(note.end() <= 16.0 + 1e-9 for note in result.parts["melody"])


def test_backend_melody_is_mostly_in_key(arrangement):
    # The corpus is key-normalized, not scale-constrained, so adherence is
    # high but not guaranteed to be 1.0 — assert a sane floor, not perfection.
    melody = arrangement.parts["melody"]
    scale = scale_pitch_classes(arrangement.controls.key, arrangement.controls.mode)
    in_scale = sum(1 for note in melody if note.pitch % 12 in scale)
    assert in_scale / len(melody) > 0.5


def test_backend_with_user_supplied_model(tmp_path):
    corpus = build_bootstrap_corpus(prompts=("romantic piano ballad",), keys=("C",), bpms=(None,))
    path = train_ngram_model(corpus).save_json(tmp_path / "custom.json")
    result = NgramMelodyBackend(model_path=path).generate(PROMPT)
    assert isinstance(result, Arrangement)
    assert result.parts["melody"]


# ---------------------------------------------------------------------------
# Registry + export integration
# ---------------------------------------------------------------------------


def test_registry_provides_ngram_backend():
    backend = get_backend("ngram_melody")
    assert isinstance(backend, NgramMelodyBackend)
    info = backend.info()
    assert info.available is True
    assert "n-gram" in info.description


def test_exported_midi_is_valid(arrangement):
    files = export_arrangement_files(arrangement)
    assert set(files) == {"full_arrangement.mid", "chords.mid", "melody.mid", "bass.mid", "drums.mid"}
    for name, data in files.items():
        parsed = mido.MidiFile(file=io.BytesIO(data))
        assert parsed.tracks, name
    melody_mid = mido.MidiFile(file=io.BytesIO(files["melody.mid"]))
    note_ons = [m for track in melody_mid.tracks for m in track if m.type == "note_on" and m.velocity > 0]
    assert len(note_ons) == len(arrangement.parts["melody"])
