from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from creative_audio_lab.models.neural_events import (BOS, EOS, PAD, NeuralNoteEvent,  # noqa: E402
                                                      NoteEventCodec, pad_event_sequences)
from creative_audio_lab.models.transformer_backend import TransformerMelodyBackend  # noqa: E402
from creative_audio_lab.models.transformer_melody import (TransformerConfig,  # noqa: E402
                                                          TransformerMelodyModel)
from creative_audio_lab.models.transformer_training import (TrainingConfig, evaluate_transformer,  # noqa: E402
                                                            load_transformer_artefact,
                                                            save_transformer_artefact,
                                                            train_transformer)
from creative_audio_lab.music_theory import Note  # noqa: E402


def fixture_sequences(codec):
    sequences = []
    for offset in range(4):
        notes = [Note(i * .5, 60 + (i + offset) % 5, .5, 80 + (i % 2) * 8)
                 for i in range(10)]
        sequences.append(codec.encode_notes(notes))
    return sequences


def tiny_config(codec):
    return TransformerConfig(*codec.vocab_sizes, d_model=16, layers=1, heads=2,
                             feedforward=32, dropout=0.0, context_length=8)


def test_codec_round_trip_boundaries_padding_and_stable_ids():
    codec = NoteEventCodec()
    note = Note(0, 60, .5, 80)
    encoded = codec.encode_notes([note])
    assert encoded[0] == NeuralNoteEvent(BOS, BOS, BOS)
    assert encoded[-1] == NeuralNoteEvent(EOS, EOS, EOS)
    assert codec.encode_notes([note]) == encoded
    decoded = codec.decode_event(encoded[1])
    assert (decoded.pitch, decoded.duration, decoded.velocity) == (60, .5, 80)
    padded, mask = pad_event_sequences([encoded, encoded[:1]])
    assert padded[1][-1].pitch == PAD and mask == [[True, True, True], [True, False, False]]
    assert codec.encode_notes([]) == [NeuralNoteEvent(BOS, BOS, BOS),
                                     NeuralNoteEvent(EOS, EOS, EOS)]
    with pytest.raises(ValueError):
        codec.encode_notes([Note(0, 128, 1, 80)])
    with pytest.raises(ValueError):
        codec.decode_event(NeuralNoteEvent(EOS, EOS, EOS))


def test_forward_is_causal_and_padding_is_ignored():
    torch.manual_seed(2)
    codec = NoteEventCodec()
    model = TransformerMelodyModel(tiny_config(codec)).eval()
    sequence = fixture_sequences(codec)[0][:6]
    def run(events, padding=None):
        values = {name: torch.tensor([[getattr(e, name) for e in events]])
                  for name in ("pitch", "duration", "velocity")}
        return model(**values, padding_mask=padding)[0]
    original = run(sequence)
    changed = list(sequence)
    changed[-1] = fixture_sequences(codec)[1][3]
    assert torch.allclose(original[:, :-1], run(changed)[:, :-1], atol=1e-6)
    padded = sequence + [NeuralNoteEvent(PAD, PAD, PAD)]
    mask = torch.tensor([[False] * len(sequence) + [True]])
    assert torch.allclose(original, run(padded, mask)[:, :len(sequence)], atol=1e-6)


def test_smoke_training_metrics_checkpoint_and_seeded_generation(tmp_path):
    codec = NoteEventCodec()
    sequences = fixture_sequences(codec)
    model, history, epoch = train_transformer(
        sequences[:3], sequences[3:], tiny_config(codec),
        TrainingConfig(epochs=2, batch_size=2, learning_rate=.01, patience=2,
                       seed=7, device="cpu"))
    metrics = evaluate_transformer(model, sequences[3:], batch_size=2)
    assert metrics["notes"] == 10 and math.isfinite(metrics["bits_per_note"])
    assert metrics["bits_per_note"] == pytest.approx(
        metrics["pitch_bits_per_note"] + metrics["duration_bits_per_note"]
        + metrics["velocity_bits_per_note"])
    assert len(history) == 2 and epoch in (1, 2)
    path = save_transformer_artefact(tmp_path / "model", model, codec,
                                    {"training_history": history, "selected_epoch": epoch})
    loaded, loaded_codec, metadata = load_transformer_artefact(path)
    assert loaded_codec.to_dict() == codec.to_dict()
    assert metadata["selected_epoch"] == epoch
    for left, right in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(left, right)
    first = TransformerMelodyBackend(path, sampling_seed=42, top_k=4).generate(
        "calm piano melody in C major", bars=2)
    second = TransformerMelodyBackend(path, sampling_seed=42, top_k=4).generate(
        "calm piano melody in C major", bars=2)
    assert first.parts["melody"] == second.parts["melody"]
    assert first.parts["chords"] == second.parts["chords"]
