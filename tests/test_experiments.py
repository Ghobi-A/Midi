import json

import pytest

from creative_audio_lab.evaluation import aggregate, analyse_generation
from creative_audio_lab.experiments import Candidate, ValidationSelector, run_experiment
from creative_audio_lab.models.artefact import load_artefact
from creative_audio_lab.models.ngram_training import train_melody_model
from creative_audio_lab.music_theory import Note
from fixture_corpus import clean_only_manifest


def _model():
    return train_melody_model([["NOTE_ON_60", "VELOCITY_20", "DURATION_4"] * 4], order=1)


def test_selection_cannot_open_test_before_validation_choice():
    selector = ValidationSelector()
    with pytest.raises(RuntimeError, match="select on validation"):
        selector.evaluate_test("one", _model(), [])


def test_only_validation_winner_can_be_tested():
    selector = ValidationSelector()
    selector.add_validation(Candidate("one", "factorised", 1), {"bits_per_note": 2.0, "notes": 10})
    selector.add_validation(Candidate("two", "factorised", 2), {"bits_per_note": 1.0, "notes": 10})
    assert selector.select() == "two"
    with pytest.raises(ValueError, match="frozen selection"):
        selector.evaluate_test("one", _model(), [])


def test_aggregation_reports_sample_spread_and_count():
    assert aggregate([1, 2, 3]) == {"mean": 2.0, "std": 1.0, "n": 3}


def test_error_analysis_is_interpretable():
    notes = [Note(start=float(i), pitch=60, duration=1.0) for i in range(12)]
    result = analyse_generation(notes, scale_pcs=[0, 2, 4, 5, 7, 9, 11], expected_beats=12)
    assert result["flags"]["repeated_note_collapse"] is True
    assert result["measurements"]["dominant_pitch_ratio"] == 1.0


def test_run_tracks_identity_selection_and_artefact(tmp_path):
    manifest = clean_only_manifest(tmp_path / "corpus")
    output = run_experiment(manifest, tmp_path / "runs", samples=1)
    run = json.loads((output / "run.json").read_text())
    assert output.name == run["experiment_id"]
    assert run["split"]["level"] == "composition"
    assert set(run["test_metrics"]) == {run["selection"]["selected"]}
    assert run["generation_metrics"]["aggregate"]["deterministic"]["note_density"]["n"] == 4
    _, metadata = load_artefact(output / "selected_model.json")
    assert metadata["experiment_id"] == run["experiment_id"]


def test_corrupt_model_artefact_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"format":"creative-audio-lab.melody-model/1","model":{}}')
    with pytest.raises((ValueError, KeyError)):
        load_artefact(path)
