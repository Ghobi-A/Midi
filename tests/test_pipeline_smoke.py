"""End-to-end smoke test: manifest → training artefact → evaluation → app-loadable.

Runs the real command-line entry points against the synthetic fixture
corpus, so the whole documented workflow is exercised, not just the
library functions underneath it.
"""

import io
import json
import subprocess
import sys
from pathlib import Path

import mido
import pytest

from creative_audio_lab.export import export_arrangement_files
from creative_audio_lab.models import NgramMelodyBackend
from creative_audio_lab.models.artefact import REQUIRED_METADATA_KEYS, load_artefact
from fixture_corpus import build_fixture_corpus

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "scripts" / "train_ngram_melody.py"
EVALUATE = ROOT / "scripts" / "evaluate_melody_models.py"
PROMPT = "hopeful piano and strings theme"


def _run(*args):
    result = subprocess.run(
        [sys.executable, *[str(arg) for arg in args]],
        capture_output=True, text=True, cwd=ROOT,
    )
    return result


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return build_fixture_corpus(tmp_path_factory.mktemp("smoke"))


@pytest.fixture(scope="module")
def artefacts(corpus, tmp_path_factory):
    out = tmp_path_factory.mktemp("artefacts")
    bootstrap = out / "bootstrap.json"
    real = out / "real.json"
    first = _run(TRAIN, "--output", bootstrap)
    assert first.returncode == 0, first.stderr
    second = _run(
        TRAIN, "--manifest", corpus["manifest_path"], "--allow-reference-only",
        "--output", real,
    )
    assert second.returncode == 0, second.stderr
    return {"bootstrap": bootstrap, "real": real, "train_stdout": second.stdout}


# ---------------------------------------------------------------------------
# Training CLI
# ---------------------------------------------------------------------------


def test_training_reports_splits_and_held_out_metrics(artefacts):
    stdout = artefacts["train_stdout"]
    assert "Corpus: manifest" in stdout
    for split in ("train", "val", "test"):
        assert split in stdout
    assert "bits/note" in stdout
    assert "rejected" in stdout


def test_manifest_training_refuses_unrightsed_data_without_the_flag(corpus, tmp_path):
    result = _run(TRAIN, "--manifest", corpus["manifest_path"], "--output", tmp_path / "x.json")
    assert result.returncode != 0
    assert "cannot be used for training" in (result.stderr + result.stdout)


def test_artefact_carries_full_training_metadata(artefacts):
    model, metadata = load_artefact(artefacts["real"])
    for key in REQUIRED_METADATA_KEYS:
        assert key in metadata, f"artefact is missing {key}"
    assert metadata["corpus"]["source"] == "manifest"
    assert metadata["corpus"]["corpus_hash"]
    assert metadata["corpus"]["datasets"][0]["license"] == "CC0 1.0"
    assert metadata["split"]["level"] == "composition"
    assert metadata["software"]["git_commit"] or metadata["software"]["package_version"]
    assert metadata["tokenizer"]["positions_per_beat"] == 4
    assert metadata["metrics"]["test"]["notes"] > 0
    assert metadata["created_at"]


def test_artefact_split_assignment_is_disjoint(artefacts):
    _, metadata = load_artefact(artefacts["real"])
    splits = metadata["split"]["compositions"]
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train and test
    assert not (train & val) and not (train & test) and not (val & test)


def test_reference_only_material_never_reaches_training(artefacts):
    _, metadata = load_artefact(artefacts["real"])
    trained_on = metadata["split"]["compositions"]["train"]
    assert not any(cid.startswith("fixture-fan-archive/") for cid in trained_on)
    flagged = [d for d in metadata["corpus"]["datasets"] if d["name"] == "fixture-fan-archive"]
    assert flagged and flagged[0]["reference_only"] is True


def test_bootstrap_artefact_is_labelled_synthetic(artefacts):
    _, metadata = load_artefact(artefacts["bootstrap"])
    assert metadata["corpus"]["source"] == "synthetic-bootstrap"
    assert "generalisation" in metadata["corpus"]["note"]


def test_flat_model_kind_is_still_trainable(corpus, tmp_path):
    output = tmp_path / "flat.json"
    result = _run(
        TRAIN, "--manifest", corpus["manifest_path"], "--allow-reference-only",
        "--model-kind", "flat", "--order", "4", "--output", output,
    )
    assert result.returncode == 0, result.stderr
    _, metadata = load_artefact(output)
    assert metadata["model_kind"] == "flat"
    assert metadata["orders"] == {"flat": 4}


# ---------------------------------------------------------------------------
# Evaluation CLI
# ---------------------------------------------------------------------------


def test_evaluation_compares_models_on_the_same_held_out_split(corpus, artefacts, tmp_path):
    table = tmp_path / "benchmark.md"
    result = _run(
        EVALUATE, artefacts["bootstrap"], artefacts["real"],
        "--manifest", corpus["manifest_path"], "--allow-reference-only",
        "--output", table,
    )
    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "bits/note" in stdout and "pitch bits/note" in stdout
    assert "synthetic bootstrap" in stdout and "fixture-cc0" in stdout
    assert "not judgments of musical quality" in stdout
    assert table.read_text(encoding="utf-8").count("\n") >= 4


def test_evaluation_reports_leakage(corpus, artefacts, tmp_path):
    # Forge an artefact that claims to have trained on the evaluation split.
    data = json.loads(Path(artefacts["real"]).read_text(encoding="utf-8"))
    splits = data["metadata"]["split"]["compositions"]
    splits["train"] = splits["train"] + splits["test"]
    leaky = tmp_path / "leaky.json"
    leaky.write_text(json.dumps(data), encoding="utf-8")

    result = _run(
        EVALUATE, leaky, "--manifest", corpus["manifest_path"], "--allow-reference-only",
    )
    assert result.returncode == 1
    assert "not held out" in result.stdout


def test_evaluation_warns_when_the_corpus_changed(corpus, artefacts, tmp_path):
    data = json.loads(Path(artefacts["real"]).read_text(encoding="utf-8"))
    data["metadata"]["corpus"]["corpus_hash"] = "0" * 64
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(data), encoding="utf-8")

    result = _run(
        EVALUATE, stale, "--manifest", corpus["manifest_path"], "--allow-reference-only",
    )
    assert result.returncode == 0
    assert "the datasets differ" in result.stdout


# ---------------------------------------------------------------------------
# The artefact is usable by the app's backend
# ---------------------------------------------------------------------------


def test_backend_generates_valid_midi_from_the_trained_artefact(artefacts):
    backend = NgramMelodyBackend(model_path=artefacts["real"])
    arrangement = backend.generate(PROMPT, bars=8)
    assert arrangement.parts["melody"]
    assert backend.artefact_metadata["corpus"]["source"] == "manifest"

    files = export_arrangement_files(arrangement)
    for name, data in files.items():
        assert mido.MidiFile(file=io.BytesIO(data)).tracks, name


def test_compare_backends_accepts_a_trained_model(artefacts):
    result = _run(
        ROOT / "scripts" / "compare_backends.py",
        "--prompt", PROMPT, "--bars", "4", "--ngram-model", artefacts["real"],
    )
    assert result.returncode == 0, result.stderr
    assert "ngram_melody" in result.stdout
    assert str(artefacts["real"]) in result.stdout
