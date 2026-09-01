"""Tests for local dataset loading: relative-path identity and content hashes."""


import pytest

from creative_audio_lab.data.midi_dataset_loader import (
    corpus_hash,
    file_sha256,
    iter_dataset_files,
    iter_midi_paths,
    load_dataset_notes,
)
from creative_audio_lab.export.midi_export import export_notes_to_bytes
from creative_audio_lab.music_theory import Note


def _write_midi(path, pitch=60):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = export_notes_to_bytes(
        [Note(start=0.0, pitch=pitch, duration=1.0, velocity=90)],
        name="melody", program=0, bpm=120.0,
    )
    path.write_bytes(data)
    return path


def test_identically_named_files_in_different_dirs_both_survive(tmp_path):
    _write_midi(tmp_path / "bach" / "song.mid", pitch=60)
    _write_midi(tmp_path / "mozart" / "song.mid", pitch=72)

    dataset = load_dataset_notes(tmp_path)
    assert set(dataset) == {"bach/song.mid", "mozart/song.mid"}
    assert dataset["bach/song.mid"]["melody"][0].pitch == 60
    assert dataset["mozart/song.mid"]["melody"][0].pitch == 72


def test_iter_dataset_files_reports_ids_and_hashes(tmp_path):
    _write_midi(tmp_path / "a" / "one.mid", pitch=60)
    _write_midi(tmp_path / "two.midi", pitch=60)

    entries = list(iter_dataset_files(tmp_path))
    assert sorted(entry.composition_id for entry in entries) == ["a/one.mid", "two.midi"]
    # Identical content ⇒ identical hash, different identity.
    assert entries[0].sha256 == entries[1].sha256
    assert all(len(entry.sha256) == 64 for entry in entries)


def test_file_sha256_changes_with_content(tmp_path):
    first = _write_midi(tmp_path / "a.mid", pitch=60)
    second = _write_midi(tmp_path / "b.mid", pitch=61)
    assert file_sha256(first) != file_sha256(second)


def test_corpus_hash_is_order_independent_and_content_sensitive():
    assert corpus_hash(["a", "b"]) == corpus_hash(["b", "a"])
    assert corpus_hash(["a", "b"]) != corpus_hash(["a", "c"])


def test_iter_midi_paths_requires_an_existing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="never downloads"):
        list(iter_midi_paths(tmp_path / "nope"))


def test_empty_directory_yields_nothing(tmp_path):
    assert list(iter_dataset_files(tmp_path)) == []
