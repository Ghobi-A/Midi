from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_midi_corpora.py"
_SPEC = importlib.util.spec_from_file_location("fetch_midi_corpora", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
fetch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch)


def test_midi_count_recurses_and_accepts_both_extensions(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.mid").write_bytes(b"midi")
    (tmp_path / "two.midi").write_bytes(b"midi")
    (tmp_path / "not-midi.txt").write_text("x")

    assert fetch._midi_count(tmp_path) == 2


def test_safe_member_path_rejects_archive_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes target directory"):
        fetch._safe_member_path(tmp_path, "../escape.mid")


def test_extract_zip_rejects_archive_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.mid", b"not really midi")

    with pytest.raises(ValueError, match="escapes target directory"):
        fetch._extract_zip(archive, tmp_path / "out")

    assert not (tmp_path / "escape.mid").exists()


def test_manifest_entry_uses_repo_relative_local_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    local = tmp_path / "data" / "corpora" / "demo"

    entry = fetch._manifest_entry(
        name="Demo",
        source_url="https://example.test/dataset",
        license_id="CC BY 4.0",
        commercial_use_allowed=True,
        attribution_required=True,
        local_path=local,
        description="demo",
        notes="demo",
        tags=("test",),
    )

    assert entry["local_path"] == "data/corpora/demo"
    assert entry["tags"] == ["test"]


def test_write_manifest_round_trips_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    entries = [{"name": "A", "local_path": "data/corpora/a"}]

    fetch._write_manifest(path, entries)

    assert json.loads(path.read_text()) == entries
