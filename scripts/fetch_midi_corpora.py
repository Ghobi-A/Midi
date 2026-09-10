#!/usr/bin/env python
"""Fetch real MIDI corpora into a gitignored local cache and write a training manifest.

The raw corpora are intentionally not committed to this repository. The script
retrieves them from their canonical distribution locations, extracts them under
``data/corpora/``, verifies what can be verified, and writes a manifest that the
existing real-data experiment runner can consume directly.

Examples
--------
    python scripts/fetch_midi_corpora.py maestro commu
    python scripts/fetch_midi_corpora.py all
    python scripts/run_real_data_experiment.py \
        --manifest data/corpora/manifest.json \
        --output-dir experiments/real-midi
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_ROOT = ROOT / "data" / "corpora"
DEFAULT_MANIFEST = DEFAULT_CORPUS_ROOT / "manifest.json"

_MAESTRO_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/"
    "maestro-v3.0.0-midi.zip"
)
_MAESTRO_SHA256 = "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c"
_COMMU_MIDI_URL = (
    "https://raw.githubusercontent.com/pozalabs/ComMU-code/master/dataset/commu_midi.tar"
)
_COMMU_META_URL = (
    "https://raw.githubusercontent.com/pozalabs/ComMU-code/master/dataset/commu_meta.csv"
)
_GIANTMIDI_DRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/1Stz3CAvMoplo79LR5I3onMWRelCugBYS"
)


def _manifest_entry(
    *,
    name: str,
    source_url: str,
    license_id: str,
    commercial_use_allowed: bool,
    attribution_required: bool,
    local_path: Path,
    description: str,
    notes: str,
    tags: Iterable[str],
) -> Dict[str, object]:
    return {
        "name": name,
        "source_url": source_url,
        "license": license_id,
        "local_path": local_path.relative_to(ROOT).as_posix(),
        "commercial_use_allowed": commercial_use_allowed,
        "attribution_required": attribution_required,
        "description": description,
        "notes": notes,
        "tags": list(tags),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, expected_sha256: Optional[str] = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_sha256 is None or _sha256(destination) == expected_sha256:
            print(f"reuse: {destination}")
            return destination
        print(f"checksum mismatch in cached file; downloading again: {destination}")
        destination.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": "creative-audio-lab/0.1"})
    partial = destination.with_name(destination.name + ".part")
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    partial.replace(destination)

    if expected_sha256 is not None:
        observed = _sha256(destination)
        if observed != expected_sha256:
            destination.unlink(missing_ok=True)
            raise ValueError(
                f"SHA256 mismatch for {destination.name}: expected {expected_sha256}, got {observed}"
            )
    return destination


def _safe_member_path(root: Path, member_name: str) -> Path:
    candidate = (root / member_name).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"Archive member escapes target directory: {member_name}")
    return candidate


def _extract_zip(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            _safe_member_path(target, member.filename)
        zf.extractall(target)


def _extract_tar(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        for member in members:
            _safe_member_path(target, member.name)
            if member.issym() or member.islnk():
                raise ValueError(f"Refusing archive link member: {member.name}")
        tf.extractall(target, members=members)


def _extract_nested_archives(target: Path) -> None:
    """Extract archives found in a downloaded folder when no MIDI is visible yet."""
    if _midi_count(target):
        return
    archives = sorted(
        p
        for p in target.rglob("*")
        if p.is_file()
        and (
            p.suffix.lower() == ".zip"
            or p.suffix.lower() == ".tar"
            or p.name.lower().endswith((".tar.gz", ".tgz"))
        )
    )
    for archive in archives:
        if archive.suffix.lower() == ".zip":
            _extract_zip(archive, archive.parent)
        else:
            _extract_tar(archive, archive.parent)


def _midi_count(root: Path) -> int:
    return sum(1 for pattern in ("*.mid", "*.midi") for _ in root.rglob(pattern))


def fetch_maestro(corpus_root: Path, *, force: bool, keep_archives: bool) -> Dict[str, object]:
    target = corpus_root / "maestro"
    archive = corpus_root / "_archives" / "maestro-v3.0.0-midi.zip"
    if force and target.exists():
        shutil.rmtree(target)
    if not target.exists() or _midi_count(target) == 0:
        _download(_MAESTRO_URL, archive, expected_sha256=_MAESTRO_SHA256)
        _extract_zip(archive, target)
    count = _midi_count(target)
    if count < 1200:
        raise RuntimeError(f"MAESTRO extraction looks incomplete: found only {count} MIDI files")
    if not keep_archives:
        archive.unlink(missing_ok=True)
    print(f"maestro: {count} MIDI files")
    return _manifest_entry(
        name="MAESTRO",
        source_url="https://magenta.tensorflow.org/datasets/maestro",
        license_id="CC BY-NC-SA 4.0",
        commercial_use_allowed=False,
        attribution_required=True,
        local_path=target,
        description="MAESTRO v3 MIDI-only: expressive solo-piano performances with composition-level metadata.",
        notes="Fetched from the official v3.0.0 MIDI-only archive; archive SHA256 is verified before extraction.",
        tags=("piano", "performance", "real-corpus"),
    )


def fetch_commu(corpus_root: Path, *, force: bool, keep_archives: bool) -> Dict[str, object]:
    target = corpus_root / "commu"
    archive = corpus_root / "_archives" / "commu_midi.tar"
    metadata = target / "commu_meta.csv"
    if force and target.exists():
        shutil.rmtree(target)
    if not target.exists() or _midi_count(target) == 0:
        _download(_COMMU_MIDI_URL, archive)
        _extract_tar(archive, target)
    _download(_COMMU_META_URL, metadata)
    count = _midi_count(target)
    if count < 10000:
        raise RuntimeError(f"ComMU extraction looks incomplete: found only {count} MIDI files")
    if not keep_archives:
        archive.unlink(missing_ok=True)
    print(f"commu: {count} MIDI files")
    return _manifest_entry(
        name="ComMU",
        source_url="https://github.com/pozalabs/ComMU-code",
        license_id="CC BY-NC-SA 4.0",
        commercial_use_allowed=False,
        attribution_required=True,
        local_path=target,
        description="11k+ short MIDI compositions with controllable metadata written by professional composers.",
        notes="Fetched from the official ComMU repository; commu_meta.csv is retained beside the raw MIDI tree.",
        tags=("conditioned-generation", "metadata", "real-corpus"),
    )


def fetch_giantmidi(corpus_root: Path, *, force: bool, keep_archives: bool) -> Dict[str, object]:
    del keep_archives  # gdown owns the downloaded layout; archives are intentionally retained.
    target = corpus_root / "giantmidi"
    if force and target.exists():
        shutil.rmtree(target)
    if not target.exists() or _midi_count(target) == 0:
        try:
            import gdown  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "GiantMIDI uses the project's official Google Drive folder. Install the data extra first: "
                "pip install -e '.[data]'"
            ) from error
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gdown",
                "--folder",
                _GIANTMIDI_DRIVE_FOLDER,
                "-O",
                str(target),
                "--remaining-ok",
            ],
            check=True,
        )
        _extract_nested_archives(target)
    count = _midi_count(target)
    if count < 7000:
        raise RuntimeError(
            f"GiantMIDI download looks incomplete: found {count} MIDI files; expected at least the curated subset"
        )
    print(f"giantmidi: {count} MIDI files")
    return _manifest_entry(
        name="GiantMIDI-Piano",
        source_url="https://github.com/bytedance/GiantMIDI-Piano",
        license_id="CC BY 4.0",
        commercial_use_allowed=True,
        attribution_required=True,
        local_path=target,
        description="Large classical-piano corpus of automatically transcribed live performances.",
        notes="Fetched from the Google Drive folder linked by the official archived GiantMIDI-Piano repository.",
        tags=("piano", "classical", "transcription", "real-corpus"),
    )


FETCHERS = {
    "maestro": fetch_maestro,
    "commu": fetch_commu,
    "giantmidi": fetch_giantmidi,
}


def _write_manifest(path: Path, entries: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["maestro", "commu"],
        choices=["maestro", "commu", "giantmidi", "all"],
        help="Corpora to fetch. Default: maestro commu. 'all' includes GiantMIDI.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true", help="Delete selected extracted corpus before fetching")
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args(argv)

    selected = list(args.datasets)
    if "all" in selected:
        selected = list(FETCHERS)
    selected = list(dict.fromkeys(selected))

    entries = []
    for key in selected:
        entries.append(
            FETCHERS[key](args.root, force=args.force, keep_archives=args.keep_archives)
        )
    _write_manifest(args.manifest, entries)

    total = sum(_midi_count(Path(entry["local_path"])) for entry in entries)
    print(f"ready: {total} MIDI files across {len(entries)} corpora")
    print(
        "next: python scripts/run_real_data_experiment.py "
        f"--manifest {args.manifest.relative_to(ROOT)} --output-dir experiments/real-midi"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
