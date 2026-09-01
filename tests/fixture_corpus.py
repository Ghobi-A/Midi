"""Build a small synthetic MIDI corpus on disk for pipeline tests.

Real training data cannot be committed to this repository (rights, size),
so the end-to-end pipeline is exercised on files written here through the
project's own MIDI exporter. The corpus deliberately includes the awkward
cases intake has to handle: identically-named files in different
subdirectories, a 3/4 piece, a drums-only file, duplicate track names, a
chordal (non-melodic) file, and a piece too short to use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import mido

from creative_audio_lab.export.midi_export import export_notes_to_bytes
from creative_audio_lab.music_theory import Note

#: Diatonic scales used to write melodies in a few different keys.
_SCALES = {
    "C": [60, 62, 64, 65, 67, 69, 71, 72],
    "G": [67, 69, 71, 72, 74, 76, 78, 79],
    "F": [65, 67, 69, 70, 72, 74, 76, 77],
    "A": [69, 71, 72, 74, 76, 77, 79, 81],
}

#: Melodic shapes (indices into a scale), so pieces differ but share structure.
_SHAPES = (
    (0, 1, 2, 3, 4, 3, 2, 1),
    (0, 2, 4, 2, 0, 2, 4, 5),
    (4, 3, 2, 1, 0, 1, 2, 3),
    (0, 4, 3, 4, 5, 4, 3, 2),
    (7, 6, 5, 4, 3, 2, 1, 0),
)

_DURATIONS = (1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0)


def melody_notes(key: str, shape_index: int, repeats: int = 4) -> List[Note]:
    """A monophonic line long enough to pass the default min_notes filter."""
    scale = _SCALES[key]
    shape = _SHAPES[shape_index % len(_SHAPES)]
    notes: List[Note] = []
    start = 0.0
    for repeat in range(repeats):
        for step, degree in enumerate(shape):
            duration = _DURATIONS[(step + repeat) % len(_DURATIONS)]
            notes.append(
                Note(start=start, pitch=scale[degree], duration=duration, velocity=88)
            )
            start += duration
    return notes


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _write_melody(path: Path, key: str, shape_index: int, time_signature=(4, 4)) -> Path:
    data = export_notes_to_bytes(
        melody_notes(key, shape_index), name="melody", program=0, bpm=110.0
    )
    if time_signature != (4, 4):
        mid = mido.MidiFile(file=__import__("io").BytesIO(data))
        numerator, denominator = time_signature
        mid.tracks[0].insert(
            0, mido.MetaMessage("time_signature", numerator=numerator, denominator=denominator, time=0)
        )
        buffer = __import__("io").BytesIO()
        mid.save(file=buffer)
        data = buffer.getvalue()
    return _write(path, data)


def _write_chords(path: Path) -> Path:
    notes = [
        Note(start=float(bar * 2), pitch=pitch, duration=2.0, velocity=80)
        for bar in range(12)
        for pitch in (60 + bar % 3, 64 + bar % 3, 67 + bar % 3)
    ]
    return _write(path, export_notes_to_bytes(notes, name="chords", program=0, bpm=110.0))


def _write_drums(path: Path) -> Path:
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="drums", time=0))
    for _ in range(32):
        track.append(mido.Message("note_on", note=36, velocity=100, channel=9, time=0))
        track.append(mido.Message("note_off", note=36, velocity=0, channel=9, time=240))
    mid.tracks.append(track)
    buffer = __import__("io").BytesIO()
    mid.save(file=buffer)
    return _write(path, buffer.getvalue())


def _write_duplicate_track_names(path: Path) -> Path:
    """One file whose two tracks share a name — the old parser lost one."""
    mid = mido.MidiFile(ticks_per_beat=480)
    for key, shape in (("C", 0), ("G", 1)):
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name="part", time=0))
        previous_tick = 0
        for note in melody_notes(key, shape):
            start_tick = round(note.start * 480)
            track.append(
                mido.Message("note_on", note=note.pitch, velocity=note.velocity,
                             time=start_tick - previous_tick)
            )
            track.append(
                mido.Message("note_off", note=note.pitch, velocity=0,
                             time=round(note.duration * 480))
            )
            previous_tick = start_tick + round(note.duration * 480)
        mid.tracks.append(track)
    buffer = __import__("io").BytesIO()
    mid.save(file=buffer)
    return _write(path, buffer.getvalue())


def _write_too_short(path: Path) -> Path:
    notes = [Note(start=float(i), pitch=60 + i, duration=1.0, velocity=80) for i in range(4)]
    return _write(path, export_notes_to_bytes(notes, name="melody", program=0, bpm=110.0))


def build_fixture_corpus(root: Path) -> Dict[str, object]:
    """Write the fixture corpus and manifest under ``root``.

    Returns a dict with the manifest path and the counts a test can assert
    on. Two datasets are written: ``fixture-cc0`` (clean, trainable) and
    ``fixture-fan-archive`` (deliberately un-trainable provenance).
    """
    root = Path(root)
    clean = root / "cc0_melodies"
    flagged = root / "forum_pack"

    keys = list(_SCALES)
    usable: List[Path] = []
    # Nested directories, including the same filename twice.
    for index in range(8):
        key = keys[index % len(keys)]
        subdir = "set_a" if index % 2 == 0 else "set_b"
        usable.append(_write_melody(clean / subdir / "song.mid" if index < 2
                                    else clean / subdir / f"piece_{index}.mid", key, index))
    # Awkward cases that intake must reject or handle.
    _write_melody(clean / "meters" / "waltz.mid", "C", 2, time_signature=(3, 4))
    _write_chords(clean / "other" / "chords.mid")
    _write_drums(clean / "other" / "drums.mid")
    _write_too_short(clean / "other" / "tiny.mid")
    _write_duplicate_track_names(clean / "other" / "two_parts.mid")

    for index in range(2):
        _write_melody(flagged / f"ripped_{index}.mid", keys[index], index + 1)

    manifest = [
        {
            "name": "fixture-cc0",
            "source_url": "https://example.org/fixture-cc0",
            "license": "CC0 1.0",
            "local_path": str(clean),
            "commercial_use_allowed": True,
            "attribution_required": False,
            "description": "Synthetic fixture corpus written by the test suite.",
            "tags": ["fixture"],
        },
        {
            "name": "fixture-fan-archive",
            "source_url": "",
            "license": "",
            "local_path": str(flagged),
            "commercial_use_allowed": None,
            "attribution_required": None,
            "description": "Deliberately un-trainable entry: no source, no license, fan archive.",
            "tags": ["fan-archive", "fixture"],
        },
    ]
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "manifest_path": manifest_path,
        "clean_dir": clean,
        "flagged_dir": flagged,
        "usable_melodies": len(usable) + 1,  # + the duplicate-track-name file
    }


def clean_only_manifest(root: Path) -> Path:
    """Write a manifest containing only the rights-clean fixture dataset."""
    built = build_fixture_corpus(root)
    manifest_path = Path(built["manifest_path"])
    entries: Sequence[dict] = json.loads(manifest_path.read_text(encoding="utf-8"))
    clean_path = root / "manifest_clean.json"
    clean_path.write_text(
        json.dumps([e for e in entries if e["name"] == "fixture-cc0"], indent=2),
        encoding="utf-8",
    )
    return clean_path


__all__ = ["build_fixture_corpus", "clean_only_manifest", "melody_notes"]
