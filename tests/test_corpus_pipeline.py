"""Tests for the manifest → rights check → melody → splits → tokens pipeline."""

import json

import pytest

from creative_audio_lab.data import (
    CorpusConfig,
    build_corpus,
    select_melody_track,
    split_compositions,
)
from creative_audio_lab.data.corpus_pipeline import normalise_melody
from creative_audio_lab.midi_parser import MidiFileInfo, TrackInfo
from creative_audio_lab.music_theory import Note
from fixture_corpus import build_fixture_corpus, melody_notes


@pytest.fixture(scope="module")
def fixture_corpus(tmp_path_factory):
    return build_fixture_corpus(tmp_path_factory.mktemp("corpus"))


@pytest.fixture(scope="module")
def bundle(fixture_corpus):
    return build_corpus(
        fixture_corpus["manifest_path"], CorpusConfig(allow_reference_only=True)
    )


# ---------------------------------------------------------------------------
# Rights gate
# ---------------------------------------------------------------------------


def test_untrainable_entry_blocks_the_build_by_default(fixture_corpus):
    with pytest.raises(ValueError, match="cannot be used for training"):
        build_corpus(fixture_corpus["manifest_path"])


def test_reference_only_material_is_confined_to_the_test_split(bundle):
    flagged = [
        piece for pieces in bundle.splits.values() for piece in pieces if piece.reference_only
    ]
    assert flagged, "the fixture includes a deliberately un-trainable dataset"
    assert all(piece.dataset == "fixture-fan-archive" for piece in flagged)
    test_ids = set(bundle.composition_ids("test"))
    assert all(piece.composition_id in test_ids for piece in flagged)
    for split in ("train", "val"):
        assert not any(piece.reference_only for piece in bundle.splits[split])


def test_datasets_record_licence_and_reference_only_reason(bundle):
    by_name = {entry["name"]: entry for entry in bundle.datasets}
    assert by_name["fixture-cc0"]["license"] == "CC0 1.0"
    assert by_name["fixture-cc0"]["reference_only"] is False
    assert by_name["fixture-fan-archive"]["reference_only"] is True
    assert "not training-ready" in by_name["fixture-fan-archive"]["reference_only_reason"]


def test_provenance_report_flags_the_bad_entry(bundle):
    assert "fixture-fan-archive" in bundle.provenance_report


def test_commercial_use_is_checked_against_the_licence(tmp_path):
    built = build_fixture_corpus(tmp_path)
    entries = json.loads(built["manifest_path"].read_text(encoding="utf-8"))
    entries = [e for e in entries if e["name"] == "fixture-cc0"]
    entries[0]["license"] = "CC BY-NC 4.0"
    entries[0]["commercial_use_allowed"] = False
    manifest = tmp_path / "nc.json"
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(ValueError, match="commercial"):
        build_corpus(manifest, CorpusConfig(intended_use="commercial"))


# ---------------------------------------------------------------------------
# Intake filters
# ---------------------------------------------------------------------------


def test_non_four_four_pieces_are_rejected_with_a_reason(bundle):
    reasons = dict(bundle.rejections)
    waltz = [cid for cid in reasons if cid.endswith("waltz.mid")]
    assert waltz and reasons[waltz[0]].startswith("time_signature")


def test_chordal_and_drum_files_are_rejected(bundle):
    reasons = dict(bundle.rejections)
    for name in ("chords.mid", "drums.mid", "tiny.mid"):
        rejected = [cid for cid in reasons if cid.endswith(name)]
        assert rejected, f"{name} should have been rejected"


def test_all_meters_can_be_accepted_when_the_filter_is_disabled(fixture_corpus):
    permissive = build_corpus(
        fixture_corpus["manifest_path"],
        CorpusConfig(allow_reference_only=True, time_signatures=None),
    )
    accepted = [
        piece
        for pieces in permissive.splits.values()
        for piece in pieces
        if piece.composition_id.endswith("waltz.mid")
    ]
    assert accepted and accepted[0].time_signature == "3/4"


def test_duplicate_filenames_in_different_directories_both_appear(bundle):
    ids = [cid for pieces in bundle.splits.values() for cid in [p.composition_id for p in pieces]]
    songs = [cid for cid in ids if cid.endswith("song.mid")]
    assert sorted(songs) == ["fixture-cc0/set_a/song.mid", "fixture-cc0/set_b/song.mid"]


def test_select_melody_track_prefers_the_long_monophonic_track():
    melody = melody_notes("C", 0)
    chords = [Note(start=float(i), pitch=p, duration=1.0) for i in range(20) for p in (60, 64, 67)]
    info = MidiFileInfo(
        ticks_per_beat=480,
        tracks=(
            TrackInfo(index=0, name="melody", note_count=len(melody)),
            TrackInfo(index=1, name="chords", note_count=len(chords)),
            TrackInfo(index=2, name="drums", note_count=8, is_drum=True),
        ),
    )
    tracks = {"melody": melody, "chords": chords, "drums": melody}
    name, notes = select_melody_track(tracks, info)
    assert name == "melody" and notes == melody


def test_select_melody_track_returns_none_when_nothing_qualifies():
    info = MidiFileInfo(ticks_per_beat=480, tracks=(TrackInfo(index=0, name="drums", is_drum=True),))
    assert select_melody_track({"drums": melody_notes("C", 0)}, info) is None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["C", "G", "F", "A"])
def test_normalise_melody_transposes_every_key_to_the_same_tonic(key):
    # The estimated key of a short fragment need not equal the scale it was
    # written from; what must hold is that normalisation maps every piece to
    # the configured tonic, so pitch statistics pool instead of splitting by key.
    from creative_audio_lab.music_theory import estimate_key

    normalised, original_key = normalise_melody(melody_notes(key, 0), CorpusConfig())
    assert original_key is not None
    assert estimate_key(normalised).key == "C"


def test_normalise_melody_can_keep_the_original_key():
    notes = melody_notes("G", 0)
    normalised, _ = normalise_melody(notes, CorpusConfig(transpose_to_tonic=None))
    assert [n.pitch for n in normalised] == [n.pitch for n in notes]


def test_normalise_melody_quantizes_onsets():
    notes = [Note(start=0.03, pitch=60, duration=1.0), Note(start=1.19, pitch=62, duration=1.0)]
    normalised, _ = normalise_melody(notes, CorpusConfig(transpose_to_tonic=None, grid=0.25))
    assert [n.start for n in normalised] == [0.0, 1.25]


# ---------------------------------------------------------------------------
# Composition-level splitting
# ---------------------------------------------------------------------------


def test_splits_are_disjoint_and_cover_everything():
    ids = [f"piece_{i}.mid" for i in range(40)]
    splits = split_compositions(ids)
    train, val, test = splits["train"], splits["val"], splits["test"]
    assert set(train) | set(val) | set(test) == set(ids)
    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    assert len(train) + len(val) + len(test) == len(ids)
    assert len(train) == 32


def test_split_is_deterministic_and_seed_dependent():
    ids = [f"piece_{i}.mid" for i in range(40)]
    assert split_compositions(ids, seed=1) == split_compositions(ids, seed=1)
    assert split_compositions(ids, seed=1) != split_compositions(ids, seed=2)
    # Filesystem order must not matter.
    assert split_compositions(ids, seed=1) == split_compositions(list(reversed(ids)), seed=1)


def test_small_corpora_still_get_a_val_and_test_split():
    splits = split_compositions([f"p{i}" for i in range(3)])
    assert all(len(splits[name]) == 1 for name in ("train", "val", "test"))


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        CorpusConfig(fractions=(0.9, 0.2, 0.1))


def test_bundle_splits_never_share_a_composition(bundle):
    seen = set()
    for name in ("train", "val", "test"):
        ids = set(bundle.composition_ids(name))
        assert not (ids & seen), f"{name} overlaps an earlier split"
        seen |= ids


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_sequences_are_note_token_triples(bundle):
    for sequence in bundle.sequences("train"):
        assert sequence and len(sequence) % 3 == 0
        assert sequence[0].startswith("NOTE_ON_")
        assert sequence[1].startswith("VELOCITY_")
        assert sequence[2].startswith("DURATION_")


def test_stats_report_counts_and_rejections(bundle):
    stats = bundle.stats()
    assert stats["accepted_pieces"] == sum(s["pieces"] for s in stats["splits"].values())
    assert stats["rejected_pieces"] == len(bundle.rejections)
    assert stats["rejection_reasons"]["time_signature"] >= 1
    assert len(stats["corpus_hash"]) == 64


def test_corpus_hash_is_stable_across_rebuilds(fixture_corpus):
    first = build_corpus(fixture_corpus["manifest_path"], CorpusConfig(allow_reference_only=True))
    second = build_corpus(fixture_corpus["manifest_path"], CorpusConfig(allow_reference_only=True))
    assert first.corpus_hash == second.corpus_hash


def test_empty_manifest_directory_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([{
        "name": "empty-set", "source_url": "https://example.org/e", "license": "CC0 1.0",
        "local_path": str(empty), "commercial_use_allowed": True,
    }]), encoding="utf-8")
    with pytest.raises(ValueError, match="No usable pieces"):
        build_corpus(manifest)
