import json
import shutil
from pathlib import Path

from creative_audio_lab.data.corpus_pipeline import CorpusConfig, build_corpus
from fixture_corpus import clean_only_manifest


def test_identical_content_cannot_cross_composition_splits(tmp_path):
    manifest = clean_only_manifest(tmp_path)
    root = Path(json.loads(manifest.read_text())[0]["local_path"])
    source = next(root.rglob("*.mid"))
    shutil.copyfile(source, root / "exact_copy.mid")
    bundle = build_corpus(manifest, CorpusConfig(min_notes=8))
    hashes = [{piece.sha256 for piece in bundle.splits[name]}
              for name in ("train", "val", "test")]
    assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
    assert any(reason.startswith("duplicate_content:") for _, reason in bundle.rejections)
