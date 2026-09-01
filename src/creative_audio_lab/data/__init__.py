"""Dataset scaffolding for the future ML roadmap (see docs/ML_ROADMAP.md).

Nothing in this package downloads data automatically. It defines a
manifest schema (:mod:`.dataset_manifest`), provenance and license-policy
validation (:mod:`.provenance`, :mod:`.license_policy`), a loader for MIDI
files the user has already placed on disk (:mod:`.midi_dataset_loader`),
preprocessing primitives (:mod:`.preprocess_midi`), and the end-to-end
manifest-to-tokenised-splits pipeline (:mod:`.corpus_pipeline`) that turns
a rights-checked local corpus into leakage-safe training data.
"""

from .corpus_pipeline import (
    SPLIT_NAMES,
    CorpusBundle,
    CorpusConfig,
    TokenisedPiece,
    build_corpus,
    select_melody_track,
    split_compositions,
)
from .dataset_manifest import RECOMMENDED_DATASETS, DatasetEntry, entry_from_dict, load_manifest
from .license_policy import LicenseAssessment, assess_license, check_entry_license, normalize_license
from .provenance import (
    ProvenanceIssue,
    assert_training_ready,
    has_errors,
    validate_entry,
    validate_manifest,
)

__all__ = [
    "SPLIT_NAMES",
    "CorpusBundle",
    "CorpusConfig",
    "TokenisedPiece",
    "build_corpus",
    "select_melody_track",
    "split_compositions",
    "DatasetEntry",
    "RECOMMENDED_DATASETS",
    "entry_from_dict",
    "load_manifest",
    "LicenseAssessment",
    "assess_license",
    "check_entry_license",
    "normalize_license",
    "ProvenanceIssue",
    "assert_training_ready",
    "has_errors",
    "validate_entry",
    "validate_manifest",
]
