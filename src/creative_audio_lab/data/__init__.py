"""Dataset scaffolding for the future ML roadmap (see docs/ML_ROADMAP.md).

Nothing in this package downloads data automatically. It defines a
manifest schema (:mod:`.dataset_manifest`), provenance and license-policy
validation (:mod:`.provenance`, :mod:`.license_policy`), a loader for MIDI
files the user has already placed on disk (:mod:`.midi_dataset_loader`),
and preprocessing primitives (:mod:`.preprocess_midi`) that a future
training pipeline would build on.
"""

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
