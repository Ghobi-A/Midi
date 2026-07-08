"""Provenance validation for dataset manifest entries.

Every dataset a future training pipeline touches should pass through
:func:`validate_entry` first, so missing licenses, unverified commercial
terms, and fan-archive material get flagged *before* they contaminate a
training corpus — not discovered afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .dataset_manifest import DatasetEntry

# Tags that mark a dataset as private reference/evaluation material only —
# never training data — until its rights are actually cleared.
UNCLEAR_RIGHTS_TAGS = frozenset(
    {
        "fan-archive",
        "fan-made",
        "game-rip",
        "anime-rip",
        "unclear-rights",
        "unlicensed",
    }
)

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


@dataclass(frozen=True)
class ProvenanceIssue:
    """One problem found while validating a dataset entry's provenance."""

    severity: str  # SEVERITY_ERROR or SEVERITY_WARNING
    field: str
    message: str


def validate_entry(entry: DatasetEntry) -> List[ProvenanceIssue]:
    """Check one dataset entry's provenance record.

    Errors (must be fixed before training use):

    - missing license
    - missing source URL
    - tagged as fan archive / unclear rights

    Warnings (should be resolved, but don't block reference use):

    - ``commercial_use_allowed`` unknown (``None``)
    """
    issues: List[ProvenanceIssue] = []

    if not entry.license.strip():
        issues.append(
            ProvenanceIssue(
                severity=SEVERITY_ERROR,
                field="license",
                message=f"{entry.name}: license is missing. Record the license (or 'unknown' plus "
                "notes on what still needs verifying) before this dataset can be considered.",
            )
        )
    if not entry.source_url.strip():
        issues.append(
            ProvenanceIssue(
                severity=SEVERITY_ERROR,
                field="source_url",
                message=f"{entry.name}: source URL is missing. Provenance requires knowing where the "
                "data actually came from — 'found on a forum' is not a source.",
            )
        )
    if entry.commercial_use_allowed is None:
        issues.append(
            ProvenanceIssue(
                severity=SEVERITY_WARNING,
                field="commercial_use_allowed",
                message=f"{entry.name}: commercial_use_allowed is unknown. Verify the license terms and "
                "set it explicitly before relying on this dataset.",
            )
        )
    flagged = UNCLEAR_RIGHTS_TAGS.intersection(tag.strip().lower() for tag in entry.tags)
    if flagged:
        issues.append(
            ProvenanceIssue(
                severity=SEVERITY_ERROR,
                field="tags",
                message=f"{entry.name}: tagged {sorted(flagged)} — treat as private reference/evaluation "
                "material only. Fan archives and unclear-rights collections must not be used for "
                "training unless properly licensed.",
            )
        )
    return issues


def validate_manifest(entries: Iterable[DatasetEntry]) -> Dict[str, List[ProvenanceIssue]]:
    """Validate a whole manifest; returns ``{dataset name: issues}`` for entries with issues."""
    report: Dict[str, List[ProvenanceIssue]] = {}
    for entry in entries:
        issues = validate_entry(entry)
        if issues:
            report[entry.name] = issues
    return report


def has_errors(issues: Sequence[ProvenanceIssue]) -> bool:
    """Return ``True`` if any issue in ``issues`` is an error (vs. a warning)."""
    return any(issue.severity == SEVERITY_ERROR for issue in issues)


def assert_training_ready(entry: DatasetEntry) -> None:
    """Raise ``ValueError`` (listing every error) if ``entry`` is not fit for training use."""
    errors = [issue for issue in validate_entry(entry) if issue.severity == SEVERITY_ERROR]
    if errors:
        details = "\n".join(f"- {issue.message}" for issue in errors)
        raise ValueError(f"Dataset {entry.name!r} is not training-ready:\n{details}")


__all__ = [
    "UNCLEAR_RIGHTS_TAGS",
    "SEVERITY_ERROR",
    "SEVERITY_WARNING",
    "ProvenanceIssue",
    "validate_entry",
    "validate_manifest",
    "has_errors",
    "assert_training_ready",
]
