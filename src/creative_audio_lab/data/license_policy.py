"""License classification and use-policy checks for dataset entries.

A small, explicit table of common data licenses mapped to what they allow.
This is engineering guardrail code, not legal advice — anything not in the
table is treated as *unknown* and surfaces as a violation until a human has
actually verified the terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .dataset_manifest import DatasetEntry


@dataclass(frozen=True)
class LicenseRule:
    """What a known license allows, for the purposes of dataset intake."""

    commercial_use: bool
    attribution_required: bool


# Canonical license id -> rule. Ids are the normalized form produced by
# normalize_license (lowercase, hyphenated, version numbers stripped).
KNOWN_LICENSES: Dict[str, LicenseRule] = {
    "cc0": LicenseRule(commercial_use=True, attribution_required=False),
    "public-domain": LicenseRule(commercial_use=True, attribution_required=False),
    "cc-by": LicenseRule(commercial_use=True, attribution_required=True),
    "cc-by-sa": LicenseRule(commercial_use=True, attribution_required=True),
    "cc-by-nc": LicenseRule(commercial_use=False, attribution_required=True),
    "cc-by-nc-sa": LicenseRule(commercial_use=False, attribution_required=True),
    "cc-by-nc-nd": LicenseRule(commercial_use=False, attribution_required=True),
    "mit": LicenseRule(commercial_use=True, attribution_required=True),
    "bsd": LicenseRule(commercial_use=True, attribution_required=True),
    "apache": LicenseRule(commercial_use=True, attribution_required=True),
    "research-only": LicenseRule(commercial_use=False, attribution_required=False),
}


def normalize_license(raw: str) -> str:
    """Normalize a license string to a canonical lookup id.

    ``"CC BY-NC-SA 4.0"`` → ``"cc-by-nc-sa"``, ``"Apache 2.0"`` → ``"apache"``,
    ``"  MIT License "`` → ``"mit"``. Unrecognized strings normalize but won't
    match :data:`KNOWN_LICENSES`.
    """
    text = raw.strip().lower()
    text = re.sub(r"\b\d+(\.\d+)*\b", "", text)  # strip version numbers
    text = text.replace("license", "").replace("creative commons", "cc")
    text = re.sub(r"[\s_/]+", "-", text.strip())
    return text.strip("-")


@dataclass(frozen=True)
class LicenseAssessment:
    """The outcome of looking a license string up against :data:`KNOWN_LICENSES`."""

    raw: str
    normalized: str
    known: bool
    commercial_use: Optional[bool]
    attribution_required: Optional[bool]


def assess_license(raw: str) -> LicenseAssessment:
    """Classify a license string; unknown licenses come back with ``None`` allowances."""
    normalized = normalize_license(raw) if raw else ""
    rule = KNOWN_LICENSES.get(normalized)
    return LicenseAssessment(
        raw=raw,
        normalized=normalized,
        known=rule is not None,
        commercial_use=rule.commercial_use if rule else None,
        attribution_required=rule.attribution_required if rule else None,
    )


def check_entry_license(entry: DatasetEntry, *, intended_use: str = "research") -> List[str]:
    """Return human-readable policy violations for using ``entry`` as ``intended_use``.

    Parameters
    ----------
    entry:
        The dataset entry to check.
    intended_use:
        Either ``"research"`` or ``"commercial"``.

    Returns
    -------
    list of str
        Empty when no violation was detected. Unknown always errs on the
        side of a violation for commercial use.
    """
    if intended_use not in ("research", "commercial"):
        raise ValueError(f"intended_use must be 'research' or 'commercial', got {intended_use!r}")

    violations: List[str] = []
    if not entry.license:
        violations.append(f"{entry.name}: no license recorded — cannot approve any training use.")
        return violations

    assessment = assess_license(entry.license)
    # The manifest's explicit flag wins over the table; the table fills gaps.
    commercial_ok = entry.commercial_use_allowed
    if commercial_ok is None:
        commercial_ok = assessment.commercial_use

    if intended_use == "commercial":
        if commercial_ok is None:
            violations.append(
                f"{entry.name}: commercial use is unverified for license {entry.license!r} — "
                "verify the terms before any commercial training use."
            )
        elif not commercial_ok:
            violations.append(
                f"{entry.name}: license {entry.license!r} does not permit commercial use."
            )
    elif not assessment.known and entry.commercial_use_allowed is None:
        violations.append(
            f"{entry.name}: license {entry.license!r} is not a recognized license id — "
            "record the actual terms (and commercial_use_allowed) after reviewing the source."
        )
    return violations


__all__ = [
    "KNOWN_LICENSES",
    "LicenseRule",
    "LicenseAssessment",
    "normalize_license",
    "assess_license",
    "check_entry_license",
]
