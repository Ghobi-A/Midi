from pathlib import Path

import pytest

from creative_audio_lab.data.dataset_manifest import (
    RECOMMENDED_DATASETS,
    DatasetEntry,
    entry_from_dict,
    load_manifest,
)
from creative_audio_lab.data.license_policy import (
    assess_license,
    check_entry_license,
    normalize_license,
)
from creative_audio_lab.data.provenance import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    assert_training_ready,
    has_errors,
    validate_entry,
    validate_manifest,
)

EXAMPLE_MANIFEST = Path(__file__).resolve().parent.parent / "examples" / "dataset_manifest.example.json"


def _clean_entry(**overrides) -> DatasetEntry:
    fields = dict(
        name="clean-set",
        source_url="https://example.org/clean",
        license="CC0 1.0",
        commercial_use_allowed=True,
        attribution_required=False,
    )
    fields.update(overrides)
    return DatasetEntry(**fields)


# ---------------------------------------------------------------------------
# Provenance validation
# ---------------------------------------------------------------------------


def test_clean_entry_has_no_issues():
    assert validate_entry(_clean_entry()) == []


def test_missing_license_is_an_error():
    issues = validate_entry(_clean_entry(license=""))
    assert any(i.severity == SEVERITY_ERROR and i.field == "license" for i in issues)


def test_missing_source_url_is_an_error():
    issues = validate_entry(_clean_entry(source_url="  "))
    assert any(i.severity == SEVERITY_ERROR and i.field == "source_url" for i in issues)


def test_unknown_commercial_use_is_a_warning():
    issues = validate_entry(_clean_entry(commercial_use_allowed=None))
    assert any(i.severity == SEVERITY_WARNING and i.field == "commercial_use_allowed" for i in issues)
    assert not has_errors(issues)


def test_fan_archive_tag_is_an_error():
    issues = validate_entry(_clean_entry(tags=("piano", "Fan-Archive")))
    tag_issues = [i for i in issues if i.field == "tags"]
    assert tag_issues and tag_issues[0].severity == SEVERITY_ERROR
    assert "reference/evaluation" in tag_issues[0].message


def test_validate_manifest_reports_only_problem_entries():
    entries = [_clean_entry(), _clean_entry(name="bad-set", license="", source_url="")]
    report = validate_manifest(entries)
    assert set(report) == {"bad-set"}
    assert has_errors(report["bad-set"])


def test_assert_training_ready_raises_with_error_details():
    with pytest.raises(ValueError, match="not training-ready"):
        assert_training_ready(_clean_entry(license="", tags=("unclear-rights",)))
    assert_training_ready(_clean_entry())  # clean entry passes


# ---------------------------------------------------------------------------
# License policy
# ---------------------------------------------------------------------------


def test_normalize_license_strips_versions_and_case():
    assert normalize_license("CC BY-NC-SA 4.0") == "cc-by-nc-sa"
    assert normalize_license("Creative Commons BY 4.0") == "cc-by"
    assert normalize_license("MIT License") == "mit"
    assert normalize_license("Public Domain") == "public-domain"
    assert normalize_license("CC0 1.0") == "cc0"


def test_assess_license_known_and_unknown():
    nc = assess_license("CC BY-NC-SA 4.0")
    assert nc.known and nc.commercial_use is False and nc.attribution_required is True
    mystery = assess_license("found on a forum")
    assert not mystery.known
    assert mystery.commercial_use is None


def test_non_commercial_license_blocks_commercial_use():
    entry = _clean_entry(license="CC BY-NC 4.0", commercial_use_allowed=None)
    violations = check_entry_license(entry, intended_use="commercial")
    assert violations and "does not permit commercial use" in violations[0]


def test_unknown_license_is_unverified_for_commercial_use():
    entry = _clean_entry(license="verify at source", commercial_use_allowed=None)
    violations = check_entry_license(entry, intended_use="commercial")
    assert violations and "unverified" in violations[0]


def test_explicit_manifest_flag_overrides_license_table():
    # A human verified commercial terms even though the license id is unknown.
    entry = _clean_entry(license="custom vendor agreement", commercial_use_allowed=True)
    assert check_entry_license(entry, intended_use="commercial") == []


def test_missing_license_always_violates():
    violations = check_entry_license(_clean_entry(license=""), intended_use="research")
    assert violations and "no license recorded" in violations[0]


def test_invalid_intended_use_rejected():
    with pytest.raises(ValueError):
        check_entry_license(_clean_entry(), intended_use="world-domination")


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def test_example_manifest_loads_and_validates_as_designed():
    entries = load_manifest(EXAMPLE_MANIFEST)
    by_name = {entry.name: entry for entry in entries}
    assert by_name["my-cc0-piano-set"].commercial_use_allowed is True
    assert by_name["research-pop-arrangements"].tags == ("pop", "example")

    report = validate_manifest(entries)
    # The deliberately-bad fan-archive entry is flagged; the clean one is not.
    assert "old-forum-game-midi-pack" in report
    assert has_errors(report["old-forum-game-midi-pack"])
    assert "my-cc0-piano-set" not in report


def test_entry_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown dataset-manifest keys"):
        entry_from_dict({"name": "x", "licence_notes": "typo"})


def test_entry_from_dict_requires_name():
    with pytest.raises(ValueError, match="'name'"):
        entry_from_dict({"license": "CC0"})


def test_missing_manifest_file_raises_without_downloading():
    with pytest.raises(FileNotFoundError, match="never auto-downloaded"):
        load_manifest("/nonexistent/manifest.json")


def test_recommended_datasets_all_carry_provenance_fields():
    for entry in RECOMMENDED_DATASETS:
        assert entry.name
        assert entry.source_url.startswith("http")
        assert entry.license
