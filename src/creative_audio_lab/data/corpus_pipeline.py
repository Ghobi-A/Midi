"""Manifest → rights check → MIDI → melody → splits → token sequences.

This is the missing end-to-end path between the dataset scaffolding and
the training code. Given a dataset manifest (see
:mod:`creative_audio_lab.data.dataset_manifest`) it:

1. loads the manifest and refuses any entry that is not training-ready
   (:func:`~creative_audio_lab.data.provenance.assert_training_ready`) or
   that violates the license policy for the intended use;
2. walks each entry's ``local_path`` (never the network), identifying every
   piece by its path relative to the dataset root and hashing its contents;
3. parses each file, filters it on quality criteria (time signature, note
   count, monophony), and selects a melody track;
4. quantises, optionally transposes to a common tonic using
   :func:`~creative_audio_lab.music_theory.estimate_key`;
5. splits at the **composition** level — never within a piece — so no
   melody has fragments in both train and test;
6. tokenises each split with the internal REMI-style tokenizer.

Every rejected piece is recorded with a reason, so a corpus report says
what was thrown away and why instead of silently shrinking.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..midi_parser import (
    MidiFileInfo,
    parse_midi_file_with_info,
    polyphony_ratio,
)
from ..music_theory import Note, estimate_key, note_name_to_pitch_class
from ..tokenization import TokenizerConfig
from .dataset_manifest import DatasetEntry, load_manifest
from .license_policy import check_entry_license
from .midi_dataset_loader import corpus_hash, iter_dataset_files
from .preprocess_midi import quantize_notes, transpose_notes
from .provenance import assert_training_ready, validate_manifest

#: Split names, in order.
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class CorpusConfig:
    """Intake and split policy for :func:`build_corpus`.

    Attributes
    ----------
    intended_use:
        ``"research"`` or ``"commercial"`` — checked against each entry's
        license before any file is read.
    time_signatures:
        Allowed time signatures. ``("4/4",)`` by default because the
        generators and the default tokenizer grid assume 4/4; pass ``None``
        to accept every meter (the tokenizer then follows each piece's own
        signature).
    min_notes:
        Pieces whose melody track has fewer notes are rejected.
    max_polyphony_ratio:
        Maximum fraction of notes sharing an onset for a track to count as
        a melody line.
    grid:
        Onset quantization grid in beats.
    transpose_to_tonic:
        Pitch-class name every piece is transposed to (``"C"``), or ``None``
        to keep original keys.
    fractions:
        Train/val/test proportions; must sum to 1.
    split_seed:
        Seed for the deterministic composition-level split.
    allow_reference_only:
        When ``True``, entries that fail the training-readiness check are
        still ingested but may only appear in the **test** split, and this
        is recorded in the corpus report.
    """

    intended_use: str = "research"
    time_signatures: Optional[Tuple[str, ...]] = ("4/4",)
    min_notes: int = 16
    max_polyphony_ratio: float = 0.1
    grid: float = 0.25
    transpose_to_tonic: Optional[str] = "C"
    fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 0
    allow_reference_only: bool = False
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    def __post_init__(self) -> None:
        if abs(sum(self.fractions) - 1.0) > 1e-9:
            raise ValueError(f"fractions must sum to 1, got {self.fractions}")
        if any(fraction < 0 for fraction in self.fractions):
            raise ValueError(f"fractions must be non-negative, got {self.fractions}")


@dataclass(frozen=True)
class TokenisedPiece:
    """One accepted composition: its identity, provenance, and token stream."""

    composition_id: str
    dataset: str
    sha256: str
    track: str
    note_count: int
    original_key: Optional[str]
    time_signature: str
    tokens: Tuple[str, ...]
    reference_only: bool = False


@dataclass(frozen=True)
class CorpusBundle:
    """The result of :func:`build_corpus`: splits, rejections, and provenance."""

    splits: Dict[str, List[TokenisedPiece]]
    rejections: List[Tuple[str, str]]
    datasets: List[Dict[str, object]]
    provenance_report: Dict[str, List[str]]
    config: CorpusConfig

    def sequences(self, split: str) -> List[List[str]]:
        """Token sequences for ``split``, ready for
        :func:`~creative_audio_lab.models.ngram_training.train_melody_model`."""
        return [list(piece.tokens) for piece in self.splits[split]]

    def composition_ids(self, split: str) -> List[str]:
        return [piece.composition_id for piece in self.splits[split]]

    @property
    def corpus_hash(self) -> str:
        """Stable hash over every accepted file's content hash."""
        return corpus_hash(
            [piece.sha256 for pieces in self.splits.values() for piece in pieces]
        )

    def stats(self) -> Dict[str, object]:
        """Counts, token totals, vocabulary sizes, and rejection reasons."""
        per_split = {}
        for name in SPLIT_NAMES:
            pieces = self.splits.get(name, [])
            tokens = [token for piece in pieces for token in piece.tokens]
            per_split[name] = {
                "pieces": len(pieces),
                "notes": sum(piece.note_count for piece in pieces),
                "tokens": len(tokens),
                "vocabulary": len(set(tokens)),
            }
        reasons: Dict[str, int] = {}
        for _, reason in self.rejections:
            reasons[reason.split(":")[0]] = reasons.get(reason.split(":")[0], 0) + 1
        return {
            "splits": per_split,
            "accepted_pieces": sum(len(p) for p in self.splits.values()),
            "rejected_pieces": len(self.rejections),
            "rejection_reasons": reasons,
            "corpus_hash": self.corpus_hash,
        }


# ---------------------------------------------------------------------------
# Melody selection and normalisation
# ---------------------------------------------------------------------------


def select_melody_track(
    tracks: Dict[str, List[Note]],
    info: MidiFileInfo,
    *,
    max_polyphony_ratio: float = 0.1,
    min_notes: int = 16,
) -> Optional[Tuple[str, List[Note]]]:
    """Pick the most plausible melody track, or ``None`` if none qualifies.

    Drum tracks are excluded, tracks above ``max_polyphony_ratio`` are
    treated as chordal accompaniment, and among the rest the one with the
    most notes wins (ties broken by name for determinism).
    """
    drum_names = {track.name for track in info.tracks if track.is_drum}
    candidates = [
        (name, notes)
        for name, notes in tracks.items()
        if name not in drum_names
        and len(notes) >= min_notes
        and polyphony_ratio(notes) <= max_polyphony_ratio
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (len(item[1]), item[0]))


def normalise_melody(
    notes: Sequence[Note], config: CorpusConfig
) -> Tuple[List[Note], Optional[str]]:
    """Quantise and (optionally) transpose ``notes`` to the configured tonic.

    Returns the normalised notes and the estimated original key name.
    """
    quantised = quantize_notes(notes, grid=config.grid)
    estimate = estimate_key(quantised)
    if config.transpose_to_tonic is None or estimate is None:
        return quantised, (estimate.key if estimate else None)
    shift = note_name_to_pitch_class(config.transpose_to_tonic) - estimate.pitch_class
    # Move by the smaller direction so pitches stay near their original register.
    if shift > 6:
        shift -= 12
    elif shift < -6:
        shift += 12
    return transpose_notes(quantised, shift), estimate.key


# ---------------------------------------------------------------------------
# Composition-level splitting
# ---------------------------------------------------------------------------


def split_compositions(
    composition_ids: Sequence[str],
    fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> Dict[str, List[str]]:
    """Assign whole compositions to train/val/test, deterministically.

    Ordering is by ``sha256(seed:id)``, so the split depends only on the
    seed and the ids — not on filesystem order — and the same corpus always
    produces the same split. No composition appears in two splits.
    """
    ordered = sorted(
        composition_ids,
        key=lambda cid: hashlib.sha256(f"{seed}:{cid}".encode("utf-8")).hexdigest(),
    )
    total = len(ordered)
    train_end = int(round(total * fractions[0]))
    val_end = train_end + int(round(total * fractions[1]))
    # With few pieces, rounding can starve val/test; keep at least one each
    # when there is enough material to do so.
    if total >= 3:
        train_end = min(max(train_end, 1), total - 2)
        val_end = min(max(val_end, train_end + 1), total - 1)
    return {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }


# ---------------------------------------------------------------------------
# Rights gate
# ---------------------------------------------------------------------------


def _entry_is_training_ready(entry: DatasetEntry, intended_use: str) -> Optional[str]:
    """Return a refusal reason for ``entry``, or ``None`` when it may be trained on."""
    try:
        assert_training_ready(entry)
    except ValueError as error:
        return str(error)
    violations = check_entry_license(entry, intended_use=intended_use)
    if violations:
        return "; ".join(violations)
    return None


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def build_corpus(
    manifest_path: Union[str, Path], config: Optional[CorpusConfig] = None
) -> CorpusBundle:
    """Build tokenised, composition-split token sequences from a dataset manifest.

    Raises
    ------
    ValueError
        If an entry with a local path is not training-ready (or its license
        does not permit ``intended_use``) and ``allow_reference_only`` is
        ``False``, or if no piece survived intake.
    FileNotFoundError
        If the manifest or a declared ``local_path`` does not exist.
    """
    config = config or CorpusConfig()
    entries = load_manifest(manifest_path)
    report = {
        name: [issue.message for issue in issues]
        for name, issues in validate_manifest(entries).items()
    }

    pieces: Dict[str, TokenisedPiece] = {}
    rejections: List[Tuple[str, str]] = []
    datasets: List[Dict[str, object]] = []

    for entry in entries:
        if not entry.local_path:
            continue
        refusal = _entry_is_training_ready(entry, config.intended_use)
        if refusal and not config.allow_reference_only:
            raise ValueError(
                f"Dataset {entry.name!r} cannot be used for training:\n{refusal}\n"
                "Fix its provenance record, drop it from the manifest, or set "
                "allow_reference_only=True to ingest it as held-out evaluation "
                "material only."
            )
        reference_only = refusal is not None
        accepted, rejected = _ingest_entry(entry, config, reference_only)
        pieces.update({piece.composition_id: piece for piece in accepted})
        rejections.extend(rejected)
        datasets.append(
            {
                "name": entry.name,
                "source_url": entry.source_url,
                "license": entry.license,
                "commercial_use_allowed": entry.commercial_use_allowed,
                "attribution_required": entry.attribution_required,
                "local_path": entry.local_path,
                "reference_only": reference_only,
                "reference_only_reason": refusal,
                "accepted_pieces": len(accepted),
                "rejected_pieces": len(rejected),
                "corpus_hash": corpus_hash([piece.sha256 for piece in accepted]),
            }
        )

    if not pieces:
        raise ValueError(
            f"No usable pieces found via {manifest_path}. Rejections: "
            f"{rejections[:5] or 'none — the manifest declared no local_path'}"
        )

    # Reference-only material is evaluation material: it may never train.
    trainable = sorted(cid for cid, piece in pieces.items() if not piece.reference_only)
    held_only = sorted(cid for cid, piece in pieces.items() if piece.reference_only)
    assignment = split_compositions(trainable, config.fractions, config.split_seed)
    assignment["test"] = assignment["test"] + held_only

    splits = {
        name: [pieces[cid] for cid in assignment[name]] for name in SPLIT_NAMES
    }
    return CorpusBundle(
        splits=splits,
        rejections=rejections,
        datasets=datasets,
        provenance_report=report,
        config=config,
    )


def _ingest_entry(
    entry: DatasetEntry, config: CorpusConfig, reference_only: bool
) -> Tuple[List[TokenisedPiece], List[Tuple[str, str]]]:
    """Parse, filter, normalise, and tokenise every file of one dataset entry."""
    from ..models.ngram_training import melody_token_stream  # local: avoids a cycle
    from ..tokenization import SymbolicTokenizer

    tokenizer = SymbolicTokenizer(config.tokenizer)
    root = Path(entry.local_path)
    accepted: List[TokenisedPiece] = []
    rejected: List[Tuple[str, str]] = []

    for dataset_file in iter_dataset_files(root):
        # Namespace ids by dataset so two datasets can share relative paths.
        piece_id = f"{entry.name}/{dataset_file.composition_id}"
        try:
            tracks, info = parse_midi_file_with_info(dataset_file.path)
        except Exception as error:  # unreadable/corrupt MIDI is data, not a crash
            rejected.append((piece_id, f"unreadable: {type(error).__name__}: {error}"))
            continue

        signature = info.primary_time_signature()
        if config.time_signatures is not None and signature not in config.time_signatures:
            rejected.append((piece_id, f"time_signature: {signature}"))
            continue
        if not info.is_single_time_signature:
            rejected.append((piece_id, "time_signature: changes mid-piece"))
            continue

        selection = select_melody_track(
            tracks, info,
            max_polyphony_ratio=config.max_polyphony_ratio,
            min_notes=config.min_notes,
        )
        if selection is None:
            rejected.append((piece_id, "no_melody_track: none monophonic and long enough"))
            continue

        track_name, notes = selection
        normalised, original_key = normalise_melody(notes, config)
        tokens = melody_token_stream(normalised, tokenizer)
        if len(tokens) < config.min_notes * 3:
            rejected.append((piece_id, "too_short: fewer notes than min_notes after tokenizing"))
            continue

        accepted.append(
            TokenisedPiece(
                composition_id=piece_id,
                dataset=entry.name,
                sha256=dataset_file.sha256,
                track=track_name,
                note_count=len(normalised),
                original_key=original_key,
                time_signature=signature,
                tokens=tuple(tokens),
                reference_only=reference_only,
            )
        )
    return accepted, rejected


__all__ = [
    "SPLIT_NAMES",
    "CorpusConfig",
    "CorpusBundle",
    "TokenisedPiece",
    "build_corpus",
    "normalise_melody",
    "select_melody_track",
    "split_compositions",
]
