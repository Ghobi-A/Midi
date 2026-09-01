#!/usr/bin/env python
"""Compare melody models on the same held-out split of a real corpus.

Scores one or more training artefacts (written by
``scripts/train_ngram_melody.py``) on a corpus's held-out split and prints a
Markdown table. This is the synthetic-trained vs real-trained comparison:
generated-sample statistics say how a model *behaves*, but only held-out
likelihood says whether it *generalises* to music it has never seen.

The split is rebuilt from the same seed and fractions the artefact recorded,
and two safety checks run before any number is printed:

- the corpus hash in the artefact is compared with the corpus on disk, so a
  changed dataset cannot be silently scored as if it were the trained-on one;
- every artefact's recorded training compositions are checked against the
  evaluation split, so a leaked piece is reported instead of inflating the
  score.

Usage:
    python scripts/evaluate_melody_models.py bootstrap.json real.json \\
        --manifest datasets.json
    python scripts/evaluate_melody_models.py real.json --manifest datasets.json \\
        --split val --output benchmark.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Allow running straight from a checkout without an editable install.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from creative_audio_lab.data import CorpusConfig, build_corpus
from creative_audio_lab.models.artefact import load_any_model, model_kind
from creative_audio_lab.models.note_event_model import evaluate_melody_model

COLUMNS = (
    ("model", "{}"),
    ("kind", "{}"),
    ("trained on", "{}"),
    ("held-out notes", "{}"),
    ("bits/note", "{:.3f}"),
    ("pitch bits/note", "{:.3f}"),
    ("perplexity", "{:.1f}"),
    ("pitch OOV", "{:.3f}"),
)


def corpus_source(metadata: Optional[dict]) -> str:
    """Human-readable description of what an artefact was trained on."""
    if not metadata:
        return "unknown (bare model file)"
    corpus = metadata.get("corpus", {})
    if corpus.get("source") == "synthetic-bootstrap":
        return "synthetic bootstrap"
    names = [dataset.get("name") for dataset in corpus.get("datasets", [])]
    return ", ".join(name for name in names if name) or corpus.get("source", "unknown")


def check_corpus_identity(name: str, metadata: Optional[dict], bundle) -> List[str]:
    """Warn when an artefact's recorded corpus differs from the one on disk."""
    warnings: List[str] = []
    if not metadata:
        return [f"{name}: no metadata — cannot verify what it was trained on."]
    recorded = metadata.get("corpus", {}).get("corpus_hash")
    if recorded and recorded != bundle.corpus_hash:
        warnings.append(
            f"{name}: was trained on corpus {recorded[:12]}… but this manifest "
            f"builds {bundle.corpus_hash[:12]}… — the datasets differ."
        )
    return warnings


def check_leakage(name: str, metadata: Optional[dict], evaluation_ids: Sequence[str]) -> List[str]:
    """Report any composition that is in both the artefact's training set and the eval split."""
    if not metadata:
        return []
    trained = set(metadata.get("split", {}).get("compositions", {}).get("train", []))
    leaked = sorted(trained.intersection(evaluation_ids))
    if not leaked:
        return []
    return [
        f"{name}: {len(leaked)} evaluation composition(s) were in its training "
        f"split, e.g. {leaked[0]} — this score is not held out."
    ]


def render_markdown(rows: Sequence[Dict[str, object]]) -> str:
    headers = [name for name, _ in COLUMNS]
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    for row in rows:
        cells = []
        for name, fmt in COLUMNS:
            value = row[name]
            cells.append(fmt.format(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artefacts", nargs="+", help="Artefact (or bare model) JSON paths")
    parser.add_argument("--manifest", required=True, help="Manifest of the evaluation corpus")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="Override the split seed (default: the first artefact's)")
    parser.add_argument("--intended-use", choices=("research", "commercial"), default="research")
    parser.add_argument("--allow-reference-only", action="store_true")
    parser.add_argument("--output", default=None, help="Also write the table to this file")
    args = parser.parse_args(argv)

    loaded = []
    for path in args.artefacts:
        model, metadata = load_any_model(path)
        loaded.append((Path(path).name, model, metadata))

    seed = args.split_seed
    if seed is None:
        seed = next(
            (meta.get("split", {}).get("seed", 0) for _, _, meta in loaded if meta),
            0,
        )
    bundle = build_corpus(
        args.manifest,
        CorpusConfig(
            intended_use=args.intended_use,
            split_seed=seed,
            allow_reference_only=args.allow_reference_only,
        ),
    )
    sequences = bundle.sequences(args.split)
    evaluation_ids = bundle.composition_ids(args.split)

    rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    for name, model, metadata in loaded:
        warnings += check_corpus_identity(name, metadata, bundle)
        warnings += check_leakage(name, metadata, evaluation_ids)
        metrics = evaluate_melody_model(model, sequences)
        rows.append(
            {
                "model": name,
                "kind": model_kind(model),
                "trained on": corpus_source(metadata),
                "held-out notes": metrics["notes"],
                "bits/note": metrics["bits_per_note"],
                "pitch bits/note": metrics["pitch_bits_per_note"],
                "perplexity": metrics["perplexity_per_note"],
                "pitch OOV": metrics["pitch_oov_rate"],
            }
        )

    table = render_markdown(rows)
    print(f"# Held-out melody-model comparison ({args.split} split, seed {seed})\n")
    print(f"Corpus: {args.manifest} — {len(evaluation_ids)} compositions, "
          f"{sum(len(s) // 3 for s in sequences)} notes, hash {bundle.corpus_hash[:12]}…\n")
    print(table)
    print(
        "\nLower bits/note is better. Pitch bits/note is the melodic component: a "
        "model that does not condition on previous pitches cannot improve it. "
        "These are likelihoods on unseen music, not judgments of musical quality."
    )
    if warnings:
        print("\n## Warnings\n")
        for warning in warnings:
            print(f"- {warning}")
    if args.output:
        Path(args.output).write_text(table + "\n", encoding="utf-8")
        print(f"\nTable written to {args.output}")
    return 1 if any("not held out" in warning for warning in warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
