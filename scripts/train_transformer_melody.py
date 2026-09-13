#!/usr/bin/env python
"""Train the causal note-event Transformer on a rights-checked manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from creative_audio_lab.data import CorpusConfig, build_corpus
from creative_audio_lab.models.neural_events import NoteEventCodec


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--epochs", type=int, default=20)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--patience", type=int, default=4)
    result.add_argument("--context-length", type=int, default=128)
    result.add_argument("--d-model", type=int, default=192)
    result.add_argument("--layers", type=int, default=4)
    result.add_argument("--heads", type=int, default=6)
    result.add_argument("--feedforward", type=int, default=768)
    result.add_argument("--dropout", type=float, default=0.1)
    result.add_argument("--split-seed", type=int, default=0)
    result.add_argument("--training-seed", type=int, default=0)
    result.add_argument("--val-fraction", type=float, default=0.1)
    result.add_argument("--test-fraction", type=float, default=0.1)
    result.add_argument("--min-notes", type=int, default=16)
    result.add_argument("--device", default="auto")
    result.add_argument("--intended-use", choices=("research", "commercial"), default="research")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        import torch  # noqa: F401
        from creative_audio_lab.models.transformer_melody import TransformerConfig
        from creative_audio_lab.models.transformer_training import (
            TrainingConfig, evaluate_transformer, save_transformer_artefact,
            train_transformer,
        )
    except ImportError as error:
        raise SystemExit("Transformer training requires: pip install -e '.[ml]'") from error

    train_fraction = 1 - args.val_fraction - args.test_fraction
    corpus_config = CorpusConfig(fractions=(train_fraction, args.val_fraction, args.test_fraction),
                                 split_seed=args.split_seed, min_notes=args.min_notes,
                                 intended_use=args.intended_use)
    started = time.monotonic()
    bundle = build_corpus(args.manifest, corpus_config)
    codec = NoteEventCodec(corpus_config.tokenizer)
    encoded = {split: [codec.encode_stream(sequence) for sequence in bundle.sequences(split)]
               for split in ("train", "val", "test")}
    architecture = TransformerConfig(*codec.vocab_sizes, d_model=args.d_model,
                                     layers=args.layers, heads=args.heads,
                                     feedforward=args.feedforward, dropout=args.dropout,
                                     context_length=args.context_length)
    training = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size,
                              learning_rate=args.learning_rate, patience=args.patience,
                              seed=args.training_seed, device=args.device)
    model, history, selected_epoch = train_transformer(encoded["train"], encoded["val"],
                                                       architecture, training)
    # Test is opened exactly once, after validation selected and restored the checkpoint.
    validation = evaluate_transformer(model, encoded["val"], batch_size=args.batch_size,
                                      device=args.device)
    test = evaluate_transformer(model, encoded["test"], batch_size=args.batch_size,
                                device=args.device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "corpus": {"identity": bundle.corpus_hash, "manifest": str(Path(args.manifest).resolve()),
                   "datasets": bundle.datasets, "stats": bundle.stats()},
        "splits": {name: bundle.composition_ids(name) for name in ("train", "val", "test")},
        "split_seed": args.split_seed, "training_seed": args.training_seed,
        "training_hyperparameters": asdict(training), "selected_epoch": selected_epoch,
        "selection_protocol": "best validation bits_per_note; test evaluated once after freeze",
        "validation_metrics": validation, "final_test_metrics": test,
        "training_history": history, "runtime_seconds": time.monotonic() - started,
    }
    artifact = save_transformer_artefact(output / "model", model, codec, metadata)
    run = {"kind": "transformer_melody", "artifact": str(artifact), **metadata}
    (output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    with (output / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (output / "report.md").write_text(
        "# Transformer melody experiment\n\n"
        f"Selected epoch: **{selected_epoch}** (validation only).\n\n"
        f"Validation: **{validation['bits_per_note']:.3f} bits/note**.\n\n"
        f"Final held-out test: **{test['bits_per_note']:.3f} bits/note**.\n\n"
        "Component likelihoods and complete provenance are in `run.json`.\n")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
