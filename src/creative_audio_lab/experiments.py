"""Reproducible real-corpus baseline experiment and model selection.

The protocol deliberately exposes the test split only after one candidate has
been selected by validation cross entropy. Generated-sample statistics are
stored in a separate section and are never labelled as generalisation.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from .data import CorpusBundle, CorpusConfig, build_corpus
from .evaluation import (aggregate, analyse_generation, evaluate_arrangement,
                         motif_retention_score)
from .models.artefact import git_commit, save_artefact
from .models.deterministic_backend import DeterministicBackend
from .models.ngram_backend import NgramMelodyBackend
from .models.ngram_training import train_melody_model
from .models.note_event_model import MelodyModel, evaluate_melody_model
from .music_theory import scale_pitch_classes

DEFAULT_PROMPTS = (
    "dark orchestral theme in C minor, 120 BPM",
    "romantic piano melody in C major, 90 BPM",
    "energetic cinematic melody in G minor, 140 BPM",
    "calm ambient melody in D major, 80 BPM",
)


@dataclass(frozen=True)
class Candidate:
    model_id: str
    kind: str
    order: int


class ValidationSelector:
    """Stateful guard preventing candidate comparison on test measurements."""

    def __init__(self) -> None:
        self.validation: Dict[str, Dict[str, float]] = {}
        self.selected_id: Optional[str] = None

    def add_validation(self, candidate: Candidate, metrics: Dict[str, float]) -> None:
        if self.selected_id is not None:
            raise RuntimeError("selection is already frozen")
        self.validation[candidate.model_id] = metrics

    def select(self) -> str:
        eligible = [(m["bits_per_note"], cid) for cid, m in self.validation.items()
                    if m.get("notes", 0) > 0]
        if not eligible:
            raise ValueError("validation split contains no scoreable notes")
        self.selected_id = min(eligible)[1]
        return self.selected_id

    def evaluate_test(self, candidate_id: str, model: MelodyModel,
                      sequences: Sequence[Sequence[str]]) -> Dict[str, float]:
        if self.selected_id is None:
            raise RuntimeError("select on validation before evaluating test")
        if candidate_id != self.selected_id:
            raise ValueError("test evaluation is permitted only for the frozen selection")
        return evaluate_melody_model(model, sequences)


def _identity(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


GENERATION_ABLATIONS = (
    {"temperature": 1.0, "top_k": None, "seed_beats": 4.0},
    {"temperature": 0.7, "top_k": None, "seed_beats": 4.0},
    {"temperature": 1.3, "top_k": None, "seed_beats": 4.0},
    {"temperature": 1.0, "top_k": 5, "seed_beats": 4.0},
    {"temperature": 1.0, "top_k": None, "seed_beats": 1.0},
)


def _generation_rows(model: MelodyModel, model_id: str, *, samples: int) -> list[dict]:
    rows = []
    for prompt in DEFAULT_PROMPTS:
        reference = DeterministicBackend().generate(prompt, bars=8).parts["melody"]
        for ablation in GENERATION_ABLATIONS:
            for seed in range(samples):
                backend = NgramMelodyBackend(model=model, sampling_seed=seed, **ablation)
                arrangement = backend.generate(prompt, bars=8)
                melody = arrangement.parts["melody"]
                metrics = evaluate_arrangement(arrangement)
                metrics["motif_retention"] = motif_retention_score(reference, melody)
                intervals = [b.pitch - a.pitch for a, b in zip(melody, melody[1:])]
                rows.append({
                    "model": model_id, "prompt": prompt, "seed": seed,
                    "generation_config": ablation,
                    "metrics": metrics,
                    "interval_distribution": {
                        str(value): intervals.count(value) / len(intervals)
                        for value in sorted(set(intervals))
                    } if intervals else {},
                    "error_analysis": analyse_generation(
                        melody,
                        scale_pcs=scale_pitch_classes(arrangement.controls.key,
                                                      arrangement.controls.mode),
                        expected_beats=arrangement.bars * 4.0,
                    ),
                })
    return rows


def run_experiment(manifest: str | Path, output_dir: str | Path, *,
                   config: Optional[CorpusConfig] = None,
                   orders: Iterable[int] = (1, 2, 3), samples: int = 3) -> Path:
    """Run intake, validation-only selection, final test, generation, and reporting."""
    started = time.monotonic()
    config = config or CorpusConfig()
    bundle: CorpusBundle = build_corpus(manifest, config)
    if not bundle.splits["train"]:
        raise ValueError("training split is empty")
    candidates = [Candidate(f"note-event-{order}gram", "factorised", order)
                  for order in orders]
    selector = ValidationSelector()
    models: Dict[str, MelodyModel] = {}
    train_metrics = {}
    for candidate in candidates:
        model = train_melody_model(bundle.sequences("train"), candidate.kind,
                                   order=candidate.order)
        models[candidate.model_id] = model
        train_metrics[candidate.model_id] = evaluate_melody_model(
            model, bundle.sequences("train"))
        selector.add_validation(candidate, evaluate_melody_model(
            model, bundle.sequences("val")))
    selected = selector.select()
    test_metrics = selector.evaluate_test(selected, models[selected],
                                          bundle.sequences("test"))

    # The deterministic generator is a generation baseline only: unlike a
    # probability model it cannot honestly be assigned cross entropy.
    generation = []
    for prompt in DEFAULT_PROMPTS:
        arrangement = DeterministicBackend().generate(prompt, bars=8)
        melody = arrangement.parts["melody"]
        metrics = evaluate_arrangement(arrangement)
        metrics["motif_retention"] = 1.0
        intervals = [b.pitch - a.pitch for a, b in zip(melody, melody[1:])]
        generation.append({"model": "deterministic", "prompt": prompt, "seed": 0,
                           "generation_config": {"temperature": None, "top_k": None,
                                                 "seed_beats": None},
                           "metrics": metrics,
                           "interval_distribution": {
                               str(value): intervals.count(value) / len(intervals)
                               for value in sorted(set(intervals))
                           } if intervals else {},
                           "error_analysis": analyse_generation(
                               melody,
                               scale_pcs=scale_pitch_classes(arrangement.controls.key,
                                                             arrangement.controls.mode),
                               expected_beats=32.0)})
    for candidate in candidates:
        generation.extend(_generation_rows(models[candidate.model_id],
                                           candidate.model_id, samples=samples))

    aggregates: Dict[str, Dict[str, dict]] = {}
    for model_id in {row["model"] for row in generation}:
        model_rows = [row for row in generation if row["model"] == model_id]
        keys = model_rows[0]["metrics"]
        aggregates[model_id] = {
            key: aggregate(row["metrics"][key] for row in model_rows)
            for key in keys
        }

    split_payload = {name: bundle.composition_ids(name) for name in ("train", "val", "test")}
    run_core = {"corpus_hash": bundle.corpus_hash, "split": split_payload,
                "config": asdict(config), "orders": [c.order for c in candidates]}
    experiment_id = _identity(run_core)[:16]
    out = Path(output_dir) / experiment_id
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(), "python": platform.python_version(),
        "corpus": {"identity": bundle.corpus_hash, "manifest": str(Path(manifest).resolve()),
                   "datasets": bundle.datasets, "stats": bundle.stats()},
        "split": {"identity": _identity(split_payload), "seed": config.split_seed,
                  "level": "composition", "compositions": split_payload},
        "preprocessing": asdict(config),
        "candidates": [asdict(candidate) for candidate in candidates],
        "selection": {"criterion": "minimum validation bits_per_note",
                      "selected": selected, "test_access": "after selection only"},
        "random_seed": config.split_seed,
        "train_metrics": train_metrics, "validation_metrics": selector.validation,
        "test_metrics": {selected: test_metrics},
        "generation_metrics": {"per_sample": generation, "aggregate": aggregates},
        "runtime_seconds": time.monotonic() - started,
        "artefact_path": str((out / "selected_model.json").resolve()),
    }
    save_artefact(models[selected], {
        "corpus": metadata["corpus"], "split": metadata["split"],
        "tokenizer": asdict(config.tokenizer), "seeds": {"split": config.split_seed},
        "stats": bundle.stats(), "metrics": {"train": train_metrics[selected],
        "val": selector.validation[selected], "test": test_metrics},
        "experiment_id": experiment_id,
    }, out / "selected_model.json")
    (out / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=True), encoding="utf-8")
    (out / "report.md").write_text(render_report(metadata), encoding="utf-8")
    return out


def render_report(run: Dict[str, Any]) -> str:
    """Render a concise report directly from a run artefact."""
    selected = run["selection"]["selected"]
    lines = [f"# Experiment {run['experiment_id']}", "", "## Data and leakage prevention",
             f"Corpus SHA-256 identity: `{run['corpus']['identity']}`.",
             f"Split identity: `{run['split']['identity']}`; whole compositions only.",
             "", "## Validation-based model selection",
             "| model | validation bits/note | validation pitch bits/note | n |",
             "| --- | ---: | ---: | ---: |"]
    for candidate, metrics in run["validation_metrics"].items():
        lines.append(f"| {candidate} | {metrics['bits_per_note']:.4f} | "
                     f"{metrics['pitch_bits_per_note']:.4f} | {metrics['notes']} |")
    test = run["test_metrics"][selected]
    lines += ["", f"Selected **{selected}** before opening test.", "",
              "## Final unseen-composition result",
              f"Test: **{test['bits_per_note']:.4f} bits/note**, "
              f"perplexity **{test['perplexity_per_note']:.4f}**, pitch OOV "
              f"**{test['pitch_oov_rate']:.4f}**, n={test['notes']} notes.", "",
              "## Generation quality", "Generation metrics are sample diagnostics, not evidence of generalisation.",
              "", "See `run.json` for per-sample values, mean, sample standard deviation, n, and error flags."]
    return "\n".join(lines) + "\n"


__all__ = ["Candidate", "ValidationSelector", "run_experiment", "render_report"]
