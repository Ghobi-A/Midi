# Experiment results

## Status: real-data experiment pending

No rights-cleared real MIDI corpus is available in this repository or in the
development environment. Consequently this document contains no manufactured
scores. Synthetic MIDI fixtures verify the machinery but are not portfolio
results and must not be represented as such.

## Problem definition

Predict and continue monophonic symbolic melody note events while retaining a
producer-facing deterministic arrangement baseline. The empirical question is
whether statistical order improves likelihood on unseen *compositions*, and
what generation failures accompany that choice.

## Dataset and provenance

Pending a user-supplied local manifest whose entries pass the source, licence,
commercial-use and attribution policy. A run records each dataset, intake
rejections and a content-derived corpus identity. Nothing downloads data.

## Methodology and leakage prevention

The canonical parser selects a plausible monophonic melody, quantises it and
optionally normalises tonic. A seeded hash assigns whole composition IDs to
train/validation/test. Unigram, bigram and trigram factorised note-event models
train on `train`; minimum validation bits/note selects one model; only that
frozen selection is evaluated on `test`. The deterministic baseline has no
probability distribution and therefore receives no invented cross entropy.

## Metrics

Generalisation: total and pitch cross entropy in bits per note, perplexity per
note, pitch perplexity and pitch OOV, always with held-out note count.
Generation quality: scale adherence, harmonic fit, note density, pitch range,
interval diversity, rhythmic diversity, repetition and motif diagnostics.
Each run stores every generated sample plus mean, sample standard deviation
and sample count. These sample properties are explicitly separate from
generalisation metrics.

## Results

Pending. Generate this section's source artefacts without editing numbers:

```bash
python scripts/run_real_data_experiment.py \
  --manifest /path/to/rights-cleared-manifest.json --output-dir experiments
```

The resulting `report.md` is generated from `run.json`, and the selected model
is saved beside it. Preserve all three files when publishing a result.

## Error analysis

The run reports measured thresholds for repeated-note collapse, excessive
stepwise motion/leaps, tonal or rhythmic collapse, motif copying, premature
end, pathological length and low pitch diversity. Review flagged sample IDs
rather than selecting only pleasant examples. Thresholds are interpretable
heuristics and need validation against human review.

## Limitations and next experiment

Exact content hashes do not find transposed or rearranged duplicates; automatic
key and melody-track inference can fail; likelihood does not equal usefulness;
and fixed prompts cover only a small control surface. The next experiment is a
complete real-corpus run followed by manual review of intake rejections,
composition-family leakage and flagged generations. Neural work remains gated
by `NEURAL_BASELINE_DECISION.md`.
