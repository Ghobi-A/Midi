# Machine-learning pipeline audit

Audit date: 2026-09-05. This audit distinguishes executable code from plans
and synthetic verification from empirical evidence.

## End-to-end inventory

| Stage | Implemented | Automated evidence | Empirical status |
| --- | --- | --- | --- |
| Provenance | Local JSON manifests, source/licence fields, intended-use policy, refusal of unclear/fan material | Unit tests cover validation and reference-only confinement | No shipped real manifest or corpus |
| Intake | Recursive local MIDI discovery and content SHA-256; no downloader | Fixture MIDI and missing/corrupt input tests | Validated only on generated fixtures |
| Parsing | One canonical `mido` parser with tempo, meter, program and drum metadata | Parser and corpus pipeline tests | No published corpus acceptance study |
| Preprocessing | Melody selection, monophony/minimum-note filters, quantisation and estimated-key transposition | Unit and end-to-end fixture tests | Key estimation/filter thresholds have not been calibrated on real data |
| Splitting | Seeded composition-ID hashing; complete compositions assigned once | Determinism, disjointness, and reference-only tests | No real-corpus split exists in this repository |
| Statistical models | Flat n-gram and factorised pitch/duration/velocity n-gram; smoothing, sampling and serialization | Probability, context-regression, reproducibility and round-trip tests | Synthetic bootstrap only |
| Generalisation evaluation | Bits/note, perplexity, pitch bits/note and pitch OOV on supplied splits | Synthetic held-out tests and leakage checks | No real held-out scores |
| Generation evaluation | Scale/chord fit, density, pitch range, interval entropy/repetition, rhythm diversity, motif retention; explicit aggregation | Metric unit tests | Proxy metrics have not been validated against listener/producer judgments |
| Tracking and selection | Canonical runner writes run JSON, selected model and generated Markdown; model order is selected on validation, then exactly one model is tested | Protocol regression tests | Ready to run, but no real run is checked in |
| Error analysis | Threshold-based flags with measurements for nine requested pathologies | Focused tests | Thresholds are diagnostic conventions, not learned perceptual boundaries |
| Neural model | Adapter placeholders only | Availability/import tests | Not implemented, appropriately gated |

## What was scaffolded rather than implemented

`Text2MidiAdapter` and `MidiLlmAdapter` are deliberately unavailable
interfaces. The roadmap's embeddings, clustering, prompt classifier and
Transformer are proposals. Optional Torch/Transformers dependencies do not
constitute a model. There are no weights, neural training loop, real-corpus
run artefacts, listening study, or human preference labels.

## Claims justified today

The project can claim that its deterministic generator is reproducible; its
local-only rights gate, canonical parser and composition-level splitting work
on automated fixtures; its factorised n-gram conditions pitch on pitch
history; and its experiment runner enforces validation-only model-order
selection. It can also claim generation proxy statistics for a named run.

It cannot claim musical quality, superiority to another system, learning from
real music, real-world generalisation, commercial suitability of user data,
or readiness for a neural model. Generated-sample metrics are not
generalisation evidence.

## What blocks a reproducible real-data result

The sole unavoidable external blocker is a locally supplied, rights-cleared
MIDI corpus and completed manifest. Corpus-specific risks remain: duplicate or
related compositions under different filenames, unreliable track labels,
key-estimation errors, very small validation/test sets, and rights that may
not permit the intended use. Content hashing detects identical bytes, but not
near-duplicate arrangements or multiple movements/versions of one work; users
must group or deduplicate these before making strong claims.

Once those inputs exist, the executable contract is:

```bash
python scripts/run_real_data_experiment.py \
  --manifest /path/to/manifest.json --output-dir experiments --split-seed 0
```

This performs manifest validation → canonical parsing → preprocessing →
composition split → unigram/bigram/trigram training → validation selection →
one final test evaluation → fixed-prompt generation benchmark → JSON/model/
Markdown artefacts. It makes no network request and does not download MIDI.

## Audit conclusion

The repository now has an experimentally disciplined *mechanism*, not a real
experimental *result*. The next work item is to execute the command on an
appropriately licensed corpus, inspect rejections and duplicates, and publish
the unedited run artefacts. Adding architecture before that would not improve
the evidence.
