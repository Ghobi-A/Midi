# ML Roadmap

Creative Audio Lab's first shipped feature — the Prompt-to-MIDI Generator —
is deliberately **deterministic and rule-based**. No model is trained in
this pass. This document explains why that's the right first step, and lays
out the concrete progression toward a learned system, so the codebase is
structured for that future without pretending to be there already.

## Why start deterministic

A rule-based baseline is:

- **Explainable.** Every note traces back to a chord degree, a scale, or an
  explicit style rule — useful for debugging and for the evaluation metrics
  in `creative_audio_lab.evaluation` to mean something.
- **A real baseline.** Any future learned model has to beat this, not an
  imagined one. `evaluation.metrics` gives concrete numbers (scale
  adherence, harmonic fit, repetition, novelty) to compare against.
- **Immediately shippable.** No training data, GPUs, or model hosting
  required — the whole app extra is `mido` + `numpy` + `streamlit`.

## Stage 1.5 — ML readiness (implemented)

Between the deterministic baseline and any actual training sits a layer of
infrastructure that has to exist first. It's implemented now:

- **Symbolic tokenization** (`creative_audio_lab.tokenization`): a
  dependency-free REMI-style tokenizer maps `Note` events to and from flat
  token sequences — the representation a symbolic sequence model trains
  on. [MidiTok](https://github.com/Natooz/MidiTok) and
  [symusic](https://github.com/Yikai-Liao/symusic) are supported as an
  *optional* extra (`pip install -e ".[symbolic]"`) for tokenizing external
  MIDI files with the reference REMI implementation; they are candidate
  future preprocessing dependencies, not core requirements.
- **Dataset provenance** (`creative_audio_lab.data.provenance`,
  `.license_policy`): manifest entries carry source URL, license,
  commercial-use/attribution flags, and tags, and validation refuses
  fan-archive/unclear-rights material before it can reach a training
  corpus.
- **Backend interface** (`creative_audio_lab.models`): the app now
  generates through a `GenerationBackend` abstraction.
  `DeterministicBackend` wraps the existing parser + generators and remains
  the default. `Text2MidiAdapter` and `MidiLlmAdapter` are placeholder
  adapters marking where learned models plug in —
  [Text2midi](https://github.com/AMAAI-Lab/Text2midi)-style
  caption-conditioned transformers and MIDI-token LLMs are *candidate
  future integration baselines*, not current dependencies; the adapters
  import nothing heavy and raise `NotImplementedError` with integration
  guidance.
- **Preview/analysis** (`creative_audio_lab.preview`): track summaries and
  piano-roll grids give both the app and future dataset tooling an
  inspectable, dependency-light view of any note content.

## Planned progression

Steps 7–10 below (the n-gram baseline onward) are planned in detail —
dataset choice, tokenizer changes, evaluation additions, and go/no-go
gates — in [`STAGE3_PLAN.md`](STAGE3_PLAN.md).

1. **Dataset ingestion** — point `creative_audio_lab.data.midi_dataset_loader`
   at a local directory of MIDI files (see `DATASETS.md` for sourcing and
   licensing guidance), registered through the dataset manifest and passed
   through provenance validation. Nothing is downloaded automatically.
2. **MIDI parsing into note events** — already implemented in
   `creative_audio_lab.midi_parser`; reused unchanged by the training
   pipeline so generation and training share one canonical `Note`
   representation.
3. **Feature extraction** — interval, rhythm, and contour features per
   `creative_audio_lab.motif_detection`, extended with tempo/key-normalized
   statistics (`creative_audio_lab.data.preprocess_midi` already provides
   quantization and transposition primitives to build on).
4. **Motif embeddings** — learn fixed-size vector representations of the
   motifs `motif_detection` already extracts, so similar melodic/rhythmic
   ideas land near each other in embedding space.
5. **Motif clustering** — group embedded motifs (e.g. k-means or HDBSCAN)
   to discover recurring melodic vocabulary per style/mood, replacing the
   hand-written `STYLE_PRESETS` tables with data-driven equivalents.
6. **Prompt-to-control classifier** — replace `prompt_parser`'s keyword
   tables with a learned classifier mapping free text to
   `GenerationControls` (mood, style, energy, instrumentation), while
   keeping the same typed output so every downstream generator is
   unaffected.
7. **Markov / n-gram continuation baseline** *(implemented — Stage 2)* — a
   first statistical generator: `creative_audio_lab.models.NgramMelodyBackend`
   samples melodic continuations from n-gram statistics over the internal
   token vocabulary, bootstrapped from the deterministic generator's own
   output. It exists to exercise the tokenize→train→sample→evaluate loop
   and to set the held-out NLL bar a neural model must beat — not to sound
   better than the rules it learned from.
8. **Transformer-based symbolic MIDI generation** — a sequence model over
   tokenized note events trained on the ingested corpus, generating full
   parts rather than just continuations. The Stage 1.5 tokenization layer
   (internal REMI-style tokenizer, optional MidiTok/symusic) defines the
   vocabulary; the Stage 1.5 backend adapters define where the model plugs
   into the app.
9. **Conditional generation** — condition the transformer on key, tempo,
   mood, genre, instrument set, and an optional seed motif, so prompts and
   UI controls steer generation the same way they do today.
10. **Evaluation** — extend `creative_audio_lab.evaluation.metrics` with
    motif retention, novelty, harmonic validity, and rhythmic coherence
    comparisons between the rule-based baseline and each learned model,
    plus human usability review (can a producer actually use this in a
    DAW without heavy editing?) before anything replaces the deterministic
    path in the default app experience.

## What does *not* change

Regardless of which stage above is reached, the output contract stays the
same: **editable, DAW-ready MIDI**, not rendered audio. This project is a
symbolic music generator, not an audio-generation clone.
