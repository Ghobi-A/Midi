# ML Roadmap

Creative Audio Lab's first shipped feature — the Prompt-to-MIDI Generator —
is deliberately **deterministic and rule-based**, and that backend remains
the default. The first *statistical* baseline (Stage 2, a factorised note-event model
that continues a melody) is now implemented on top of it, together with a
rights-checked corpus pipeline and held-out evaluation; no neural model is
trained, and no external data is downloaded. This document explains why starting deterministic was the right
first step, and lays out the concrete progression toward a learned system,
so the codebase is structured for that future without pretending to be
there already.

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

## Stage 2 — statistical melody baseline (implemented; corrected)

The first statistical generator is in place, deliberately scoped small.

### The context bug, and what replaced it

The first version of this baseline fitted a single `NGramModel(order=3)`
to the interleaved token stream

    NOTE_ON_p → VELOCITY_v → DURATION_d → NOTE_ON_p' → ...

An order-3 model conditions on the two preceding tokens, so when it
predicted a `NOTE_ON` its context was `(VELOCITY_prev, DURATION_prev)` —
**the previous pitch had already fallen out of the window.** The model
therefore learned pitch given rhythm and dynamics, velocity given pitch,
and duration given pitch and velocity, but never a pitch-to-pitch melodic
transition. The tests of the time checked trigram mechanics and valid MIDI
output, so nothing caught it.

The fix models a note as a composite event whose attributes are each
predicted from their own history
(`creative_audio_lab.models.note_event_model.NoteEventModel`):

- `P(pitch_i | pitch_{i-1}, pitch_{i-2})` — a genuine melodic trigram,
- `P(duration_i | duration_{i-1}, ...)`,
- `P(velocity_i | velocity_{i-1}, ...)`.

`tests/test_note_event_model.py` pins this down with a corpus where the
next pitch is a deterministic function of the previous one while every
velocity and duration is identical: the factorised model reaches
`P(next | prev) > 0.9`, the flat order-3 model provably cannot separate
the cycles, and flat order-4 recovers the dependency. Cross-attribute
dependencies (duration given pitch, say) are deliberately not modelled
yet; that is the next refinement, not an oversight.

### Components

- **Model** (`models.ngram_model`): a pure-Python count-based n-gram model
  — configurable order, backoff to shorter contexts, temperature and top-k
  sampling, deterministic sampling under a fixed seed, JSON save/load.
  Sequences carry `BOS`/`EOS` boundaries, so piece starts and ends are
  learned rather than smeared together. Witten-Bell interpolation gives
  every token a positive probability, which makes `cross_entropy`,
  `perplexity`, and `oov_rate` well defined on unseen data.
- **Note-event model** (`models.note_event_model`): the factorised default
  described above, plus `evaluate_flat_model` so both kinds report the same
  per-note units (bits/note, with the pitch component broken out).
- **Corpus pipeline** (`data.corpus_pipeline`): manifest → provenance and
  licence gate → MIDI parse → melody-track selection → quantise and
  transpose to a common tonic → **composition-level** train/val/test split
  → tokenised sequences. Splitting whole compositions is what makes the
  held-out numbers meaningful; splitting token windows would leak a
  melody's own continuation into its evaluation.
- **Training** (`models.ngram_training`, `scripts/train_ngram_melody.py`):
  fits either model kind on the synthetic bootstrap corpus or on a real
  corpus via `--manifest`, and writes a **training artefact** — the model
  plus corpus identity and licences, split assignment, tokenizer config,
  seeds, package version and git commit, corpus statistics, and held-out
  metrics.
- **Backend** (`models.ngram_backend`): `NgramMelodyBackend` keeps the
  deterministic chords/bass/drums/structure and replaces only the melody;
  the deterministic motif's first bar seeds the model, which continues the
  line. Loads an artefact through `model_path` and exposes its metadata to
  the app.
- **Evaluation** (`scripts/evaluate_melody_models.py`): scores artefacts on
  the same held-out split, and refuses to report a number when an
  artefact's training compositions appear in that split or when the corpus
  on disk no longer matches the one the artefact records.
- **Sample benchmark** (`scripts/compare_backends.py`): note density,
  harmonic fit, novelty, repetition, scale adherence, motif retention, note
  counts, MIDI validity. These characterise *generated samples*; they say
  nothing about generalisation, which is what the held-out benchmark is
  for.

### What is still missing

Training on real music. The repository ships no corpus and claims no
real-corpus result; the benchmark is a command you run on data you have
cleared. The test suite proves the path works end to end on a synthetic
fixture corpus it generates itself.

## Planned progression

1. **Dataset ingestion** — ✅ *implemented*:
   `creative_audio_lab.data.corpus_pipeline.build_corpus` takes a dataset
   manifest (see `DATASETS.md` for sourcing and licensing guidance),
   enforces provenance and licence policy, walks each local directory,
   and produces tokenised, composition-level splits. Pieces are identified
   by their path relative to the dataset root and pinned by content hash.
   Nothing is downloaded automatically.
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
7. **Markov / n-gram continuation baseline** — ✅ *implemented; melodic
   context corrected*: `creative_audio_lab.models.ngram_backend` samples
   melodic continuations from a factorised note-event model over symbolic
   tokens (see the Stage 2 section above). Training, sampling,
   serialization, held-out evaluation, the rights-checked corpus pipeline,
   and artefact metadata are done. What remains for this item is running it
   on a real, rights-cleared corpus — a command, not more code.
8. **Transformer-based symbolic MIDI generation** — *not started, and
   deliberately gated on item 7 producing a real-corpus number first: a
   neural model is only worth training once there is a held-out baseline
   for it to beat.* A sequence model over
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
