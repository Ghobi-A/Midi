# Creative Audio Lab

**What this is:** a symbolic-music engineering project — deterministic,
rule-based MIDI generation with the architecture laid out for an ML
pipeline to be added later. It is not a trained-model or applied-ML
portfolio piece; today's generation logic is rules and music theory, not
a learned model.

**Prompt-to-MIDI Generator** — a prompt-conditioned symbolic music generator
that creates DAW-ready MIDI arrangements instead of finished audio.

Type a prompt like `"dark orchestral boss battle theme, 140 BPM, brass
ostinato, strings, piano arps"`, pick a key/BPM/style if you want to override
the parser, and get back a full arrangement plus separated MIDI stems you can
drag straight into a DAW.

This is **not** an audio-generation clone of Suno or similar tools — it does
not render waveforms, and it makes no attempt to imitate a trained
audio-generation model. It's a symbolic (MIDI) generator: deterministic and
rule-based today, with the codebase structured so a real ML pipeline can be
layered in later (see [ML Roadmap](#ml-roadmap) below).

## Why MIDI instead of generated audio

- **Editable.** Every note is a discrete, movable MIDI event — not a
  waveform you have to fight to change.
- **DAW-ready.** Output loads directly into any DAW (Ableton, Logic,
  Reaper, FL Studio, ...) as a normal MIDI file.
- **Useful for producers and composers.** You get a sketch to build on —
  swap instruments, edit notes, re-harmonize — not a finished, opaque track.
- **Better for controlled composition workflows.** Key, scale, chord
  progression, and instrumentation are explicit and inspectable, not
  latent in a black-box model.

## Installation

Core install is intentionally lightweight — just [`mido`](https://mido.readthedocs.io/)
and `numpy`:

```bash
pip install -e .
```

Optional extras layer on top as needed:

```bash
pip install -e ".[app]"      # Streamlit UI
pip install -e ".[dev]"      # pytest, ruff
pip install -e ".[audio]"    # librosa, soundfile, pyloudnorm, demucs, spleeter, torchaudio
pip install -e ".[ml]"       # torch, transformers, scikit-learn, datasets (future ML work)
pip install -e ".[symbolic]" # miditok, symusic (optional MidiTok REMI tokenization)
```

None of `torch`, `torchaudio`, `transformers`, `demucs`, `spleeter`, or
`pyloudnorm` are required to generate MIDI. They're only needed if you use
the legacy audio-production utilities described [below](#legacy-audio-utilities).

## Quickstart

```python
from creative_audio_lab.prompt_parser import parse_prompt
from creative_audio_lab.generators.arrangement import build_arrangement
from creative_audio_lab.export import export_arrangement_files

controls = parse_prompt("romantic piano loop in C minor, 90 BPM")
arrangement = build_arrangement(controls)
files = export_arrangement_files(arrangement)  # {"full_arrangement.mid": bytes, "chords.mid": bytes, ...}

for name, data in files.items():
    with open(name, "wb") as fh:
        fh.write(data)
```

## Running the Streamlit app

```bash
pip install -e ".[app]"
streamlit run app.py
```

The app lets you:

1. Enter a text prompt.
2. Optionally override key, scale/mode, BPM, bar length, style preset,
   energy, rhythmic density, and instrument set (leave any of these on
   "Auto" to let the prompt parser infer it).
3. Click **Generate**.
4. Download `full_arrangement.mid` plus the individual
   `chords.mid` / `melody.mid` / `bass.mid` / `drums.mid` stems.
5. Review the arrangement summary, evaluation metrics, and detected motifs.

## Example prompts

- `dark orchestral boss battle theme, 140 BPM, brass ostinato, strings, piano arps`
- `romantic piano loop in C minor, 90 BPM`
- `trap melody with dark bells and sliding bass`
- `nostalgic JRPG town theme with piano and flute`
- `aggressive cinematic trailer cue with low brass and taiko drums`
- `hopeful piano and strings theme for a fantasy village`

## Generated outputs

Every generation produces:

| File | Contents |
| --- | --- |
| `full_arrangement.mid` | All parts combined on separate tracks/channels |
| `chords.mid` | The chord/harmony part |
| `melody.mid` | The lead melody part |
| `bass.mid` | The bassline |
| `drums.mid` | The drum pattern (General MIDI percussion, channel 10) |

## Architecture

```
app.py                              Streamlit UI
src/creative_audio_lab/
    prompt_parser.py                Text -> GenerationControls (deterministic, keyword-based)
    music_theory.py                 Scales, diatonic chords, the canonical Note type
    midi_parser.py                  MIDI file/track -> Note events
    motif_detection.py              Interval/rhythm/contour extraction, repeated-motif ranking
    motif_variation.py              Transpose, invert, augment/compress rhythm, call-and-response
    generators/
        chords.py                  Chord-progression generator
        melody.py                  Motif-based melody generator
        bass.py                    Bassline generator (root/fifth, ostinato, sliding)
        drums.py                   Drum-pattern generator (GM drum map)
        arrangement.py             Combines the above + assigns GM instrument programs
    export/midi_export.py           Note events -> MIDI bytes (full arrangement + stems)
    evaluation/metrics.py           Note density, harmonic fit, novelty, scale adherence, ...
    models/                         GenerationBackend interface, DeterministicBackend (default),
                                    placeholder Text2midi / MIDI-LLM adapters
    tokenization/                   Dependency-free REMI-style symbolic tokenizer
                                    + optional MidiTok/symusic adapter
    preview/                        MIDI summaries and piano-roll grids (data only, no audio)
    data/                           Dataset manifest + provenance/license validation +
                                    local-only MIDI loader + preprocessing
                                    (scaffolding for ML_ROADMAP.md — no auto-download)
tests/                              One test module per component above
docs/
    ML_ROADMAP.md                   Planned progression from rules to learned models
    DATASETS.md                     Candidate datasets, licensing notes, and hard rules
```

## Deterministic baseline approach

Every generator is a pure function: `GenerationControls` in, a list of
`Note` events out, with any internal randomness scoped to a local
`random.Random` seeded from the prompt/controls — so the same prompt always
produces the same arrangement. The pipeline is:

1. **Prompt parsing** — keyword and regex tables map mood, genre, section,
   instrumentation, texture, energy, and rhythmic density onto a typed
   `GenerationControls` object. No ML, no external calls.
2. **Chords** — a style-specific scale-degree progression, voiced as
   diatonic triads built by stacking thirds within the chosen scale (so it
   works for major, natural minor, harmonic minor, dorian, and phrygian
   without hardcoding each mode's chord qualities).
3. **Melody** — a short scale-degree motif is generated once per prompt,
   then restated across the progression with light transposition,
   occasional passing tones on large leaps, and a resolved phrase ending
   every four bars — instead of sampling pitches independently, which
   produces unmusical "note soup".
4. **Bass** — follows chord roots and fifths by default, with a repeating
   ostinato for cinematic/boss-battle cues, a glide into the second half
   of the bar for trap/drill presets, and a walking passing tone for
   high-energy styles.
5. **Drums** — tiles one of four General MIDI patterns (basic, trap,
   cinematic, dance) per bar, chosen from style and energy.
6. **Arrangement** — combines all four parts, picks General MIDI program
   numbers from the requested instrument set, and hands off to `export/`.

## Stage 1.5: ML-ready architecture

The generator is still deterministic, but the codebase now has the
scaffolding a text-conditioned symbolic model needs — implemented and
tested, without pretending a model is already trained:

- **Tokenization layer** (`creative_audio_lab.tokenization`) — a
  dependency-free, REMI-style tokenizer converts `Note` events to and from
  flat symbolic token sequences (`BAR`/`POSITION`/`NOTE_ON`/`DURATION`/
  `VELOCITY`/`TEMPO`/`PROGRAM`/`TIME_SIGNATURE`). An optional adapter wraps
  [MidiTok](https://github.com/Natooz/MidiTok) REMI for tokenizing external
  MIDI files (`pip install -e ".[symbolic]"`); without the extra it fails
  with a clear install hint instead of an obscure import error.
- **Dataset provenance registry** (`creative_audio_lab.data`) — dataset
  manifest entries now carry source URL, license, commercial-use and
  attribution flags, notes, and tags, with validation that errors on
  missing licenses/sources and on fan-archive/unclear-rights material, and
  warns on unverified commercial terms. See
  [`examples/dataset_manifest.example.json`](examples/dataset_manifest.example.json)
  (placeholder paths only — nothing is ever downloaded).
- **Backend interface** (`creative_audio_lab.models`) — a
  `GenerationBackend` ABC with `DeterministicBackend` (the current parser +
  generators) as the default implementation the app now calls.
- **Future model adapters** — `Text2MidiAdapter` and `MidiLlmAdapter` are
  scaffolded behind the same interface; they report themselves unavailable
  and raise `NotImplementedError` with concrete integration guidance.
- **Preview layer** (`creative_audio_lab.preview`) — per-track summaries
  (note counts, pitch range, bars, density, drum presence) and a
  piano-roll grid data structure, all pure data with no audio rendering.

Why still no trained model? Because training one properly requires a
rights-cleared corpus (see [`docs/DATASETS.md`](docs/DATASETS.md)), a
tokenization scheme that's already settled (this layer), and a baseline to
beat (the deterministic generator plus `evaluation.metrics`). Stage 1.5
puts those prerequisites in place so a model can be added honestly — not
bolted on. See [`docs/ML_ROADMAP.md`](docs/ML_ROADMAP.md).

## MIDI Motif Lab

A supporting module (`motif_detection.py` + `motif_variation.py`) for
analyzing and reworking any melodic line — generated or imported:

- Extract pitch intervals, rhythm durations, and up/down/repeat contour.
- Detect repeated melodic and rhythmic n-grams.
- Rank candidate motifs by repetition count, phrase length, rhythmic
  clarity, and melodic salience.
- Generate variations: transpose, invert the contour, augment/compress the
  rhythm, insert passing tones, or continue with a call-and-response
  answering phrase.
- Export any variation as a standalone `.mid` file.

The Streamlit app surfaces the top-ranked motifs detected in each
generated melody under "Detected motifs".

## Evaluation metrics

`creative_audio_lab.evaluation.metrics` computes, with no external corpus
or trained model required:

- **Note density** — notes per bar.
- **Pitch range** — lowest/highest pitch and span.
- **Repetition score** — fraction of repeated melodic n-grams.
- **Harmonic fit score** — fraction of notes matching the sounding chord's tones.
- **Rhythmic complexity score** — normalized entropy of note durations.
- **Motif retention score** — how much of a motif survives a variation.
- **Novelty score** — normalized entropy of melodic intervals.
- **Scale adherence score** — fraction of notes in the selected scale.

These are the numbers a future learned model would need to beat — not
proxies for "sounds good," but concrete, inspectable baselines.

## ML Roadmap

This first pass ships **no trained model** — see
[`docs/ML_ROADMAP.md`](docs/ML_ROADMAP.md) for the full plan, and
[`docs/DATASETS.md`](docs/DATASETS.md) for dataset candidates (MAESTRO,
POP909, GiantMIDI-Piano, a carefully-filtered Lakh MIDI) with licensing
notes. In short, the planned progression is:

dataset ingestion → MIDI parsing → feature extraction → motif embeddings →
motif clustering → a prompt-to-control classifier → an n-gram continuation
baseline → a transformer-based symbolic generator conditioned on key,
tempo, mood, genre, instrument set, and seed motif → evaluation against
motif retention, novelty, harmonic validity, rhythmic coherence, and human
usability.

The `creative_audio_lab.data` package (`dataset_manifest.py`,
`midi_dataset_loader.py`, `preprocess_midi.py`) is scaffolding for that
pipeline today — it never downloads anything automatically.

## Limitations

- Melody, bass, and drum patterns are template- and rule-driven, not
  learned — they're deliberately simple and will sound repetitive over
  many bars compared to a trained model or a human arranger.
- Harmony is diatonic only (no modulation, borrowed chords, or extended/
  altered chords).
- "Sliding bass" and similar expressive articulations are approximated
  with grace notes rather than pitch-bend automation.
- The prompt parser is keyword-based; prompts using vocabulary outside its
  tables fall back to sensible defaults rather than failing, but won't be
  specifically recognized.

## Future work

- Replace/augment each deterministic stage per the ML Roadmap, one stage
  at a time, keeping the same typed interfaces so the app and evaluation
  metrics don't need to change.
- Expand the style-preset and prompt-vocabulary tables with more genres
  and section types.
- Add pitch-bend/CC automation for expressive articulations (slides,
  swells) once a stem-editing workflow exists to consume them.

## Legacy audio utilities

Kept for now, but **not** part of the portfolio story above and not
required for the MIDI generator: `midi/` (raw MIDI note-number helpers),
`separation/` (Demucs/Spleeter stem-separation wrappers), `mix_prep/`
(EQ/compression/stereo-imaging helpers), `human_mix/` (reference-mix
comparison), `mastering/` (loudness normalization), and `upscaling/`
(stem-enhancement wrappers). These require `pip install -e ".[audio]"`
and are unrelated to prompt-to-MIDI generation — see
[`examples/demo_pipeline.ipynb`](examples/demo_pipeline.ipynb) for a
walkthrough if you need them.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/ app.py tests/
```

CI (`.github/workflows/tests.yml`) installs only `.[dev]` (core deps +
pytest/ruff) and runs the test suite — no ML/audio extras required.
