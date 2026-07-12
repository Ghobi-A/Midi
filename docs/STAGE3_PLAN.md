# Stage 3 Plan — decisions before training on a real corpus

This document resolves the judgment calls that have to be settled *before*
Stage 3 (training a model on a real MIDI corpus) begins: which dataset, at
what tokenization granularity, evaluated how, and in what order. A wrong
choice here wastes an entire training run, or worse, quietly produces a
model overfit to a narrow style that the evaluation suite can't detect.

Everything below is a **decision, not implemented work**. Where a
recommendation depends on code, it cites the module that actually exists.

## 1. Where the repo actually stands

- **Stage 1 (implemented):** the deterministic rule-based generator
  (`creative_audio_lab.prompt_parser` + `generators`).
- **Stage 1.5 (implemented):** ML scaffolding — the internal REMI-style
  tokenizer (`creative_audio_lab.tokenization`), the dataset provenance
  registry (`creative_audio_lab.data`), the `GenerationBackend` interface
  (`creative_audio_lab.models`), and the preview layer.
- **Stage 2 (not built):** the Markov/n-gram continuation baseline is step 7
  of [`ML_ROADMAP.md`](ML_ROADMAP.md)'s planned progression. No statistical
  or learned backend exists in the repo today — `creative_audio_lab.models`
  contains the deterministic backend and two placeholder adapters that raise
  `NotImplementedError`. The staged plan in section 5 treats the n-gram
  baseline as the first gate, because it exercises the whole
  tokenize→train→generate→evaluate loop at near-zero compute cost.
- **Evaluation (implemented, but scoped):**
  `creative_audio_lab.evaluation.metrics` was built as a corpus-free,
  model-free statistics suite (its own docstring: "None of these require a
  trained model"). It can score any arrangement, but it was never designed
  to compare a trained model against a corpus. Section 4 covers what's
  missing.

## 2. Dataset decision

The question is not "which is the best MIDI dataset" in the abstract — it's
which dataset can teach the controls this project already exposes.
`prompt_parser.STYLE_PRESETS` defines seven real presets (cinematic,
orchestral, trap, drill, ambient, jrpg, ballad), `MOOD_KEYWORDS` nine moods,
and every arrangement is a four-part texture (melody / chords / bass /
drums). Measured against that:

| Candidate | Stylistic coverage of the preset table | Rights status under this repo's rules | Verdict |
| --- | --- | --- | --- |
| **MAESTRO** | Solo classical piano performance. At a stretch: `ballad`/`ambient` piano. Nothing for trap, drill, jrpg, orchestral, or any multi-instrument texture. Unquantized expressive timing (see §3). | Passes `provenance.validate_entry` (license recorded, CC BY-NC-SA). | Rejected as primary — too narrow for the prompt vocabulary. |
| **GiantMIDI-Piano** | Classical piano only; same coverage gap as MAESTRO, plus transcription noise. | Registry entry has `license="unknown"`, so `license_policy.check_entry_license` flags it until a human verifies the terms. | Rejected. |
| **Filtered Lakh subset** | The only candidate with real genre breadth — plausibly the only one that could touch trap/drill-adjacent or orchestral material. | Registry entry is tagged `"unclear-rights"`, so `provenance.validate_entry` errors and `data.provenance.assert_training_ready` refuses it outright. A per-file rights-checked subset would be a large curation project of its own. | Rejected for Stage 3 by the repo's own validation code. Revisit later only with per-item license checks. |
| **POP909** | Pop songs with an explicit melody / chord / accompaniment split — structurally the closest match to this project's part layout. Stylistically: pop ballads, i.e. the `ballad` preset and the softer moods (romantic, nostalgic, sad). Nothing for trap/drill/jrpg/orchestral. | `license="research-only"` (a recognized id in `license_policy.KNOWN_LICENSES`, non-commercial); underlying songs are copyrighted commercial works. Passes `validate_entry` for research use. | **Recommended pilot corpus**, scoped to the `ballad` preset. |

**The honest headline: no candidate covers the prompt vocabulary.** The
trap, drill, and jrpg presets describe material that, in practice, exists as
fan-made game/anime MIDI archives — exactly what
[`DATASETS.md`](DATASETS.md)'s hard rules and the `UNCLEAR_RIGHTS_TAGS`
check in `provenance.py` exclude from training. That is not a temporary
inconvenience; it is a structural constraint on this project.

The consequence for Stage 3:

- The trained backend launches **style-scoped**. Presets without a
  rights-clean corpus (trap, drill, jrpg, cinematic, orchestral) stay on
  `DeterministicBackend`. The `GenerationBackend` abstraction already lets
  the app route per request, so a partially-learned system is an ordinary
  configuration, not a hack.
- **Primary pilot corpus: POP909**, restricted to the `ballad` preset. Its
  melody/chord/accompaniment split maps directly onto the parts this
  project generates, which makes melody-only extraction (section 5)
  trivial rather than a heuristic.
- **Named alternative: ComMU** (already in
  `dataset_manifest.RECOMMENDED_DATASETS`, CC BY-NC 4.0). ~11k short,
  quantized samples, each with 12 metadata fields (BPM, key, chord
  progression, track role) nearly isomorphic to `GenerationControls`. It is
  the better corpus for testing *conditioning* specifically; POP909 is the
  better corpus for learning *melody over harmony*. If the pilot's main risk
  turns out to be conditioning rather than melodic quality, swap to ComMU.
- **Stage 3b candidate: Slakh2100** (CC BY 4.0, attribution) — the only
  multi-track option in the registry whose redistributed set is commercially
  clean. Irrelevant to the melody-only pilot; noted here so multi-instrument
  work later doesn't restart the dataset debate.

## 3. Tokenization decision

The internal `SymbolicTokenizer` was built for Stage 1.5 and, per its module
docstring, everything the deterministic generators emit sits exactly on its
grid and round-trips losslessly. Real corpus MIDI does not have that
property. Concretely, on real files the current implementation
(`tokenization/symbolic_tokenizer.py`):

1. **Emits an open vocabulary for bars.** `encode` writes `BAR_<index>` with
   the *absolute* bar number. Synthetic data is 8–16 bars, so the issue is
   invisible today; a 300-bar corpus piece emits `BAR_299`, a token a model
   trained on shorter pieces has never seen, and the vocabulary grows
   without bound. **This must change before any training run** — to a
   single bar-advance token (the actual REMI convention) or
   segment-relative bar indices. This is the one mandatory pre-pilot
   tokenizer fix.
2. **Cannot represent tempo or meter changes.** `TEMPO` and `TIME_SIGNATURE`
   are single optional header tokens; `encode` accepts one scalar of each
   and `decode` keeps whichever it saw last. Mid-piece tempo maps
   (ubiquitous in performance MIDI, present in pop MIDI) are silently
   unrepresentable. The pilot mitigates by corpus choice — quantized,
   fixed-meter segments — and inline tempo/meter tokens are deferred to
   Stage 3b rather than speculatively added now.
3. **Assumes 4/4.** `TokenizerConfig.beats_per_bar` defaults to 4.0 and
   nothing ever varies it, so 3/4 or 6/8 files get wrong bar/position
   anchors. Handled as a preprocessing *filter* (keep 4/4 segments only),
   not a tokenizer change.
4. **Quantizes to a 16th grid.** `positions_per_beat=4` rounds every onset
   and duration; triplets and swing collapse onto the grid. Fine for
   quantized score-like data, lossy on performance MIDI. For corpus work,
   raise to `positions_per_beat=12` (divisible by 3 and 4, so both straight
   16ths and triplets survive) via the existing `TokenizerConfig` — and
   validate the choice by *measuring* round-trip loss on the actual corpus
   (encode→decode, count moved/merged notes), not by assuming.
5. **Has no drum or track awareness.** One optional `PROGRAM` header covers
   the whole sequence, and nothing distinguishes channel-10 percussion, so a
   kick drum (pitch 36) would be modeled as a C2 bass note. The melody-only
   pilot sidesteps this by excluding drum tracks in preprocessing;
   multi-track token schemes are a Stage 3b question.

Decision: **keep the internal tokenizer as the single training vocabulary**,
apply fix (1), adopt the config change (4), and push (2), (3), (5) into
preprocessing filters for the pilot. The already-scaffolded
`MidiTokRemiAdapter` is used as a *cross-check* — tokenize the same files
with reference REMI and compare sequence lengths and vocab usage — not as a
second training vocabulary, so the project keeps one canonical token stream
end to end.

## 4. Evaluation additions

`evaluation/metrics.py` (note density, pitch range, repetition, harmonic
fit, rhythmic complexity, novelty, motif retention, scale adherence) can
score any single arrangement. What it cannot do is tell you whether a
trained model is *good*, because it has no notion of a reference corpus, no
notion of whether the output matched the request, and no protection against
the flattering failure mode — a model that copies its training data scores
beautifully on every current metric.

Smallest additions that close the gap, in the same dependency-light style:

- **Corpus-distribution similarity.** Compare a *set* of generated melodies
  against a held-out corpus split on pitch-class, melodic-interval, and
  duration distributions (one small histogram-distance helper; the
  interval machinery already exists in `motif_detection.extract_intervals`
  and the entropy/counter plumbing in `metrics._normalized_entropy`).
  This is what "sounds like the corpus, statistically" means here.
- **Controls adherence.** One function taking an `Arrangement` and scoring
  its output against its *own* `controls`: scale adherence against the
  requested key/mode (already computable via `scale_adherence_score` +
  `scale_pitch_classes`), and measured note density against the requested
  density band. This is the style-conditioning check: did "sad ballad, low
  density" actually come out sparse and minor.
- **Training-set plagiarism check.** Reuse the existing
  `motif_retention_score` with the roles swapped: score generated output
  against its nearest training pieces. High retention against training data
  = memorization. For POP909 — whose underlying songs are copyrighted —
  this doubles as a rights guardrail, not just a quality metric.
- **A fixed held-out prompt list**, checked into the repo as data, so the
  deterministic backend, the Stage 2 n-gram baseline, and any trained model
  are always compared on *identical* prompts rather than whatever was typed
  during development.

What deliberately does **not** go in `metrics.py`: held-out token
perplexity/NLL. It requires a trained model, and the module's documented
contract is that nothing in it does. Perplexity belongs with the future
training code, reported alongside these metrics rather than inside them.

## 5. Staged plan

Each stage is a gate: the next one starts only if the previous one's
"working" criteria hold.

**Stage 2 — n-gram baseline (prerequisite, currently unbuilt).**
An n-gram melody-continuation backend behind `GenerationBackend`, trained on
output sampled from the deterministic generator itself. Its purpose is not
musical quality — synthetic-on-synthetic statistics can't exceed their
source. Its purpose is to exercise tokenizer → training data → sampling →
`evaluate_arrangement` end to end at near-zero compute, and to produce the
held-out NLL number any neural model must beat. If the eval harness can't
cleanly compare deterministic vs n-gram, that's a bug found for free instead
of during a GPU run.

**Stage 3a — smallest viable trained model.**
- **Scope:** melody only, single style (`ballad`), POP909 melody tracks.
  The deterministic chords/bass/drums accompany the generated melody
  through the existing arrangement pipeline — so `harmonic_fit_score`
  stays meaningful, and the app output remains a complete arrangement.
- **Model:** a small decoder-only transformer (single-digit millions of
  parameters) over the internal token vocabulary, conditioned on a handful
  of control tokens (mode, density). An LSTM is an acceptable fallback if
  the transformer is fiddly at this scale; the point is the pipeline, not
  the architecture.
- **This validates:** the bar-token fix, the 12-steps-per-beat config, the
  preprocessing filters, the new metrics, and the backend integration —
  before any multi-genre ambition spends real compute.

**"This approach is working" means, on the fixed held-out prompt list:**
- held-out NLL clearly better than the Stage 2 n-gram baseline;
- corpus-distribution similarity within the band that held-out *corpus*
  pieces score against each other (the corpus-vs-corpus spread is the
  yardstick, measured before training);
- scale adherence and harmonic fit at or above the deterministic baseline
  *without* hard-constraining the sampler;
- conditioning check: flipping the mode/density condition measurably shifts
  the output distributions in the right direction.

**"This approach is broken" means:**
- the plagiarism check shows generated melodies are near-copies of training
  pieces;
- scale adherence sits near the unconditioned-random floor;
- conditioning tokens have no measurable effect on output;
- or tokenizer round-trip on corpus data loses/moves more notes than the
  threshold set when the corpus is first tokenized.

Thresholds get written down (in this document or the pilot's config) *before*
the first training run. Hitting a "broken" criterion means fixing the
tokenizer, data, or conditioning scheme — not scaling the model up in the
hope that size fixes it.

**Stage 3b — only after 3a passes:** multi-style training where corpora
exist, multi-track generation (Slakh2100), inline tempo/meter tokens, and
revisiting whether the prompt-to-control classifier (roadmap step 6) should
precede or follow it.

## 6. Licensing implications

- **POP909 (research-only) and ComMU (CC BY-NC 4.0):** both pass
  `provenance.validate_entry` for research use (license and source recorded,
  commercial flags set explicitly in `RECOMMENDED_DATASETS`), and both are
  non-commercial under `license_policy.KNOWN_LICENSES`. Consequence: **any
  weights trained on them are non-commercial artifacts.** The repo's code is
  MIT-licensed; the code license does not and cannot extend to model weights
  derived from NC data. If a trained backend ships, its weights carry their
  own restriction, must not be presented as MIT, and cannot be the default
  path for any commercial use of the app. This needs to be stated wherever
  the weights are distributed.
- **POP909 specifically:** the underlying songs are copyrighted commercial
  works. The plagiarism metric in section 4 is the operational guardrail —
  memorized output would reproduce copyrighted melodies, which is a rights
  problem before it is a quality problem.
- **Lakh and GiantMIDI-Piano** are blocked by the repo's own code today
  (`"unclear-rights"` tag → `assert_training_ready` raises; `"unknown"`
  license → policy violation until verified). No process change needed —
  the existing validation already enforces the right answer.

## 7. What would make this plan wrong

- **POP909 may be too narrow even for its own preset.** If ballad-scoped
  training can't generalize past its ~900 songs, the corpus-similarity band
  will be unreachable. Mitigation is already named: swap to ComMU, whose
  metadata matches the project's conditioning needs more directly.
- **The 16th/12-step grid assumption could be wrong even for quantized
  pop.** That's why round-trip loss is a measured gate, not an assumption —
  if measured loss is high, the grid decision reopens before training, not
  after.
- **Melody-only may under-test the tokenizer.** Bass and accompaniment
  tracks have different register, density, and overlap characteristics; a
  tokenizer that survives melodies could still fail on them in Stage 3b.
  Accepted knowingly — the pilot's job is to validate the loop, not every
  future input.
- **The n-gram bar may be too low.** A baseline trained on synthetic data
  from the deterministic generator could be so weak that "beats the
  baseline" means little. If Stage 3a clears it trivially while failing the
  corpus-similarity band, the similarity band is the binding criterion, not
  the NLL comparison.
- **MetaScore's public subset might dominate POP909** (multi-genre, per-item
  CC licensing, genre tags) once actually inspected. It stays in the
  registry; if a pilot-scale inspection shows enough clean 4/4 material in
  relevant genres, the dataset decision reopens with this document's same
  criteria.
