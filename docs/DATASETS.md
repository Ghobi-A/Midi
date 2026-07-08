# Datasets

This project does not download or train on any dataset today. This document
exists so the eventual ML work described in `ML_ROADMAP.md` starts from a
deliberate, rights-aware plan rather than whatever MIDI happens to be lying
around.

`creative_audio_lab.data` never fetches data automatically:
`dataset_manifest.load_manifest` reads a manifest file you point it at, and
`midi_dataset_loader.load_dataset_notes` reads MIDI files you've already
placed on disk. Neither one reaches out to the network.

Since Stage 1.5, every manifest entry carries a provenance record (source
URL, license, commercial-use and attribution flags, tags), and
`creative_audio_lab.data.provenance.validate_entry` flags entries with
missing licenses or sources, unverified commercial terms, or
fan-archive/unclear-rights tags **before** they can reach any training
pipeline. See `examples/dataset_manifest.example.json` for a template
(placeholder paths only).

## Recommended datasets for future ML work

| Dataset | What it offers | Licensing notes |
| --- | --- | --- |
| **[MAESTRO](https://magenta.tensorflow.org/datasets/maestro)** | ~200 hours of expressive solo piano performance MIDI, tightly aligned to audio. Good for learning expressive timing/dynamics. | CC BY-NC-SA 4.0 — non-commercial use only unless you obtain a separate license. |
| **[POP909](https://github.com/music-x-lab/POP909-Dataset)** | 909 pop songs with melody, chord, and accompaniment-arrangement structure. Good for the melody/chord/arrangement split this project already models. | Research use; underlying songs are copyrighted commercial works. Do not redistribute audio, and confirm terms before any commercial use of derived models. |
| **[GiantMIDI-Piano](https://github.com/bytedance/GiantMIDI-Piano)** | ~10,000 transcribed classical piano pieces. Useful for larger-scale symbolic pretraining. | Transcriptions are largely of public-domain classical works, but verify each source recording's rights before use — the transcription process itself doesn't clear licensing on the underlying audio. |
| **[Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)** | ~176,000 MIDI files matched to the Million Song Dataset; broad genre coverage. | Mixed provenance and heavy duplication. Only usable with careful deduplication, and only for the subset whose licensing has actually been checked — do not train on it in bulk without review. |
| **[MidiCaps](https://huggingface.co/datasets/amaai-lab/MidiCaps)** | ~168k MIDI files paired with rich text captions — the natural training pair for text-conditioned symbolic generation (Text2midi was trained on it). | Captions/metadata are CC BY-SA 4.0, but the MIDI itself comes from Lakh, so Lakh's mixed-provenance caveats apply to the note content. |
| **[MetaScore (public subset)](https://github.com/salu133445/metascore)** | MuseScore-derived scores with genre tags and user descriptions; the public subset is restricted to permissively licensed scores. | Rights vary per score — use only the CC-licensed public subset and record each score's individual license. |
| **[Slakh2100](http://www.slakh.com/)** | 2100 multi-track MIDI arrangements with professionally synthesized audio stems; good for multi-instrument arrangement modelling. | CC BY 4.0 for the redistributed set — attribution required. |
| **[ComMU](https://github.com/POZAlabs/ComMU-code)** | ~11k short MIDI samples by professional composers, each with 12 metadata fields (BPM, key, chord progression, track role, ...) — very close to this project's `GenerationControls` framing. | CC BY-NC 4.0 — non-commercial only. |
| **[EMOPIA](https://annahung31.github.io/EMOPIA/)** | ~1k pop-piano clips with valence/arousal emotion labels; useful for mood conditioning. | CC BY-NC-SA 4.0, and the clips are transcriptions of copyrighted pop covers — non-commercial, underlying works remain copyrighted. |
| **[Aria-MIDI](https://github.com/EleutherAI/aria)** | ~1M piano MIDI transcriptions from public performance recordings; large-scale symbolic pretraining material. | Verify the release license at source before use — transcription does not clear rights on the underlying recordings; treat commercial use as unresolved. |
| **[music21 corpus](https://www.music21.org/music21docs/about/referenceCorpus.html)** | Curated symbolic corpus bundled with music21 (Bach chorales, folk songs, ...); small, clean, mostly public domain. | music21 itself is BSD; corpus works are mostly public domain but a few carry their own restrictions — check per work. |

These are also registered (with the same provenance fields) in
`creative_audio_lab.data.dataset_manifest.RECOMMENDED_DATASETS`.

## Hard rules

- **Do not train on random copyrighted game or anime MIDI packs** scraped
  from fan sites without checking rights. "I found it on a forum" is not a
  license.
- **Fan OST / game MIDI archives are private reference and evaluation
  material only** — listening, analysis, sanity-checking metrics — unless
  they are properly licensed for training. Tag such entries
  `"fan-archive"` (or `"unclear-rights"`) in the manifest;
  `provenance.validate_entry` will refuse to consider them training-ready.
- **Do not claim style cloning of living artists or copyrighted
  franchises.** This project's prompt vocabulary (see `prompt_parser.py`)
  deliberately uses generic descriptors instead of naming artists or IP.
- **Prefer generic descriptors** in prompts and in any future training
  labels — e.g. *"JRPG boss battle"*, *"cinematic orchestral"*, *"dark trap
  melody"*, or *"romantic piano theme"* — rather than naming a specific
  game, franchise, or artist.

## Adding a dataset locally

1. Source the MIDI files yourself and place them in a local directory —
   nothing here fetches them for you.
2. Describe them with a manifest (see `DatasetEntry` in
   `creative_audio_lab/data/dataset_manifest.py`, and
   `examples/dataset_manifest.example.json` for the JSON shape): name,
   source URL, license, commercial-use/attribution flags, notes, and tags.
3. Run the entries through
   `creative_audio_lab.data.provenance.validate_manifest` and resolve every
   error (and ideally every warning) before using them for anything
   training-related.
4. Use `creative_audio_lab.data.midi_dataset_loader.load_dataset_notes` to
   parse the directory into `Note` events for feature extraction.
