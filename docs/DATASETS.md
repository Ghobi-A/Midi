# Datasets

This project does not download or train on any dataset today. This document
exists so the eventual ML work described in `ML_ROADMAP.md` starts from a
deliberate, rights-aware plan rather than whatever MIDI happens to be lying
around.

`creative_audio_lab.data` never fetches data automatically:
`dataset_manifest.load_manifest` reads a manifest file you point it at, and
`midi_dataset_loader.load_dataset_notes` reads MIDI files you've already
placed on disk. Neither one reaches out to the network.

## Recommended datasets for future ML work

| Dataset | What it offers | Licensing notes |
| --- | --- | --- |
| **[MAESTRO](https://magenta.tensorflow.org/datasets/maestro)** | ~200 hours of expressive solo piano performance MIDI, tightly aligned to audio. Good for learning expressive timing/dynamics. | CC BY-NC-SA 4.0 — non-commercial use only unless you obtain a separate license. |
| **[POP909](https://github.com/music-x-lab/POP909-Dataset)** | 909 pop songs with melody, chord, and accompaniment-arrangement structure. Good for the melody/chord/arrangement split this project already models. | Research use; underlying songs are copyrighted commercial works. Do not redistribute audio, and confirm terms before any commercial use of derived models. |
| **[GiantMIDI-Piano](https://github.com/bytedance/GiantMIDI-Piano)** | ~10,000 transcribed classical piano pieces. Useful for larger-scale symbolic pretraining. | Transcriptions are largely of public-domain classical works, but verify each source recording's rights before use — the transcription process itself doesn't clear licensing on the underlying audio. |
| **[Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)** | ~176,000 MIDI files matched to the Million Song Dataset; broad genre coverage. | Mixed provenance and heavy duplication. Only usable with careful deduplication, and only for the subset whose licensing has actually been checked — do not train on it in bulk without review. |

## Hard rules

- **Do not train on random copyrighted game or anime MIDI packs** scraped
  from fan sites without checking rights. "I found it on a forum" is not a
  license.
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
   `creative_audio_lab/data/dataset_manifest.py`) noting the source, URL,
   and license notes above.
3. Use `creative_audio_lab.data.midi_dataset_loader.load_dataset_notes` to
   parse the directory into `Note` events for feature extraction.
