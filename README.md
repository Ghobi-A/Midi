# Midi

Simple utilities to convert between MIDI note numbers and note names.

## Installation

Install the core dependencies and this package:

```bash
pip install -r requirements.txt
pip install .
```

## Coding

The `midi` package provides a few helper functions:

- `note_to_number(note)` converts a note name like `C4` to a MIDI note number.
- `number_to_note(number)` converts a MIDI note number back to a note name.
- `create_orchestral_midi(layers)` builds a `mido.MidiFile` from multiple
  instrument layers. Each layer can optionally include a MIDI program number
  to set the instrument for that track.

A small command line interface is available via `python -m midi.cli`.

### Examples

```bash
$ python -m midi.cli --note C4
60

$ python -m midi.cli --number 60
C4
```

#### Creating orchestral MIDI

```python
from midi import create_orchestral_midi

layers = {
    "piano": ([(0.0, 60, 1.0, 64)], 0),  # program 0 (Acoustic Grand Piano)
    "strings": [(0.5, 67, 1.5, 64)],
}
mid = create_orchestral_midi(layers)
mid.save("score.mid")
```

#### Selecting a separation backend

```python
from separation import get_separator

separate = get_separator("spleeter")  # or "demucs"
separate("track.wav", "stems/")
```

## Project Structure

Beyond the core `midi` utilities, the repository now includes placeholder packages for a complete audio production pipeline:

- `generation/` – interfaces for AI-assisted audio generation.
  - `suno_wrapper.py` – stub for integrating with the Suno API.
- `separation/` – tools for stem separation.
  - `demucs_wrapper.py` – stub for Demucs-based separation.
  - `spleeter_wrapper.py` – stub for Spleeter-based separation.
  - `__init__.py` – factory exposing `get_separator` to pick a backend.
- `upscaling/` – audio enhancement and upscaling helpers.
  - `voc_upscaler.py` – placeholder vocal enhancer.
  - `guitar_upscaler.py` – placeholder guitar track enhancer.
  - `utils.py` – shared utilities for loading models.
- `mix_prep/` – preparation steps prior to mixing.
  - `mix_prep_wrapper.py` – stub for organizing and cleaning stems.
- `human_mix/` – modules supporting human-in-the-loop mixing.
  - `human_mix_wrapper.py` – stub for manual mixing routines.
- `mastering/` – final mastering stages.
  - `mastering_wrapper.py` – stub for automated mastering.
- `examples/` – demonstration scripts showing how modules fit together.
  - `example_workflow.py` – minimal example pipeline.

These modules are placeholders and will be fleshed out as the project evolves.
