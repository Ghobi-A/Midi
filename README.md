# Midi

Simple utilities to convert between MIDI note numbers and note names.

## Installation

Install the core dependencies and this package.  The included
`requirements.txt` bundles optional libraries used by the
demonstration modules (e.g. the Suno client and Demucs separator):

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

## Project Structure

Beyond the core `midi` utilities, the repository now includes modules for a
complete audio production pipeline:

- `generation/` – interfaces for AI-assisted audio generation.
  - `suno_client.py` – minimal client for the [Suno API](https://docs.suno.ai).
- `separation/` – tools for stem separation.
  - `demucs_separator.py` – wrapper around [Demucs](https://github.com/facebookresearch/demucs).
- `upscaling/` – audio enhancement and upscaling helpers.
  - `voc_upscaler.py` – simple vocal upscaler built on neural vocoders (see [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)).
- `mix_prep/` – preparation steps prior to mixing.
  - `mix_prep.py` – organizes and cleans stems before the mix stage; see the [example workflow](examples/example_workflow.py).
- `human_mix/` – modules supporting human-in-the-loop mixing.
  - `human_mix.py` – utilities for manual or assisted mixing sessions (also showcased in the [example workflow](examples/example_workflow.py)).
- `mastering/` – final mastering stages.
  - `mastering_engine.py` – basic loudness and EQ mastering routines, used at the end of the [example workflow](examples/example_workflow.py).
- `examples/` – demonstration scripts showing how modules fit together.
  - `example_workflow.py` – minimal end-to-end pipeline example.

Each module is self-contained and can be invoked directly.  For instance the
Suno client can generate a short track with:

```bash
python -m generation.suno_client output.wav --prompt "lofi beat"
```

See the individual source files or linked projects for more detailed
documentation.
