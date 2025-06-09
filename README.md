# Midi

Simple utilities to convert between MIDI note numbers and note names.

## Coding

The `midi` package provides two helper functions:

- `note_to_number(note)` converts a note name like `C4` to a MIDI note number.
- `number_to_note(number)` converts a MIDI note number back to a note name.

A small command line interface is available via `python -m midi.cli`.

### Examples

```bash
$ python -m midi.cli --note C4
60

$ python -m midi.cli --number 60
C4
```

