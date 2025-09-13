"""Minimal MIDI utilities."""

# Standard note names using sharps for indexing
NOTE_NAMES = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]

# Equivalent note names expressed with flats. These are used for output so
# that either form can round-trip through the conversion functions.
NOTE_NAMES_FLAT = [
    "C",
    "Db",
    "D",
    "Eb",
    "E",
    "F",
    "Gb",
    "G",
    "Ab",
    "A",
    "Bb",
    "B",
]

# Mapping from any accepted note name (sharp or flat) to its index within an
# octave. The keys are normalised to have a capital letter followed by any
# accidental.
NOTE_NAME_TO_INDEX = {name: i for i, name in enumerate(NOTE_NAMES)}
NOTE_NAME_TO_INDEX.update({name: i for i, name in enumerate(NOTE_NAMES_FLAT)})


def note_to_number(note: str) -> int:
    """Convert a note name like 'C4' to a MIDI note number."""
    note = note.strip()
    if len(note) < 2:
        raise ValueError("Invalid note format")
    if note[-1].isdigit():
        octave = int(note[-1])
        name = note[:-1]
    else:
        raise ValueError("Note must end with octave number")
    name = name[0].upper() + name[1:].lower()
    if name not in NOTE_NAME_TO_INDEX:
        raise ValueError(f"Invalid note name: {name}")
    return NOTE_NAME_TO_INDEX[name] + (octave + 1) * 12


def number_to_note(number: int) -> str:
    """Convert a MIDI note number to its note name."""
    if not (0 <= number <= 127):
        raise ValueError("MIDI note number must be between 0 and 127")
    octave, index = divmod(number, 12)
    return f"{NOTE_NAMES_FLAT[index]}{octave - 1}"


from .orchestra import create_orchestral_midi, NoteEvent


__all__ = [
    "note_to_number",
    "number_to_note",
    "create_orchestral_midi",
    "NoteEvent",
]
