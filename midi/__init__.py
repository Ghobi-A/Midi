"""Minimal MIDI utilities."""

# Mapping from note names to their semitone index within an octave.  Both
# sharps and flats are supported and note names are normalized to uppercase
# by :func:`note_to_number`.
NOTE_NAMES = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}

# Canonical sharp and flat representations for mapping a semitone index back to
# a note name.  :func:`number_to_note` defaults to the sharp names but can
# optionally return the flat equivalents.
NOTE_NAMES_SHARP = [
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


def note_to_number(note: str) -> int:
    """Convert a note name like 'C4' or 'Db4' to a MIDI note number."""
    note = note.strip().upper()
    if len(note) < 2:
        raise ValueError("Invalid note format")
    if note[-1].isdigit():
        octave = int(note[-1])
        name = note[:-1]
    else:
        raise ValueError("Note must end with octave number")
    if name not in NOTE_NAMES:
        raise ValueError(f"Invalid note name: {name}")
    return NOTE_NAMES[name] + (octave + 1) * 12


def number_to_note(number: int, prefer_sharps: bool = True) -> str:
    """Convert a MIDI note number to its note name.

    Parameters
    ----------
    number:
        MIDI note number (0-127).
    prefer_sharps:
        If ``True`` (default) use sharp notation, otherwise use flats.
    """
    if not (0 <= number <= 127):
        raise ValueError("MIDI note number must be between 0 and 127")
    octave, index = divmod(number, 12)
    names = NOTE_NAMES_SHARP if prefer_sharps else NOTE_NAMES_FLAT
    return f"{names[index]}{octave - 1}"


from .orchestra import create_orchestral_midi, NoteEvent


__all__ = [
    "note_to_number",
    "number_to_note",
    "create_orchestral_midi",
    "NoteEvent",
]
