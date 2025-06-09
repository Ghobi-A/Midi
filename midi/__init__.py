"""Minimal MIDI utilities."""

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


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
    if name not in NOTE_NAMES:
        raise ValueError(f"Invalid note name: {name}")
    return NOTE_NAMES.index(name) + (octave + 1) * 12


def number_to_note(number: int) -> str:
    """Convert a MIDI note number to its note name."""
    if not (0 <= number <= 127):
        raise ValueError("MIDI note number must be between 0 and 127")
    octave, index = divmod(number, 12)
    return f"{NOTE_NAMES[index]}{octave - 1}"


from .orchestra import create_orchestral_midi, NoteEvent


__all__ = [
    "note_to_number",
    "number_to_note",
    "create_orchestral_midi",
    "NoteEvent",
]
