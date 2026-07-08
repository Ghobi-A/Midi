"""Piano-roll grid data structure (data only — no image rendering).

A :class:`PianoRoll` is a time-major velocity grid: ``grid[step][row]`` is
the loudest velocity sounding at that time step for the pitch
``pitch_low + row`` (0 = silent). Plain nested tuples keep it trivially
testable and serializable; a UI layer can render it however it likes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..music_theory import Note


@dataclass(frozen=True)
class PianoRoll:
    """A quantized pitch/time velocity grid built from note events."""

    pitch_low: int
    pitch_high: int
    steps_per_beat: int
    grid: Tuple[Tuple[int, ...], ...]  # grid[step][pitch_row] = velocity (0 = silent)

    @property
    def num_steps(self) -> int:
        """Number of time steps (grid rows)."""
        return len(self.grid)

    @property
    def num_pitches(self) -> int:
        """Number of pitch rows spanned (``pitch_high - pitch_low + 1``; 0 when empty)."""
        return (self.pitch_high - self.pitch_low + 1) if self.grid else 0

    @property
    def duration_beats(self) -> float:
        """Total time span covered, in beats."""
        return self.num_steps / self.steps_per_beat

    def active_pitches(self, step: int) -> List[int]:
        """Return the MIDI pitches sounding at ``step`` (ascending)."""
        return [
            self.pitch_low + row
            for row, velocity in enumerate(self.grid[step])
            if velocity > 0
        ]


def build_piano_roll(
    notes: Sequence[Note],
    *,
    steps_per_beat: int = 4,
    pitch_low: Optional[int] = None,
    pitch_high: Optional[int] = None,
) -> PianoRoll:
    """Quantize ``notes`` onto a piano-roll grid.

    Each note occupies every step from its onset up to (but excluding) its
    end, with a minimum of one step; overlapping notes on the same pitch
    keep the loudest velocity. ``pitch_low``/``pitch_high`` default to the
    notes' own pitch extremes. An empty note list yields an empty grid.
    """
    if steps_per_beat < 1:
        raise ValueError("steps_per_beat must be >= 1")
    if not notes:
        return PianoRoll(pitch_low=0, pitch_high=0, steps_per_beat=steps_per_beat, grid=())

    low = pitch_low if pitch_low is not None else min(note.pitch for note in notes)
    high = pitch_high if pitch_high is not None else max(note.pitch for note in notes)
    if low > high:
        raise ValueError(f"pitch_low ({low}) must be <= pitch_high ({high})")

    num_steps = max(round(note.end() * steps_per_beat) for note in notes)
    num_steps = max(num_steps, 1)
    num_pitches = high - low + 1
    cells: List[List[int]] = [[0] * num_pitches for _ in range(num_steps)]

    for note in notes:
        if not low <= note.pitch <= high:
            continue
        start_step = round(note.start * steps_per_beat)
        end_step = max(round(note.end() * steps_per_beat), start_step + 1)
        row = note.pitch - low
        for step in range(start_step, min(end_step, num_steps)):
            cells[step][row] = max(cells[step][row], note.velocity)

    return PianoRoll(
        pitch_low=low,
        pitch_high=high,
        steps_per_beat=steps_per_beat,
        grid=tuple(tuple(step_cells) for step_cells in cells),
    )


__all__ = ["PianoRoll", "build_piano_roll"]
