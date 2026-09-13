"""Leakage-free note-event vocabulary for the neural melody baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple

from ..music_theory import Note
from ..tokenization import TokenizerConfig
from .note_event_model import split_note_stream

PAD, BOS, EOS = 0, 1, 2
VALUE_OFFSET = 3


@dataclass(frozen=True)
class NeuralNoteEvent:
    pitch: int
    duration: int
    velocity: int


class NoteEventCodec:
    """Fixed vocabularies defined by MIDI and tokenizer bounds, never corpus frequencies."""

    def __init__(self, tokenizer: TokenizerConfig | None = None) -> None:
        self.tokenizer = tokenizer or TokenizerConfig()
        self.pitch_size = VALUE_OFFSET + 128
        self.duration_size = VALUE_OFFSET + max(
            round(self.tokenizer.max_duration_beats * self.tokenizer.positions_per_beat), 1
        )
        self.velocity_size = VALUE_OFFSET + (128 + self.tokenizer.velocity_bin_width - 1) // self.tokenizer.velocity_bin_width

    @property
    def vocab_sizes(self) -> Tuple[int, int, int]:
        return self.pitch_size, self.duration_size, self.velocity_size

    def to_dict(self) -> dict:
        return {"format": "creative-audio-lab.neural-note-events/1", "tokenizer": asdict(self.tokenizer)}

    @classmethod
    def from_dict(cls, data: dict) -> "NoteEventCodec":
        if data.get("format") != "creative-audio-lab.neural-note-events/1":
            raise ValueError("Unsupported neural note-event format")
        return cls(TokenizerConfig(**data["tokenizer"]))

    def encode_stream(self, stream: Sequence[str], boundaries: bool = True) -> List[NeuralNoteEvent]:
        pitches, velocities, durations = split_note_stream(stream)
        events = [self._encode_values(int(p.rsplit("_", 1)[1]), int(d.rsplit("_", 1)[1]),
                                      int(v.rsplit("_", 1)[1]))
                  for p, v, d in zip(pitches, velocities, durations)]
        boundary = NeuralNoteEvent(BOS, BOS, BOS)
        end = NeuralNoteEvent(EOS, EOS, EOS)
        return ([boundary] + events + [end]) if boundaries else events

    def encode_notes(self, notes: Sequence[Note], boundaries: bool = True) -> List[NeuralNoteEvent]:
        events = []
        for note in sorted(notes, key=lambda n: (n.start, n.pitch)):
            duration = min(max(round(note.duration * self.tokenizer.positions_per_beat), 1),
                           self.duration_size - VALUE_OFFSET)
            velocity = min(max(note.velocity, 0), 127) // self.tokenizer.velocity_bin_width
            events.append(self._encode_values(note.pitch, duration, velocity))
        if boundaries:
            return [NeuralNoteEvent(BOS, BOS, BOS), *events, NeuralNoteEvent(EOS, EOS, EOS)]
        return events

    def _encode_values(self, pitch: int, duration: int, velocity: int) -> NeuralNoteEvent:
        if not 0 <= pitch <= 127:
            raise ValueError(f"pitch outside MIDI range: {pitch}")
        if not 1 <= duration <= self.duration_size - VALUE_OFFSET:
            raise ValueError(f"duration outside tokenizer range: {duration}")
        if not 0 <= velocity < self.velocity_size - VALUE_OFFSET:
            raise ValueError(f"velocity bin outside tokenizer range: {velocity}")
        return NeuralNoteEvent(pitch + VALUE_OFFSET, duration - 1 + VALUE_OFFSET,
                               velocity + VALUE_OFFSET)

    def decode_event(self, event: NeuralNoteEvent, start: float = 0.0) -> Note:
        if min(event.pitch, event.duration, event.velocity) < VALUE_OFFSET:
            raise ValueError("Cannot decode PAD/BOS/EOS as a note")
        pitch = event.pitch - VALUE_OFFSET
        duration_steps = event.duration - VALUE_OFFSET + 1
        velocity_bin = event.velocity - VALUE_OFFSET
        self._encode_values(pitch, duration_steps, velocity_bin)
        return Note(start=start, pitch=pitch,
                    duration=duration_steps / self.tokenizer.positions_per_beat,
                    velocity=min(127, velocity_bin * self.tokenizer.velocity_bin_width))


def pad_event_sequences(sequences: Sequence[Sequence[NeuralNoteEvent]]):
    """Return padded attribute lists and a True-for-real-event attention mask."""
    width = max((len(sequence) for sequence in sequences), default=0)
    pad = NeuralNoteEvent(PAD, PAD, PAD)
    padded = [list(sequence) + [pad] * (width - len(sequence)) for sequence in sequences]
    mask = [[index < len(sequence) for index in range(width)] for sequence in sequences]
    return padded, mask
