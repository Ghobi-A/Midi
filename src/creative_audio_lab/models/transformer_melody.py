"""Small causal Transformer jointly conditioned on complete previous notes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerConfig:
    pitch_vocab: int
    duration_vocab: int
    velocity_vocab: int
    d_model: int = 192
    layers: int = 4
    heads: int = 6
    feedforward: int = 768
    dropout: float = 0.1
    context_length: int = 128

    def __post_init__(self) -> None:
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")

    def to_dict(self) -> dict:
        return asdict(self)


class TransformerMelodyModel(nn.Module):
    """Decoder-style encoder stack with causal attention and three categorical heads."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.pitch_embedding = nn.Embedding(config.pitch_vocab, config.d_model, padding_idx=0)
        self.duration_embedding = nn.Embedding(config.duration_vocab, config.d_model, padding_idx=0)
        self.velocity_embedding = nn.Embedding(config.velocity_vocab, config.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        layer = nn.TransformerEncoderLayer(config.d_model, config.heads, config.feedforward,
                                           config.dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.layers,
                                                 norm=nn.LayerNorm(config.d_model))
        self.pitch_head = nn.Linear(config.d_model, config.pitch_vocab)
        self.duration_head = nn.Linear(config.d_model, config.duration_vocab)
        self.velocity_head = nn.Linear(config.d_model, config.velocity_vocab)

    def forward(self, pitch, duration, velocity, padding_mask=None):
        if pitch.ndim != 2 or pitch.shape[1] > self.config.context_length:
            raise ValueError("inputs must be [batch, time] within context_length")
        positions = torch.arange(pitch.shape[1], device=pitch.device).unsqueeze(0)
        hidden = (self.pitch_embedding(pitch) + self.duration_embedding(duration)
                  + self.velocity_embedding(velocity) + self.position_embedding(positions))
        causal = torch.triu(torch.ones((pitch.shape[1], pitch.shape[1]), dtype=torch.bool,
                                       device=pitch.device), diagonal=1)
        hidden = self.transformer(hidden, mask=causal, src_key_padding_mask=padding_mask)
        return self.pitch_head(hidden), self.duration_head(hidden), self.velocity_head(hidden)
