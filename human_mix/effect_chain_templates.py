"""Reusable effect-chain presets for common mixing tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import copy


@dataclass
class Effect:
    """Representation of a single audio effect."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


EffectChain = List[Effect]


ROCK_LEAD_VOX: EffectChain = [
    Effect("highpass", {"freq": 100}),
    Effect("compressor", {"ratio": 4, "attack": 10, "release": 50}),
    Effect("de-esser", {"freq": 7000}),
    Effect("eq", {"freq": 3000, "gain": 3}),
    Effect("reverb", {"room_size": 0.35, "wet": 0.2}),
]

TEMPLATES: Dict[str, EffectChain] = {"ROCK_LEAD_VOX": ROCK_LEAD_VOX}


def get_template(name: str) -> EffectChain:
    """Return a deep copy of a named effect chain template."""
    if name not in TEMPLATES:
        raise KeyError(f"Unknown template: {name}")
    return copy.deepcopy(TEMPLATES[name])
