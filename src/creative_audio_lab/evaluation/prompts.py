"""The fixed held-out prompt list for backend comparison.

Every backend — deterministic, the Stage 2 n-gram baseline, and any future
trained model — is evaluated on exactly these prompts, so metric numbers
are comparable across backends and across time instead of depending on
whatever was typed during development (see docs/STAGE3_PLAN.md, section 4).

These prompts are *held out*: they must never be used as training or
bootstrap material. The n-gram baseline's bootstrap prompts
(:data:`creative_audio_lab.models.ngram_backend.BOOTSTRAP_PROMPTS`) are a
disjoint list, and a test enforces the disjointness.

The list covers every style preset in
:data:`creative_audio_lab.prompt_parser.STYLE_PRESETS`, a spread of moods,
and all three density levels, using the generic descriptors that
docs/DATASETS.md's hard rules require.
"""

from __future__ import annotations

from typing import Tuple

HELD_OUT_PROMPTS: Tuple[str, ...] = (
    "dark orchestral boss battle theme, 140 BPM, brass ostinato, strings",
    "hopeful cinematic trailer theme in D major, 110 BPM",
    "sad orchestral theme in A minor, strings and piano, sparse",
    "aggressive trap melody, 142 BPM, bells and bass",
    "dark drill loop in F minor, busy",
    "mysterious ambient piano loop, 70 BPM, minimal",
    "happy jrpg town theme, flute and strings, 120 BPM",
    "tense fantasy battle theme in E minor, 132 BPM",
    "romantic piano ballad in C major, 85 BPM, simple",
    "nostalgic piano loop in G major, 90 BPM",
    "energetic cinematic theme, taiko and brass, 150 BPM, dense",
    "calm ambient strings in D dorian, 8 bars",
)

__all__ = ["HELD_OUT_PROMPTS"]
