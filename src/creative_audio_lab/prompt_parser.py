"""Deterministic natural-language prompt parser.

Maps free-text prompts such as ``"dark orchestral boss battle theme, 140 BPM,
brass ostinato, strings, piano arps"`` into a structured
:class:`GenerationControls` object that the generators consume. No ML is
involved: this is keyword and regex matching against small, explicit tables,
which keeps behaviour predictable and easy to unit test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


@dataclass
class GenerationControls:
    """Fully-resolved musical controls consumed by the generators."""

    prompt: str
    key: str = "C"
    mode: str = "major"
    bpm: int = 100
    bars: int = 8
    style: str = "cinematic"
    mood: str = "neutral"
    energy: str = "medium"
    density: str = "medium"
    instruments: List[str] = field(default_factory=list)
    section: str = "theme"
    texture: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StylePreset:
    bpm: int
    mode: str
    bars: int
    instruments: Tuple[str, ...]
    progression: Tuple[int, ...]  # scale degrees
    density: str


STYLE_PRESETS: Dict[str, StylePreset] = {
    "cinematic": StylePreset(120, "natural_minor", 16, ("strings", "brass", "piano"), (1, 6, 4, 5), "medium"),
    "orchestral": StylePreset(100, "natural_minor", 16, ("strings", "brass", "piano"), (1, 4, 6, 5), "medium"),
    "trap": StylePreset(140, "natural_minor", 8, ("bells", "bass", "drums"), (1, 6, 7, 1), "high"),
    "drill": StylePreset(142, "natural_minor", 8, ("bells", "bass", "drums"), (1, 7, 6, 7), "high"),
    "ambient": StylePreset(80, "dorian", 8, ("piano", "strings"), (1, 4, 1, 5), "low"),
    "jrpg": StylePreset(128, "major", 16, ("flute", "strings", "piano"), (1, 5, 6, 4), "medium"),
    "ballad": StylePreset(90, "major", 8, ("piano", "strings"), (1, 6, 4, 5), "low"),
    "default": StylePreset(100, "major", 8, ("piano", "strings"), (1, 5, 6, 4), "medium"),
}

DEFAULT_STYLE = "default"

# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------

# mood -> (mood label, preferred mode)
MOOD_KEYWORDS: Dict[str, Tuple[str, str]] = {
    "dark": ("dark", "harmonic_minor"),
    "happy": ("happy", "major"),
    "sad": ("sad", "natural_minor"),
    "hopeful": ("hopeful", "major"),
    "tense": ("tense", "phrygian"),
    "mysterious": ("mysterious", "dorian"),
    "romantic": ("romantic", "major"),
    "aggressive": ("aggressive", "phrygian"),
    "nostalgic": ("nostalgic", "dorian"),
}

# genre/section keyword phrase -> style preset name
GENRE_KEYWORDS: Dict[str, str] = {
    "orchestral": "orchestral",
    "cinematic": "cinematic",
    "trailer": "cinematic",
    "trap": "trap",
    "drill": "drill",
    "ambient": "ambient",
    "jrpg": "jrpg",
    "rpg town": "jrpg",
    "rpg": "jrpg",
    "boss battle": "cinematic",
    "battle theme": "cinematic",
    "fantasy": "jrpg",
    "ballad": "ballad",
    "piano loop": "ballad",
}

# phrase -> section tag
SECTION_KEYWORDS: Dict[str, str] = {
    "boss battle": "boss_battle",
    "battle theme": "battle",
    "town theme": "town",
    "rpg town": "town",
    "village": "town",
    "trailer": "trailer",
    "loop": "loop",
    "verse": "verse",
    "chorus": "chorus",
}

# instrument keyword -> canonical instrument tag
INSTRUMENT_KEYWORDS: Dict[str, str] = {
    "piano": "piano",
    "strings": "strings",
    "brass": "brass",
    "bells": "bells",
    "flute": "flute",
    "drums": "drums",
    "taiko": "taiko_drums",
    "bass": "bass",
}

# texture / arrangement-intent keyword -> tag
TEXTURE_KEYWORDS: Dict[str, str] = {
    "ostinato": "ostinato",
    "arpeggio": "arpeggio",
    "arps": "arpeggio",
    "arp": "arpeggio",
    "sliding bass": "sliding_bass",
    "sliding": "sliding_bass",
}

HIGH_ENERGY_WORDS = {"aggressive", "energetic", "intense", "fast", "boss battle", "battle theme", "drill", "trap"}
LOW_ENERGY_WORDS = {"romantic", "ambient", "calm", "slow", "soft", "ballad", "nostalgic"}

HIGH_DENSITY_WORDS = {"busy", "complex", "dense"}
LOW_DENSITY_WORDS = {"sparse", "simple", "minimal"}

_BPM_RE = re.compile(r"(\d{2,3})\s*bpm")
_BARS_RE = re.compile(r"(\d{1,3})\s*(?:-|\s)?bar")
_KEY_RE = re.compile(
    r"\b([a-g])\s*(#|sharp|b|flat)?\s*"
    r"(major|minor|harmonic minor|dorian|phrygian)\b",
    re.IGNORECASE,
)

_MODE_WORD_TO_MODE = {
    "major": "major",
    "minor": "natural_minor",
    "harmonic minor": "harmonic_minor",
    "dorian": "dorian",
    "phrygian": "phrygian",
}


def _match_phrases(text: str, table: Dict[str, str]) -> List[str]:
    """Return the canonical values of every key in ``table`` found in ``text``.

    Longer phrases are matched first so e.g. ``"boss battle"`` takes priority
    over a lone ``"battle"`` style match.
    """
    found: List[str] = []
    for phrase in sorted(table, key=len, reverse=True):
        if phrase in text and table[phrase] not in found:
            found.append(table[phrase])
    return found


def _parse_key(text: str) -> Optional[Tuple[str, str]]:
    match = _KEY_RE.search(text)
    if not match:
        return None
    letter, accidental, mode_word = match.groups()
    key = letter.upper()
    if accidental in ("#", "sharp"):
        key += "#"
    elif accidental in ("b", "flat"):
        key += "b"
    mode = _MODE_WORD_TO_MODE[mode_word.lower()]
    return key, mode


def parse_prompt(
    prompt: str,
    *,
    key: Optional[str] = None,
    mode: Optional[str] = None,
    bpm: Optional[int] = None,
    bars: Optional[int] = None,
    style: Optional[str] = None,
    energy: Optional[str] = None,
    density: Optional[str] = None,
    instruments: Optional[List[str]] = None,
) -> GenerationControls:
    """Parse ``prompt`` into :class:`GenerationControls`.

    Any explicitly-provided keyword argument (typically coming from UI
    selectors) overrides the value inferred from the text.
    """
    text = prompt.lower()

    genres = _match_phrases(text, GENRE_KEYWORDS)
    resolved_style = style or (genres[0] if genres else DEFAULT_STYLE)
    preset = STYLE_PRESETS.get(resolved_style, STYLE_PRESETS[DEFAULT_STYLE])

    moods = [MOOD_KEYWORDS[word] for word in MOOD_KEYWORDS if word in text]
    resolved_mood = moods[0][0] if moods else "neutral"
    mood_mode = moods[0][1] if moods else None

    sections = _match_phrases(text, SECTION_KEYWORDS)
    resolved_section = sections[0] if sections else "theme"

    parsed_key_mode = _parse_key(text)

    resolved_key = key or (parsed_key_mode[0] if parsed_key_mode else "C")
    resolved_mode = mode or (
        parsed_key_mode[1] if parsed_key_mode else (mood_mode or preset.mode)
    )

    bpm_match = _BPM_RE.search(text)
    resolved_bpm = bpm or (int(bpm_match.group(1)) if bpm_match else preset.bpm)

    bars_match = _BARS_RE.search(text)
    resolved_bars = bars or (int(bars_match.group(1)) if bars_match else preset.bars)

    if energy:
        resolved_energy = energy
    elif any(word in text for word in HIGH_ENERGY_WORDS):
        resolved_energy = "high"
    elif any(word in text for word in LOW_ENERGY_WORDS):
        resolved_energy = "low"
    else:
        resolved_energy = "medium"

    if density:
        resolved_density = density
    elif any(word in text for word in HIGH_DENSITY_WORDS):
        resolved_density = "high"
    elif any(word in text for word in LOW_DENSITY_WORDS):
        resolved_density = "low"
    elif resolved_energy == "high":
        resolved_density = "high"
    elif resolved_energy == "low":
        resolved_density = "low"
    else:
        resolved_density = preset.density

    found_instruments = _match_phrases(text, INSTRUMENT_KEYWORDS)
    resolved_instruments = instruments or found_instruments or list(preset.instruments)

    texture = _match_phrases(text, TEXTURE_KEYWORDS)

    return GenerationControls(
        prompt=prompt,
        key=resolved_key,
        mode=resolved_mode,
        bpm=resolved_bpm,
        bars=resolved_bars,
        style=resolved_style,
        mood=resolved_mood,
        energy=resolved_energy,
        density=resolved_density,
        instruments=resolved_instruments,
        section=resolved_section,
        texture=texture,
    )


__all__ = [
    "GenerationControls",
    "StylePreset",
    "STYLE_PRESETS",
    "parse_prompt",
]
