"""Deterministic, rule-based music generators.

Each submodule is a pure function pipeline: :class:`GenerationControls` in,
:class:`~creative_audio_lab.music_theory.Note` events out. No randomness
escapes a generator's local :class:`random.Random` instance, so results are
fully reproducible for a given prompt and control set.
"""
