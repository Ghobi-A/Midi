import inspect

import pytest

from creative_audio_lab.generators.arrangement import Arrangement, build_arrangement
from creative_audio_lab.models import (
    BACKEND_REGISTRY,
    DEFAULT_BACKEND_NAME,
    BackendInfo,
    DeterministicBackend,
    GenerationBackend,
    MidiLlmAdapter,
    Text2MidiAdapter,
    get_backend,
    list_backends,
)
from creative_audio_lab.prompt_parser import parse_prompt

PROMPT = "dark orchestral boss battle theme, 140 BPM, brass ostinato, strings, piano arps"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_all_registered_backends():
    assert set(BACKEND_REGISTRY) == {"deterministic", "ngram", "text2midi", "midi_llm"}


def test_default_backend_is_deterministic():
    assert DEFAULT_BACKEND_NAME == "deterministic"
    assert isinstance(get_backend(), DeterministicBackend)


def test_get_backend_unknown_name_lists_valid_names():
    with pytest.raises(KeyError, match="deterministic"):
        get_backend("suno_clone")


def test_list_backends_reports_availability():
    infos = {info.name: info for info in list_backends()}
    assert all(isinstance(info, BackendInfo) for info in infos.values())
    assert infos["deterministic"].available is True
    assert infos["ngram"].available is True
    assert infos["text2midi"].available is False
    assert infos["midi_llm"].available is False


# ---------------------------------------------------------------------------
# Interface shape
# ---------------------------------------------------------------------------


def test_every_backend_implements_the_generate_interface():
    expected_params = {"prompt", "key", "mode", "bpm", "bars", "style", "energy", "density", "instruments"}
    for backend_cls in BACKEND_REGISTRY.values():
        assert issubclass(backend_cls, GenerationBackend)
        assert backend_cls.name and backend_cls.display_name and backend_cls.description
        params = set(inspect.signature(backend_cls.generate).parameters) - {"self"}
        assert params == expected_params


def test_backend_cannot_be_instantiated_without_generate():
    with pytest.raises(TypeError):
        GenerationBackend()  # abstract


# ---------------------------------------------------------------------------
# Deterministic backend
# ---------------------------------------------------------------------------


def test_deterministic_backend_matches_direct_pipeline():
    arrangement = DeterministicBackend().generate(PROMPT)
    direct = build_arrangement(parse_prompt(PROMPT))
    assert isinstance(arrangement, Arrangement)
    assert arrangement.parts == direct.parts
    assert arrangement.programs == direct.programs
    assert arrangement.bpm == direct.bpm


def test_deterministic_backend_honours_overrides():
    arrangement = DeterministicBackend().generate(PROMPT, bpm=97, bars=4, key="D")
    assert arrangement.bpm == 97
    assert arrangement.bars == 4
    assert arrangement.controls.key == "D"


# ---------------------------------------------------------------------------
# Placeholder adapters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter_cls", [Text2MidiAdapter, MidiLlmAdapter])
def test_placeholder_adapters_raise_with_guidance(adapter_cls):
    adapter = adapter_cls()
    assert adapter.is_available is False
    with pytest.raises(NotImplementedError) as excinfo:
        adapter.generate(PROMPT)
    message = str(excinfo.value)
    assert "deterministic" in message  # points back at the working default
    assert "creative-audio-lab[ml]" in message  # names the extras needed later
