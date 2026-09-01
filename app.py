"""Creative Audio Lab — Prompt-to-MIDI Generator (Streamlit app).

A prompt-conditioned symbolic music generator that creates DAW-ready MIDI
arrangements instead of finished audio. Run with:

    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running straight from a checkout without an editable install.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from creative_audio_lab.evaluation import evaluate_arrangement
from creative_audio_lab.export import export_arrangement_files
from creative_audio_lab.models import DEFAULT_BACKEND_NAME, get_backend, list_backends
from creative_audio_lab.motif_detection import detect_motifs
from creative_audio_lab.music_theory import SCALES
from creative_audio_lab.preview import summarize_arrangement
from creative_audio_lab.prompt_parser import STYLE_PRESETS

#: Environment variable naming a trained melody artefact to load by default.
NGRAM_MODEL_ENV = "CREATIVE_AUDIO_LAB_NGRAM_MODEL"
#: Directory that sidebar-supplied model paths are confined to.
MODEL_ROOT_ENV = "CREATIVE_AUDIO_LAB_MODEL_ROOT"


class ModelPathRejected(ValueError):
    """A sidebar-supplied model path escaped the allowed directory."""


def model_root(env: dict = None) -> Path:
    """Directory that sidebar-supplied model paths must live under.

    Defaults to the working directory the app was launched from; override
    with ``CREATIVE_AUDIO_LAB_MODEL_ROOT`` to keep artefacts elsewhere.
    """
    env = os.environ if env is None else env
    configured = (env.get(MODEL_ROOT_ENV) or "").strip()
    return (Path(configured).expanduser() if configured else Path.cwd()).resolve()


def resolve_ngram_model_path(user_input: str = "", env: dict = None) -> Path | None:
    """Resolve which melody model the n-gram backend should load.

    The two sources are *not* equally trusted, and that is the whole point
    of this function. ``CREATIVE_AUDIO_LAB_NGRAM_MODEL`` is set by whoever
    launches the process — the operator — so it may name any path. The
    sidebar field is typed by whoever is *viewing* the app, who on a
    deployed Streamlit instance is not the same person; an unconstrained
    path there would let a visitor point the app at any file on the host
    and learn, from the resulting success or error, whether it exists and
    whether it parses as JSON.

    So a sidebar value is resolved against :func:`model_root`, must stay
    inside it after symlinks and ``..`` are normalised away, and must name
    a ``.json`` file. The sidebar wins over the environment variable; empty
    at both levels means "use the synthetic bootstrap model".

    Raises
    ------
    ModelPathRejected
        If the sidebar value escapes the root or is not a ``.json`` file.
    """
    env = os.environ if env is None else env
    typed = (user_input or "").strip()
    if typed:
        root = model_root(env)
        # resolve() normalises "..", follows symlinks, and makes a relative
        # entry relative to the root rather than the process cwd.
        candidate = (root / typed).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise ModelPathRejected(
                f"Model paths must stay inside {root}. Set {MODEL_ROOT_ENV} to "
                "allow a different directory."
            ) from None
        if candidate.suffix.lower() != ".json":
            raise ModelPathRejected("A melody model must be a .json artefact file.")
        return candidate
    configured = (env.get(NGRAM_MODEL_ENV) or "").strip()
    return Path(configured).expanduser() if configured else None


def describe_artefact(metadata: dict) -> list[str]:
    """Bullet lines describing a loaded artefact's corpus, licence, and held-out score."""
    corpus = metadata.get("corpus", {})
    lines = []
    if corpus.get("source") == "synthetic-bootstrap":
        lines.append("**Corpus:** synthetic bootstrap (this project's deterministic backend)")
    else:
        datasets = corpus.get("datasets", [])
        names = ", ".join(str(d.get("name")) for d in datasets) or "unnamed corpus"
        licences = ", ".join(sorted({str(d.get("license")) for d in datasets if d.get("license")}))
        lines.append(f"**Corpus:** {names}")
        if licences:
            lines.append(f"**Licence:** {licences}")
    kind = metadata.get("model_kind")
    orders = metadata.get("orders", {})
    if kind:
        lines.append(f"**Model:** {kind} (orders: {orders})")
    test_metrics = (metadata.get("metrics") or {}).get("test") or {}
    if test_metrics.get("notes"):
        lines.append(
            f"**Held-out (test):** {test_metrics['bits_per_note']:.2f} bits/note, "
            f"pitch {test_metrics['pitch_bits_per_note']:.2f} bits/note over "
            f"{test_metrics['notes']} notes"
        )
    return lines

EXAMPLE_PROMPTS = [
    "dark orchestral boss battle theme, 140 BPM, brass ostinato, strings, piano arps",
    "romantic piano loop in C minor, 90 BPM",
    "trap melody with dark bells and sliding bass",
    "nostalgic JRPG town theme with piano and flute",
    "aggressive cinematic trailer cue with low brass and taiko drums",
    "hopeful piano and strings theme for a fantasy village",
]

DOWNLOAD_LABELS = {
    "full_arrangement.mid": "Full arrangement",
    "chords.mid": "Chords",
    "melody.mid": "Melody",
    "bass.mid": "Bass",
    "drums.mid": "Drums",
}

st.set_page_config(page_title="Creative Audio Lab — Prompt-to-MIDI", page_icon="🎼", layout="wide")
st.title("🎼 Creative Audio Lab")
st.caption(
    "**Prompt-to-MIDI Generator** — a prompt-conditioned symbolic music generator that creates "
    "DAW-ready MIDI arrangements instead of finished audio."
)

if "prompt_text" not in st.session_state:
    st.session_state["prompt_text"] = ""

backend_infos = list_backends()
available_backends = [info for info in backend_infos if info.available]
future_backends = [info for info in backend_infos if not info.available]

with st.sidebar:
    st.header("Backend")
    backend_labels = {info.display_name: info.name for info in available_backends}
    backend_label = st.selectbox("Generation backend", list(backend_labels))
    backend_name = backend_labels.get(backend_label, DEFAULT_BACKEND_NAME)
    backend_kwargs = {}
    if backend_name == "ngram_melody":
        # Deliberately not pre-filled from the environment: a value typed
        # here is viewer input and is confined to model_root(), while the
        # environment variable is operator configuration and is not.
        configured_model = os.environ.get(NGRAM_MODEL_ENV, "").strip()
        if configured_model:
            st.caption(f"Default from `{NGRAM_MODEL_ENV}`: `{configured_model}`")
        model_input = st.text_input(
            "Trained melody model (JSON path)",
            help=f"A training artefact from scripts/train_ngram_melody.py, "
            f"relative to {model_root()}. Leave empty to use "
            f"{'the environment default' if configured_model else 'the synthetic bootstrap model'}.",
        )
        try:
            model_path = resolve_ngram_model_path(model_input)
        except ModelPathRejected as rejection:
            st.error(f"{rejection} Falling back to the bootstrap model.")
            model_path = None
        if model_path is None:
            st.caption(
                "⚠️ **Stage 2 baseline, bootstrap model.** Chords/bass/drums stay "
                "rule-based; only the melody is continued by a small n-gram model, "
                "trained here on synthetic melodies from the deterministic backend "
                "— it demonstrates the statistical pipeline, not learning from real "
                "music."
            )
        elif not model_path.exists():
            st.error(f"No model file at {model_path} — falling back to the bootstrap model.")
        else:
            backend_kwargs["model_path"] = model_path
            try:
                from creative_audio_lab.models.artefact import load_any_model

                _, artefact_metadata = load_any_model(model_path)
            except Exception as error:  # noqa: BLE001 - surfaced to the user below
                st.error(f"Could not read {model_path}: {error}")
                backend_kwargs.pop("model_path", None)
                artefact_metadata = None
            if backend_kwargs.get("model_path") is not None:
                if artefact_metadata:
                    st.success("Loaded trained melody model.")
                    for line in describe_artefact(artefact_metadata):
                        st.caption(line)
                else:
                    st.warning(
                        "Loaded a bare model file with no training metadata — its "
                        "corpus, licence, and held-out scores are unknown."
                    )
    for info in future_backends:
        st.caption(f"🔒 **{info.display_name}** — scaffolded, not yet available")
    with st.expander("ML readiness"):
        st.markdown(
            "- The **deterministic backend** (rule-based parser + generators) is the "
            "current default — no trained model is involved.\n"
            "- The **n-gram melody baseline** (Stage 2) is the first statistical "
            "backend: it samples melody continuations from a trained n-gram model "
            "over symbolic tokens, bootstrap-trained on synthetic data unless you "
            "supply a licensed local corpus.\n"
            "- A **symbolic tokenization layer** (`creative_audio_lab.tokenization`) "
            "converts note events to/from REMI-style tokens, with an optional MidiTok "
            "adapter.\n"
            "- A **dataset manifest + provenance layer** (`creative_audio_lab.data`) "
            "tracks source, license, and commercial-use rights before any training data "
            "is touched.\n"
            "- **Future model adapters** (Text2midi, MIDI-LLM) are scaffolded behind the "
            "same `GenerationBackend` interface, so a heavier learned model can plug in "
            "without changing this app."
        )

    st.header("Controls")
    st.caption("Leave a control on Auto to let the deterministic prompt parser decide.")
    key_choice = st.selectbox("Key", ["Auto", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
    mode_choice = st.selectbox("Scale / Mode", ["Auto"] + list(SCALES.keys()))
    bpm_choice = st.number_input("BPM (0 = infer from prompt)", min_value=0, max_value=220, value=0, step=1)
    bars_choice = st.number_input("Bar length (0 = infer from prompt)", min_value=0, max_value=64, value=0, step=4)
    style_choice = st.selectbox("Style preset", ["Auto"] + [name for name in STYLE_PRESETS if name != "default"])
    energy_choice = st.selectbox("Energy", ["Auto", "low", "medium", "high"])
    density_choice = st.selectbox("Rhythmic density", ["Auto", "low", "medium", "high"])
    instruments_choice = st.multiselect(
        "Instrument set (leave empty to infer)",
        ["piano", "strings", "brass", "bells", "flute", "drums", "taiko_drums", "bass"],
    )

with st.expander("Example prompts"):
    example_cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_PROMPTS):
        if example_cols[i % 2].button(example, key=f"example_{i}"):
            st.session_state["prompt_text"] = example

st.text_area("Text prompt", key="prompt_text", height=90, placeholder="e.g. dark orchestral boss battle theme, 140 BPM, brass ostinato, strings, piano arps")
prompt = st.session_state["prompt_text"]

generate = st.button("Generate", type="primary")

if generate:
    if not prompt.strip():
        st.warning("Enter a prompt first.")
    else:
        backend = get_backend(backend_name, **backend_kwargs)
        arrangement = backend.generate(
            prompt,
            key=None if key_choice == "Auto" else key_choice,
            mode=None if mode_choice == "Auto" else mode_choice,
            bpm=None if bpm_choice == 0 else int(bpm_choice),
            bars=None if bars_choice == 0 else int(bars_choice),
            style=None if style_choice == "Auto" else style_choice,
            energy=None if energy_choice == "Auto" else energy_choice,
            density=None if density_choice == "Auto" else density_choice,
            instruments=instruments_choice or None,
        )
        st.session_state["arrangement"] = arrangement
        st.session_state["arrangement_backend"] = backend_name
        st.session_state["arrangement_model_path"] = str(backend_kwargs.get("model_path", ""))

if "arrangement" in st.session_state:
    arrangement = st.session_state["arrangement"]
    arrangement_backend = st.session_state.get("arrangement_backend", DEFAULT_BACKEND_NAME)
    controls = arrangement.controls

    st.subheader("Arrangement summary")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Key / Mode", f"{controls.key} {controls.mode}")
    summary_cols[1].metric("BPM", controls.bpm)
    summary_cols[2].metric("Bars", controls.bars)
    summary_cols[3].metric("Style", controls.style)
    st.write(
        f"**Mood:** {controls.mood} · **Energy:** {controls.energy} · "
        f"**Density:** {controls.density} · **Section:** {controls.section}"
    )
    st.write(f"**Instruments:** {', '.join(controls.instruments) or '—'}")
    if controls.texture:
        st.write(f"**Texture cues:** {', '.join(controls.texture)}")

    degrees = [str(event.chord.degree) for event in arrangement.chord_events]
    shown = " – ".join(degrees[:16]) + (" ..." if len(degrees) > 16 else "")
    st.write(f"**Chord progression (scale degrees):** {shown}")

    summary = summarize_arrangement(arrangement)
    with st.expander("Track summary"):
        st.table(
            [
                {
                    "Track": track.name,
                    "Program": "GM drums" if track.is_drum else (track.program if track.program is not None else "—"),
                    "Notes": track.note_count,
                    "Bars": track.duration_bars,
                    "Pitch range": "—" if track.pitch_min is None or track.is_drum else f"{track.pitch_min}–{track.pitch_max}",
                    "Notes/bar": f"{track.notes_per_bar:.1f}",
                }
                for track in summary.tracks
            ]
        )

    st.subheader("Download MIDI")
    files = export_arrangement_files(arrangement)
    download_cols = st.columns(len(DOWNLOAD_LABELS))
    for col, (filename, label) in zip(download_cols, DOWNLOAD_LABELS.items()):
        col.download_button(label, data=files[filename], file_name=filename, mime="audio/midi")

    st.subheader("Evaluation metrics")
    metrics = evaluate_arrangement(arrangement)
    metric_cols = st.columns(4)
    for i, (name, value) in enumerate(metrics.items()):
        display_value = f"{value:.2f}" if isinstance(value, float) else value
        metric_cols[i % 4].metric(name.replace("_", " ").title(), display_value)

    st.subheader("Detected motifs (MIDI Motif Lab)")
    motifs = detect_motifs(arrangement.parts.get("melody", []))
    if not motifs:
        st.write("No repeated motifs detected in this arrangement.")
    else:
        for i, motif in enumerate(motifs[:5]):
            st.write(
                f"**Motif {i + 1}** ({motif.kind}) — repeated {motif.repetition_count}x, "
                f"length {motif.length} notes, score {motif.score:.2f}"
            )
            st.caption(f"Intervals: {motif.intervals} · Durations: {motif.durations}")

    with st.expander("How this baseline works, and what comes next"):
        if arrangement_backend == "ngram_melody":
            used_model = st.session_state.get("arrangement_model_path", "")
            provenance = (
                f"The melody model came from `{used_model}`; its corpus, licence, and "
                "held-out scores are recorded in that artefact."
                if used_model
                else "The model is bootstrap-trained on synthetic melodies from the "
                "deterministic backend, so this demonstrates the training and sampling "
                "machinery, not musical generalisation."
            )
            st.markdown(
                "Chords, bass, drums, and structure in this arrangement were produced by the "
                "**deterministic, rule-based generators**; the **melody** continues the opening "
                "motif by sampling from a small **n-gram model over symbolic tokens** — the "
                f"Stage 2 statistical baseline. {provenance} See `docs/ML_ROADMAP.md`."
            )
        else:
            st.markdown(
                "This arrangement was produced by **deterministic, rule-based generators** "
                "(chord progression → motif-based melody → bass → drums), not a trained model. "
                "See the README's ML Roadmap section and `docs/ML_ROADMAP.md` for how this baseline "
                "is designed to be replaced or augmented by learned models."
            )
else:
    st.info("Enter a prompt (or pick an example) and click Generate to create a MIDI arrangement.")
