"""Creative Audio Lab — Prompt-to-MIDI Generator (Streamlit app).

A prompt-conditioned symbolic music generator that creates DAW-ready MIDI
arrangements instead of finished audio. Run with:

    streamlit run app.py
"""

from __future__ import annotations

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
    for info in future_backends:
        st.caption(f"🔒 **{info.display_name}** — scaffolded, not yet available")
    with st.expander("ML readiness"):
        st.markdown(
            "- The **deterministic backend** (rule-based parser + generators) is the "
            "current default — no trained model is involved.\n"
            "- A **symbolic tokenization layer** (`creative_audio_lab.tokenization`) "
            "converts note events to/from REMI-style tokens, with an optional MidiTok "
            "adapter.\n"
            "- A **dataset manifest + provenance layer** (`creative_audio_lab.data`) "
            "tracks source, license, and commercial-use rights before any training data "
            "is touched.\n"
            "- **Future model adapters** (Text2midi, MIDI-LLM) are scaffolded behind the "
            "same `GenerationBackend` interface, so a learned model can plug in without "
            "changing this app."
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
        backend = get_backend(backend_name)
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

if "arrangement" in st.session_state:
    arrangement = st.session_state["arrangement"]
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
        st.markdown(
            "This arrangement was produced by **deterministic, rule-based generators** "
            "(chord progression → motif-based melody → bass → drums), not a trained model. "
            "See the README's ML Roadmap section and `docs/ML_ROADMAP.md` for how this baseline "
            "is designed to be replaced or augmented by learned models."
        )
else:
    st.info("Enter a prompt (or pick an example) and click Generate to create a MIDI arrangement.")
