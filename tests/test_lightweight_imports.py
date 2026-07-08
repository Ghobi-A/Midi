"""Guard the lightweight install story: core + Stage 1.5 packages must not
require (or even touch) the ML/audio/symbolic extras."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Everything that only the optional extras should ever pull in.
FORBIDDEN_MODULES = (
    "torch",
    "torchaudio",
    "transformers",
    "sklearn",
    "datasets",
    "miditok",
    "symusic",
    "librosa",
    "soundfile",
    "demucs",
    "spleeter",
    "tensorflow",
)

CHECK_SCRIPT = f"""
import sys
sys.path.insert(0, {str(ROOT / "src")!r})
import creative_audio_lab
import creative_audio_lab.data
import creative_audio_lab.evaluation
import creative_audio_lab.export
import creative_audio_lab.models
import creative_audio_lab.preview
import creative_audio_lab.tokenization
loaded = [m for m in {FORBIDDEN_MODULES!r} if m in sys.modules]
if loaded:
    raise SystemExit("heavy modules loaded by core import: " + ", ".join(loaded))
"""


def test_core_packages_import_without_ml_audio_extras():
    """Importing every creative_audio_lab package must not load any heavy extra."""
    result = subprocess.run(
        [sys.executable, "-c", CHECK_SCRIPT],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_app_module_imports_without_ml_audio_extras():
    """`import app` (the Streamlit UI) needs streamlit but nothing heavier."""
    pytest.importorskip("streamlit", reason="app extra not installed (core-only environment)")
    script = (
        f"import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"import app\n"
        f"loaded = [m for m in {FORBIDDEN_MODULES!r} if m in sys.modules]\n"
        f"if loaded:\n"
        f"    raise SystemExit('heavy modules loaded by app import: ' + ', '.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout
