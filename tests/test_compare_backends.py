"""The benchmark script must run end-to-end and emit a readable table."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "compare_backends.py"


def test_compare_backends_script_runs(tmp_path):
    output = tmp_path / "benchmark.md"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prompt",
            "trap melody with dark bells",
            "--bars",
            "4",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    stdout = result.stdout
    assert "| prompt | backend |" in stdout
    assert "deterministic" in stdout
    assert "ngram_melody" in stdout
    assert "harmonic_fit_score" in stdout
    # The honesty disclaimer must ship with the numbers.
    assert "not evidence" in stdout

    table = output.read_text(encoding="utf-8")
    assert table.count("\n") >= 4  # header + separator + one row per backend
