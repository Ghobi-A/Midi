from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

mastering = pytest.importorskip("mastering")
master_track = mastering.master_track


def test_master_track_creates_file_and_changes_loudness(tmp_path: Path):
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    input_path = tmp_path / "in.wav"
    output_path = tmp_path / "out.wav"
    sf.write(input_path, audio, sr)
    master_track(input_path, output_path, target_lufs=-14.0)
    assert output_path.exists()
    original, _ = sf.read(input_path)
    mastered, _ = sf.read(output_path)
    orig_rms = np.sqrt(np.mean(original**2))
    new_rms = np.sqrt(np.mean(mastered**2))
    assert new_rms > orig_rms
