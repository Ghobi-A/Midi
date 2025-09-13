from pathlib import Path

import separation


def test_get_separator_demucs(monkeypatch, tmp_path):
    called = {}

    def fake_separate(input_path, output_dir):
        called["called"] = True
        return Path(output_dir)

    monkeypatch.setattr(separation.demucs_wrapper, "separate", fake_separate)
    separator = separation.get_separator("demucs")
    result = separator("in.wav", tmp_path)
    assert called.get("called") is True
    assert result == Path(tmp_path)


def test_get_separator_spleeter(monkeypatch, tmp_path):
    called = {}

    def fake_separate(input_path, output_dir):
        called["called"] = True
        return Path(output_dir)

    monkeypatch.setattr(separation.spleeter_wrapper, "separate", fake_separate)
    separator = separation.get_separator("spleeter")
    result = separator("in.wav", tmp_path)
    assert called.get("called") is True
    assert result == Path(tmp_path)
