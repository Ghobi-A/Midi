"""Wrapper around Suno's music generation API."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests

# Default location for a JSON config file containing {"api_key": "..."}
CONFIG_FILE = Path(os.getenv("SUNO_CONFIG_FILE", Path.home() / ".suno.json"))
API_URL = os.getenv("SUNO_API_URL", "https://api.suno.ai/v2/generate")


def get_api_key() -> str:
    """Return the API key from env var or config file."""
    key = os.getenv("SUNO_API_KEY")
    if key:
        return key
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)
            if isinstance(data, dict) and "api_key" in data:
                return data["api_key"]
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON in {CONFIG_FILE}")
    raise RuntimeError(
        "Suno API key not found. Set SUNO_API_KEY or create "
        f"{CONFIG_FILE} with an 'api_key' field."
    )


def generate_track(output_path: str, **kwargs: Any) -> Path:
    """Generate a music track and save it to ``output_path``.

    Parameters
    ----------
    output_path:
        Path to the output WAV/FLAC file.
    **kwargs:
        Additional JSON parameters passed directly to Suno's API.
    """
    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(API_URL, json=kwargs, headers=headers, timeout=60)
    response.raise_for_status()
    out = Path(output_path)
    out.write_bytes(response.content)
    return out


def main(argv: list[str] | None = None) -> None:
    """Simple CLI entry point for track generation."""
    parser = argparse.ArgumentParser(description="Generate music with Suno's API")
    parser.add_argument("output_path", help="Where to save the generated audio file")
    parser.add_argument(
        "--prompt", required=True, help="Text prompt for music generation"
    )
    parser.add_argument(
        "--duration", type=int, default=None, help="Optional duration parameter"
    )
    args = parser.parse_args(argv)

    kwargs: dict[str, Any] = {"prompt": args.prompt}
    if args.duration is not None:
        kwargs["duration"] = args.duration

    generate_track(args.output_path, **kwargs)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
