"""Minimal wrapper for the Suno music generation API.

This module exposes a :func:`generate_track` helper that sends a request to
Suno's HTTP API and stores the resulting audio file (WAV or FLAC) to disk.
API credentials are read from the ``SUNO_API_KEY`` environment variable or
from a small JSON configuration file.  Example configuration files are looked
for at ``~/.suno.json`` or ``~/.config/suno.json`` and must contain a single
``{"api_key": "..."}`` entry.

The actual parameters required by Suno's API are forwarded via ``**kwargs``;
at minimum a ``prompt`` describing the music to generate is typically
expected.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests


DEFAULT_ENDPOINT = "https://api.suno.ai/v1/generate"
"""Default HTTP endpoint for the Suno generation service."""


def _load_api_key() -> str:
    """Retrieve the API key from the environment or a config file.

    The search order is:

    1. ``SUNO_API_KEY`` environment variable.
    2. JSON files at ``SUNO_CONFIG``, ``~/.suno.json`` or ``~/.config/suno.json``.

    Returns
    -------
    str
        The API key string.

    Raises
    ------
    RuntimeError
        If no API key can be found.
    """

    key = os.getenv("SUNO_API_KEY")
    if key:
        return key

    config_paths = [
        os.getenv("SUNO_CONFIG"),
        Path.home() / ".suno.json",
        Path.home() / ".config" / "suno.json",
    ]

    for path in config_paths:
        if not path:
            continue
        path = Path(path)
        if path.exists():
            with path.open() as fh:
                data = json.load(fh)
            key = data.get("api_key")
            if key:
                return key

    raise RuntimeError(
        "Suno API key not found. Set SUNO_API_KEY or create a config file."
    )


def generate_track(output_path: str, **kwargs: Any) -> str:
    """Generate an audio track using Suno's API.

    Parameters
    ----------
    output_path:
        Destination path where the generated audio (WAV/FLAC) will be saved.
    **kwargs:
        Additional parameters forwarded to the API request.  The ``endpoint``
        parameter can be used to override :data:`DEFAULT_ENDPOINT`.

    Returns
    -------
    str
        The ``output_path`` where the audio file was written.
    """

    api_key = _load_api_key()
    endpoint = kwargs.pop("endpoint", DEFAULT_ENDPOINT)

    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(endpoint, headers=headers, json=kwargs, timeout=60)
    response.raise_for_status()

    data = response.json()
    audio_url = data.get("audio_url")
    if not audio_url:
        raise RuntimeError("API response did not include an 'audio_url'")

    audio_response = requests.get(audio_url, timeout=60)
    audio_response.raise_for_status()

    with open(output_path, "wb") as fh:
        fh.write(audio_response.content)

    return output_path


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for standalone execution."""

    parser = argparse.ArgumentParser(description="Generate audio via Suno's API")
    parser.add_argument("output_path", help="Path to the output .wav or .flac file")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt describing the music to generate",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Optional Suno API endpoint override",
    )
    parser.add_argument("--title", help="Optional title for the track")
    parser.add_argument("--tags", nargs="*", help="Optional tags for the track")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    params: dict[str, Any] = {
        k: v
        for k, v in vars(args).items()
        if k not in {"output_path", "endpoint"} and v is not None
    }
    generate_track(args.output_path, endpoint=args.endpoint, **params)

