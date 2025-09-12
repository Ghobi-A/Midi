"""Wrapper around the `demucs` command line tool."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .base import BaseSeparator


class DemucsSeparator(BaseSeparator):
    """Audio separation using the `demucs` backend."""

    name = "demucs"

    def separate(self, input_path: str, output_dir: str) -> None:
        input_path = str(Path(input_path))
        output_dir = str(Path(output_dir))
        cmd = ["demucs", "-o", output_dir, input_path]
        subprocess.run(cmd, check=True)


def separate(input_path: str, output_dir: str) -> None:
    """Convenience function mirroring :meth:`BaseSeparator.separate`."""
    DemucsSeparator().separate(input_path, output_dir)
