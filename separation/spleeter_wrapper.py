"""Wrapper around the `spleeter` command line tool."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .base import BaseSeparator


class SpleeterSeparator(BaseSeparator):
    """Audio separation using the `spleeter` backend."""

    name = "spleeter"

    def separate(self, input_path: str, output_dir: str) -> None:
        input_path = str(Path(input_path))
        output_dir = str(Path(output_dir))
        cmd = ["spleeter", "separate", "-o", output_dir, input_path]
        subprocess.run(cmd, check=True)


def separate(input_path: str, output_dir: str) -> None:
    """Convenience function mirroring :meth:`BaseSeparator.separate`."""
    SpleeterSeparator().separate(input_path, output_dir)
