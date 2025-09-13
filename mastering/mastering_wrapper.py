from __future__ import annotations

"""Entry point wrapper around :func:`ai_master.master_track`."""

import argparse

from .ai_master import master_track


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for simple mastering."""

    parser = argparse.ArgumentParser(description="Simple mastering wrapper")
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("output", help="Path for mastered output")
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-14.0,
        help="Target loudness in LUFS (default: -14)",
    )
    args = parser.parse_args(argv)
    master_track(args.input, args.output, args.target_lufs)


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    main()
