#!/usr/bin/env python
"""Run the canonical rights-cleared MIDI baseline experiment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from creative_audio_lab.data import CorpusConfig
from creative_audio_lab.experiments import run_experiment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        path = run_experiment(args.manifest, args.output_dir,
                              config=CorpusConfig(split_seed=args.split_seed),
                              samples=args.samples)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"Run artefacts: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
