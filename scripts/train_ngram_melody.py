#!/usr/bin/env python
"""Train the Stage 2 n-gram melody model and save it as JSON.

By default this trains on the synthetic bootstrap corpus (melodies generated
by the deterministic backend — nothing is downloaded). See
``creative_audio_lab/models/ngram_training.py`` for what that corpus is and
is not.

Usage:
    python scripts/train_ngram_melody.py --output ngram_melody.json
    python scripts/train_ngram_melody.py --output ngram_melody.json --order 4
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running straight from a checkout without an editable install.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from creative_audio_lab.models.ngram_training import main

if __name__ == "__main__":
    raise SystemExit(main())
