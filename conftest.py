import sys
from pathlib import Path

# Ensure project root (legacy top-level packages) and src/ (creative_audio_lab)
# are both on the module search path for tests, independent of whether the
# package has been pip-installed.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
