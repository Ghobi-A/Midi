import sys
from pathlib import Path

# Ensure project root is on the module search path for tests
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
