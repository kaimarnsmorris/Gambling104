import sys
from pathlib import Path

INV = Path(__file__).resolve().parents[1]
if str(INV) not in sys.path:
    sys.path.insert(0, str(INV))

# The model directory is a sibling of the investigations, shared between them, and
# its block files import their neighbours by bare name (they are loaded by path, not
# as a package). Tests that import `tailfold` or `_chainlink_fv_export` directly need
# the same path the blocks give themselves.
_MODEL = INV.parent.parent / "models" / "chainlink_fv"
if str(_MODEL) not in sys.path:
    sys.path.insert(0, str(_MODEL))
