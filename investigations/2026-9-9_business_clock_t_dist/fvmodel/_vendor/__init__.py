"""Put the vendored packages on sys.path under their own top-level names.

`clkit.state` does `from rvforecast.forward import ForwardModel`, and the fv
modules import `rvforecast.streaming` directly. Rewriting every one of those to a
relative import would be a diff against code we want to keep byte-comparable with
upstream, so the path is adjusted instead and the modules are left alone.
"""
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
