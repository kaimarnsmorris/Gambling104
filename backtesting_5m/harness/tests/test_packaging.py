"""Pins that `harness` is importable as an installed package.

Investigations live at the repository root, two directories away from this
module. The sys.path.insert every runner used to carry was depth-dependent: it
resolved correctly from backtesting_5m/investigations/x/ and silently resolved
to the WRONG directory from investigations/x/ -- and the block resolver does not
error on that, it falls back to defaults and quietly evaluates a different
model. An installed package removes the failure mode.
"""
import subprocess
import sys

import harness


def test_harness_exposes_a_version():
    assert isinstance(harness.__version__, str)
    assert harness.__version__


def test_harness_imports_from_an_unrelated_working_directory(tmp_path):
    """The real test: no sys.path help, cwd nowhere near the module."""
    out = subprocess.run(
        [sys.executable, "-c", "import harness; print(harness.__version__)"],
        cwd=str(tmp_path), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == harness.__version__
