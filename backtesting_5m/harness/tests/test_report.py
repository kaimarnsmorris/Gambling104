"""Pins that reporting is separate from running, and multi-run by default.

`run()` sees exactly one run, so a hyperparameter sweep -- the thing anyone
actually wants to look at -- cannot be plotted from inside it. The loader
therefore takes a mapping of label -> run directory, and a single run is the
one-element case rather than a separate path.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from harness.report import cumulative_pnl, load_runs


def _run_dir(tmp_path, name, pnl):
    d = tmp_path / name
    os.makedirs(str(d))
    pd.DataFrame({
        "market_id": [f"m{i}" for i in range(len(pnl))],
        "open_ts": 1786665600 + np.arange(len(pnl)) * 300,
        "day": ["2026-08-20"] * len(pnl),
        "seed": [0] * len(pnl),
        "pnl_net": pnl,
        "pnl_gross": [p + 0.5 for p in pnl],
        "shares": [10.0] * len(pnl),
        "n_fills": [1] * len(pnl),
    }).to_parquet(str(d / "markets.parquet"))
    with open(str(d / "summary.json"), "w") as fh:
        json.dump({"headline": {"n_markets": len(pnl)}}, fh)
    return str(d)


def test_load_runs_tags_each_row_with_its_run(tmp_path):
    m = load_runs({"a": _run_dir(tmp_path, "a", [1.0, 2.0]),
                   "b": _run_dir(tmp_path, "b", [3.0])})
    assert set(m["run"]) == {"a", "b"}
    assert len(m) == 3


def test_a_single_run_is_the_one_element_case(tmp_path):
    m = load_runs({"only": _run_dir(tmp_path, "only", [1.0])})
    assert list(m["run"]) == ["only"]


def test_cumulative_pnl_writes_one_figure_for_many_runs(tmp_path):
    m = load_runs({"a": _run_dir(tmp_path, "a", [1.0, 2.0]),
                   "b": _run_dir(tmp_path, "b", [-1.0, -2.0])})
    out = str(tmp_path / "cum.png")
    cumulative_pnl(m, out)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_run_no_longer_accepts_a_plots_flag():
    from harness.core.config import Output
    assert not hasattr(Output(), "plots"), (
        "plotting belongs in harness.report, not in the engine")
