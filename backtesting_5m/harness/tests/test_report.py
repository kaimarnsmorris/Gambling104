"""Pins that reporting is separate from running, and multi-run by default.

`run()` sees exactly one run, so a hyperparameter sweep -- the thing anyone
actually wants to look at -- cannot be plotted from inside it. The loader
therefore takes a mapping of label -> run directory, and a single run is the
one-element case rather than a separate path.
"""
import json
import os

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest

from harness.report import (calibration, cumulative_pnl, load_ledgers,
                            load_runs, load_summaries, markout_distribution,
                            pnl_by_tte)
from harness.blocks.defaults.fees import Liquidity


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


def test_maker_and_taker_are_not_inverted():
    """A hardcoded enum value once had these backwards, so pin it."""
    from harness.blocks.defaults.fees import Liquidity
    from harness.report.figures import LIQ_MAKER
    assert LIQ_MAKER == int(Liquidity.MAKER) == 0


# -- the four figures/loaders with zero call sites elsewhere ---------------
#
# `markout_distribution`, `pnl_by_tte`, `calibration`, `load_ledgers` and
# `load_summaries` are exercised nowhere in the harness, the tests, the
# template or any investigation -- only `cumulative_pnl` and `load_runs` are.
# `figures.py` shipped once with maker and taker inverted via a hardcoded
# constant no test caught, because every existing test only checked that a
# PNG got written. These monkeypatch the matplotlib call that actually draws
# each series so the test can see which VALUES landed in which series --
# a PNG-exists check would pass on an inverted plot exactly as it did before.

def _ledger_dir(tmp_path, name, rows):
    d = tmp_path / name
    os.makedirs(str(d))
    pd.DataFrame(rows).to_parquet(str(d / "ledger.parquet"))
    return str(d)


def test_load_ledgers_tags_each_row_with_its_run(tmp_path):
    a = _ledger_dir(tmp_path, "a", {"market_id": ["m0"], "liquidity": [0]})
    b = _ledger_dir(tmp_path, "b",
                    {"market_id": ["m1", "m2"], "liquidity": [0, 1]})
    m = load_ledgers({"a": a, "b": b})
    assert set(m["run"]) == {"a", "b"}
    assert len(m) == 3
    assert list(m[m["run"] == "b"]["market_id"]) == ["m1", "m2"]


def test_load_summaries_maps_label_to_its_own_json(tmp_path):
    a = _run_dir(tmp_path, "a", [1.0])
    b = _run_dir(tmp_path, "b", [2.0, 3.0])
    out = load_summaries({"a": a, "b": b})
    assert out["a"]["headline"]["n_markets"] == 1
    assert out["b"]["headline"]["n_markets"] == 2


def test_markout_distribution_keeps_maker_and_taker_series_separate(
        tmp_path, monkeypatch):
    """This is the shape of test that would have caught the LIQ_MAKER
    inversion: it checks which VALUES ended up labelled "maker" and which
    "taker", not merely that a histogram of some kind was drawn."""
    calls = []
    orig_hist = matplotlib.axes.Axes.hist

    def spy(self, x, *a, **kw):
        calls.append((list(np.asarray(x)), kw.get("label", "")))
        return orig_hist(self, x, *a, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "hist", spy)

    ledgers = pd.DataFrame({
        "run": ["r1"] * 5,
        "liquidity": ([int(Liquidity.MAKER)] * 3
                      + [int(Liquidity.TAKER)] * 2),
        "delta_quality_c": [5.0, 6.0, 7.0, -2.0, -3.0],
    })
    markout_distribution(ledgers, str(tmp_path / "markout.png"))

    maker_call = next(c for c in calls if "maker" in c[1])
    taker_call = next(c for c in calls if "taker" in c[1])
    assert sorted(maker_call[0]) == [5.0, 6.0, 7.0]
    assert sorted(taker_call[0]) == [-3.0, -2.0]
    assert "n=3" in maker_call[1]
    assert "n=2" in taker_call[1]


def test_pnl_by_tte_buckets_and_sums_the_proxy_correctly(
        tmp_path, monkeypatch):
    calls = []
    orig_plot = matplotlib.axes.Axes.plot

    def spy(self, *a, **kw):
        calls.append((a, kw))
        return orig_plot(self, *a, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy)

    # tte = 300 - t_ms/1000; with bucket_s=100 that puts t_ms=0 alone in
    # bucket 300, and t_ms=150000/180000 together in bucket 100.
    ledgers = pd.DataFrame({
        "run": ["r1"] * 3,
        "t_ms": [0, 150000, 180000],
        "shares": [1.0, 2.0, 1.0],
        "delta_quality_c": [1000.0, 200.0, 300.0],
    })
    pnl_by_tte(ledgers, str(tmp_path / "tte.png"), bucket_s=100.0)

    assert len(calls) == 2       # one plot per axis: pnl proxy, then fills
    pnl_x, pnl_y = list(calls[0][0][0]), list(calls[0][0][1])
    fills_x, fills_y = list(calls[1][0][0]), list(calls[1][0][1])

    assert pnl_x == [300, 100] == fills_x     # descending tte, per sort_index
    assert np.allclose(pnl_y, [10.0, 7.0])    # 1000/100*1; (200/100*2+300/100*1)
    assert fills_y == [1, 2]


def test_calibration_pairs_predicted_fair_p_with_the_realised_outcome(
        tmp_path, monkeypatch):
    calls = []
    orig_scatter = matplotlib.axes.Axes.scatter

    def spy(self, x, y, *a, **kw):
        calls.append((np.asarray(x), np.asarray(y), kw.get("label", "")))
        return orig_scatter(self, x, y, *a, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", spy)

    ledgers = pd.DataFrame({
        "run": ["r1"] * 4,
        "market_id": ["m0", "m1", "m2", "m3"],
        "fair_p": [0.2, 0.2, 0.8, 0.8],
    })
    markets = pd.DataFrame({
        "run": ["r1"] * 4,
        "market_id": ["m0", "m1", "m2", "m3"],
        "winner_up": [0, 0, 1, 1],
    })
    calibration(ledgers, markets, str(tmp_path / "cal.png"), n_buckets=2)

    assert len(calls) == 1
    predicted, realised, label = calls[0]
    assert sorted(np.round(predicted, 2)) == [0.2, 0.8]
    assert sorted(realised) == [0.0, 1.0]
    assert "n=4" in label
