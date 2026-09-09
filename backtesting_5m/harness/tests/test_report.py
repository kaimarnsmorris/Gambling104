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

import matplotlib.pyplot as plt

from harness.report import (calibration, cumulative_pnl, load_ledgers,
                            load_runs, load_summaries, load_ticks,
                            market_detail, markout_distribution,
                            pnl_by_tte)
from harness.blocks.defaults.fees import Liquidity
from harness.core.types import Side


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


def _ticks_dir(tmp_path, name, seeds=(0,)):
    """A run folder with per-tick diagnostics for one market, per seed."""
    d = tmp_path / name
    os.makedirs(str(d), exist_ok=True)
    rows = []
    for seed in seeds:
        n = 5
        rows.append(pd.DataFrame({
            "market_id": ["m0"] * n,
            "seed": [seed] * n,
            "t_ms": np.arange(n) * 100,
            "s": 70000.0 + np.arange(n),
            "sigma": [0.0004] * n,
            "z": np.linspace(-0.2, 0.2, n),
            "fair_p": np.linspace(0.45, 0.55, n),
            "eff_bid": np.linspace(0.43, 0.53, n),
            "eff_ask": np.linspace(0.47, 0.57, n),
            "book_bid": [0.49] * n,
            "book_ask": [0.51] * n,
            "mid": [0.50] * n,
            "book_age_ms": [0.0] * n,
            "spot": 70050.0 + np.arange(n),
            "chainlink": 70010.0 + np.arange(n),
            # the seed is what distinguishes the replays; make it visible
            "q": np.full(n, float(seed)),
            "cash": [0.0] * n,
            "cum_pnl": np.full(n, float(seed) * 10.0),
            "orders_live": [2] * n,
        }))
    pd.concat(rows, ignore_index=True).to_parquet(str(d / "ticks.parquet"))
    pd.DataFrame({
        "market_id": ["m0", "m0"],
        "seed": [seeds[0], seeds[0]],
        "t_ms": [100, 300],
        "side": [int(Side.BUY), int(Side.SELL)],
        "price": [0.49, 0.51],
        "liquidity": [int(Liquidity.MAKER)] * 2,
    }).to_parquet(str(d / "ledger.parquet"))
    return str(d)


def test_load_ticks_tags_each_row_with_its_run(tmp_path):
    t = load_ticks({"a": _ticks_dir(tmp_path, "a")})
    assert set(t["run"]) == {"a"}
    assert len(t) == 5


def test_load_ticks_skips_a_run_that_emitted_none(tmp_path):
    """A mapping may mix runs that emitted ticks with runs that did not --
    `emit_ticks` is per-run and off by default, so raising here would make
    the common case the error case."""
    mapping = {"with": _ticks_dir(tmp_path, "with"),
               "without": _run_dir(tmp_path, "without", [1.0])}
    t = load_ticks(mapping)
    assert set(t["run"]) == {"with"}


def _drawn(monkeypatch, ticks, ledger, market_id, path):
    """Run `market_detail` but keep the figure alive so its data can be read.

    Asserting on a saved PNG only proves a file appeared. These figures had a
    maker/taker inversion ship green once precisely because nothing looked at
    what was drawn, so these tests read the artists.
    """
    import harness.report.figures as figs
    monkeypatch.setattr(figs.plt, "close", lambda *a, **k: None)
    figs.market_detail(ticks, ledger, market_id, path)
    fig = figs.plt.gcf()
    series = {}
    for ax in fig.axes:
        for ln in ax.get_lines():
            series[ln.get_label()] = ln.get_ydata()
    figs.plt.close(fig)
    return fig, series


def test_market_detail_draws_one_seed_not_all_of_them(tmp_path, monkeypatch):
    """Every seed replays the SAME market, so overlaying them would draw
    three inventory paths on one axis and read as noise in the model rather
    than a choice about latency draws.

    The fixture makes `q` and `cum_pnl` equal to the seed, so the drawn
    position says outright which replay reached the plot."""
    d = _ticks_dir(tmp_path, "multi", seeds=(0, 1, 2))
    t, ledger = load_ticks({"a": d}), load_ledgers({"a": d})
    _, series = _drawn(monkeypatch, t, ledger, "m0",
                       str(tmp_path / "m0.png"))
    assert list(series["position"]) == [0.0] * 5, (
        "seed 1 and 2 have q == 1 and 2; anything but all-zero means more "
        "than one replay was drawn")
    assert list(series["MTM PnL"]) == [0.0] * 5


def test_market_detail_refuses_a_market_with_no_ticks(tmp_path):
    """`emit_ticks` is opt-in and `tick_markets` restricts it further, so
    asking for a market that was not emitted is the likely mistake and must
    say so rather than draw an empty axis."""
    t = load_ticks({"a": _ticks_dir(tmp_path, "a")})
    with pytest.raises(ValueError, match="tick_markets"):
        market_detail(t, pd.DataFrame(), "not_emitted",
                      str(tmp_path / "x.png"))


def test_market_detail_plots_btc_space_alongside_probability(tmp_path,
                                                             monkeypatch):
    """The BTC panel is the point. `spot` is BTC/USDT, `chainlink` is the
    BTC/USD oracle that settles the market, and `s` is the model's estimate
    of where that oracle lands. The gap between the first two is the basis
    the fair block has to learn, and a detail plot that omits it cannot show
    whether it did."""
    t = load_ticks({"a": _ticks_dir(tmp_path, "a")})
    _, series = _drawn(monkeypatch, t, pd.DataFrame(), "m0",
                       str(tmp_path / "m0.png"))
    assert series["venue spot"][0] == 70050.0
    assert series["chainlink"][0] == 70010.0
    assert series["s (fair BTC)"][0] == 70000.0
    # and the probability panel is still there, in probability units
    assert 0.0 <= series["fair_p"][0] <= 1.0


def test_market_detail_omits_the_oracle_line_when_none_was_loaded(tmp_path,
                                                                  monkeypatch):
    """`rtds_path` is optional, and the oracle capture does not span every
    day the panel does. An all-NaN column must drop the line rather than
    draw an empty one that reads as "the oracle was flat"."""
    d = _ticks_dir(tmp_path, "nocl")
    t = load_ticks({"a": d})
    t["chainlink"] = np.nan
    _, series = _drawn(monkeypatch, t, pd.DataFrame(), "m0",
                       str(tmp_path / "m0.png"))
    assert "chainlink" not in series
    assert "venue spot" in series


def test_market_detail_marks_buys_and_sells_on_the_right_sides(tmp_path,
                                                               monkeypatch):
    """A detail plot exists to answer "why did this trade", so the fills are
    the one thing that must be on it -- and on the correct side.

    `Side` is BUY=1, SELL=-1, so the obvious 0-and-1 guess labels every buy a
    sell and draws no buys at all. That shipped in the first draft here and
    was only visible because the position climbed while the legend claimed
    four sells, so this asserts the labels and counts, not just that markers
    exist."""
    d = _ticks_dir(tmp_path, "fills")
    t, ledger = load_ticks({"a": d}), load_ledgers({"a": d})
    fig, _ = _drawn(monkeypatch, t, ledger, "m0", str(tmp_path / "m0.png"))
    marked = {}
    for ax in fig.axes:
        for c in ax.collections:
            marked[c.get_label()] = len(c.get_offsets())
    assert marked == {"buy (1)": 1, "sell (1)": 1}, (
        f"expected one buy and one sell, drawn as such; got {marked}")
