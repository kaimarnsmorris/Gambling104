"""Unit tests for the 5 m book reconciliation primitives.

The build's own gates test it against real data; these pin the pieces to
inputs whose right answer is known by construction, so a failure localises.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

import common          # noqa: E402
import s03b_vantage    # noqa: E402
import s04_panel       # noqa: E402


# -- window arithmetic -----------------------------------------------------

def test_market_open_snaps_to_the_300s_boundary():
    assert common.market_open([1786665600, 1786665899.9, 1786665900]).tolist() \
        == [1786665600, 1786665600, 1786665900]


def test_era_start_is_the_first_60s_twap_market():
    """2026-08-14 00:00:00 UTC -- the market after the last 30 s one."""
    assert common.ERA_START == 1786665600
    assert common.ERA_START % common.H == 0
    assert pd.to_datetime(common.ERA_START, unit="s").strftime(
        "%Y-%m-%d %H:%M:%S") == "2026-08-14 00:00:00"


# -- quote hygiene ---------------------------------------------------------

def _q(ts, bid, ask, mid="m"):
    return pd.DataFrame({"market_id": mid, "ts": ts, "bid": bid, "ask": ask})


def test_write_quotes_drops_a_crossed_book(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA", str(tmp_path))
    monkeypatch.setitem(common.QUOTES, "archive",
                        str(tmp_path / "q.parquet"))
    out = common.write_quotes("archive",
                              _q([1.0, 2.0], [0.6, 0.4], [0.5, 0.45]))
    assert len(out) == 1 and out.bid.iloc[0] == 0.4


def test_write_quotes_drops_out_of_range_probabilities(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "DATA", str(tmp_path))
    monkeypatch.setitem(common.QUOTES, "archive", str(tmp_path / "q.parquet"))
    out = common.write_quotes("archive",
                              _q([1.0, 2.0], [0.4, -0.1], [0.5, 1.4]))
    assert len(out) == 1


# -- vantage calibration ---------------------------------------------------

def test_changepoints_keeps_only_uniquely_identifying_states():
    """A state that recurs in a market cannot identify one event."""
    d = pd.DataFrame({
        "market_id": "m",
        "ts": [1.0, 1.1, 2.0, 3.0],
        "bid": [0.40, 0.40, 0.41, 0.40],   # 0.40 appears twice: ambiguous
        "ask": [0.41, 0.41, 0.42, 0.41],
    })
    out = s03b_vantage.changepoints(d)
    assert out.bid.tolist() == [0.41]


def test_offsets_recovers_a_known_shift():
    ref = pd.DataFrame({"market_id": "m", "ts": [100.0, 200.0, 300.0],
                        "bid": [0.40, 0.41, 0.42],
                        "ask": [0.41, 0.42, 0.43]})
    other = ref.assign(ts=ref.ts + 0.050)          # 50 ms behind
    g = s03b_vantage.offsets(ref, other, "tdb_old")
    assert float(g.offset_ms.iloc[0]) == pytest.approx(50.0, abs=1e-6)


def test_offsets_ignores_a_match_further_than_the_window():
    ref = pd.DataFrame({"market_id": "m", "ts": [100.0],
                        "bid": [0.40], "ask": [0.41]})
    other = ref.assign(ts=[100.0 + 2 * s03b_vantage.MAX_GAP_S])
    g = s03b_vantage.offsets(ref, other, "tdb_old")
    assert g.empty or bool(g.n.sum() == 0)


# -- bucketing and assembly ------------------------------------------------

def _frame(src, mkt, t_ms, bid, ask, ts=None):
    t_ms = np.asarray(t_ms, dtype="int32")
    return pd.DataFrame({
        "mkt": np.full(len(t_ms), mkt, dtype="int32"),
        "t_ms": t_ms,
        "ts": (t_ms / 1000.0 if ts is None else np.asarray(ts, "float64")),
        "bid": np.asarray(bid, dtype="float32"),
        "ask": np.asarray(ask, dtype="float32"),
        "src": src,
    })


def test_assemble_prefers_the_archive():
    p = s04_panel.assemble([
        _frame("tdb_old", 0, [100], [0.30], [0.31]),
        _frame("archive", 0, [100], [0.40], [0.41]),
    ])
    row = p.loc[(0, 100)]
    assert row.src == "archive" and float(row.bid) == pytest.approx(0.40)


def test_assemble_counts_only_agreeing_captures():
    p = s04_panel.assemble([
        _frame("archive", 0, [100], [0.40], [0.41]),
        _frame("tdb_old", 0, [100], [0.405], [0.415]),   # within a cent
        _frame("tdb_live", 0, [100], [0.90], [0.91]),    # not
    ])
    row = p.loc[(0, 100)]
    assert int(row.n_src) == 2
    assert float(row.src_spread_c) == pytest.approx(50.0, abs=0.1)


def test_assemble_leaves_spread_null_for_a_lone_capture():
    p = s04_panel.assemble([_frame("archive", 0, [100], [0.40], [0.41])])
    assert pd.isna(p.loc[(0, 100)].src_spread_c)


def test_assemble_keeps_only_observed_buckets():
    """Nothing is carried forward: an unseen bucket is simply absent."""
    p = s04_panel.assemble([_frame("archive", 0, [0, 200], [0.4, 0.5],
                                   [0.41, 0.51])])
    assert p.index.get_level_values("t_ms").tolist() == [0, 200]


# -- market selection ------------------------------------------------------

def _panel_for(t_ms, mkt=0):
    idx = pd.MultiIndex.from_arrays(
        [np.full(len(t_ms), mkt, dtype="int32"),
         np.asarray(t_ms, dtype="int32")], names=["mkt", "t_ms"])
    return pd.DataFrame({"bid": 0.4}, index=idx)


def test_select_drops_a_market_truncated_at_the_close():
    full = np.arange(0, 200000, 100)                  # stops at 200 s
    st = s04_panel.select(_panel_for(full), {"M"}, {"M": 0})
    assert not bool(st.keep.iloc[0])
    assert st.reason.iloc[0] == "no_close_10s"


def test_select_drops_a_market_that_starts_late():
    late = np.arange(20000, 300000, 100)
    st = s04_panel.select(_panel_for(late), {"M"}, {"M": 0})
    assert st.reason.iloc[0] == "no_open_10s"


def test_select_drops_a_market_with_a_hole_in_the_middle():
    t = np.r_[np.arange(0, 100000, 100), np.arange(250000, 300000, 100)]
    st = s04_panel.select(_panel_for(t), {"M"}, {"M": 0})
    assert st.reason.iloc[0] == "coverage_below_90pct"


def test_select_keeps_a_complete_market():
    t = np.arange(0, 300000, 100)
    st = s04_panel.select(_panel_for(t), {"M"}, {"M": 0})
    assert bool(st.keep.iloc[0]) and st.reason.iloc[0] == ""


def test_select_drops_a_market_with_no_strike():
    t = np.arange(0, 300000, 100)
    st = s04_panel.select(_panel_for(t), set(), {"M": 0})
    assert st.reason.iloc[0] == "no_strike"


def test_select_tolerates_a_sparse_but_complete_market():
    """91 % coverage with both ends present survives; 89 % does not."""
    keep_n = int(common.N_BUCKET * 0.91)
    t = np.r_[np.arange(0, 5000, 100),
              np.linspace(5000, 299900, keep_n - 50).astype(int) // 100 * 100]
    st = s04_panel.select(_panel_for(np.unique(t)), {"M"}, {"M": 0})
    assert bool(st.keep.iloc[0])


# -- spike removal ---------------------------------------------------------

def test_drop_spikes_removes_a_roll_artefact_at_the_open():
    """0.00 for one bucket, 0.78 either side, is the previous market's book."""
    p = s04_panel.assemble([
        _frame("archive", 0, [6000, 6100, 6200], [0.00, 0.78, 0.78],
               [0.001, 0.79, 0.79])])
    out = s04_panel.drop_spikes(p)
    assert out.index.get_level_values("t_ms").tolist() == [6100, 6200]


def test_drop_spikes_keeps_a_book_that_is_extreme_but_stable():
    """Near expiry the binary really does sit at a bound -- that is not a spike."""
    p = s04_panel.assemble([
        _frame("archive", 0, [299700, 299800, 299900], [0.99, 0.99, 0.99],
               [1.0, 1.0, 1.0])])
    assert len(s04_panel.drop_spikes(p)) == 3


def test_drop_spikes_keeps_a_genuine_large_move():
    """A move that PERSISTS is a move, not an artefact."""
    p = s04_panel.assemble([
        _frame("archive", 0, [100, 200, 300, 400], [0.40, 0.40, 0.75, 0.75],
               [0.41, 0.41, 0.76, 0.76])])
    assert len(s04_panel.drop_spikes(p)) == 4


def test_drop_spikes_leaves_a_lone_observation_alone():
    """With no neighbour either side there is nothing to contradict it."""
    p = s04_panel.assemble([_frame("archive", 0, [100], [0.40], [0.41])])
    assert len(s04_panel.drop_spikes(p)) == 1
