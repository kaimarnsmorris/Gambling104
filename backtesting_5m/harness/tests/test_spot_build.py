"""Pins the spot build's bucketing and its clock gate.

The panel's captures were corrected onto polydata's vantage; stream_venue_l1
is a different host with its own drifting clock, and that offset cannot be
measured from these two streams (they carry receipts of different event
types). Joining them with an unvalidated configured offset would misalign
spot and book -- and beta collapses to zero inside tau = 13 s, so a silent
50 ms error is the difference between signal and noise.
"""
import numpy as np
import pandas as pd
import pytest

from harness.build.spot_5m_100ms import (MAX_BASIS_AGE_S, ClockGateError,
                                         basis_ok, bucket_venue_l1,
                                         require_offset)


def _venue(open_ts=1786665600, n=30, step=0.1, basis=0.0):
    ts = open_ts + np.arange(n) * step
    return pd.DataFrame({
        "ts": ts,
        "bn_spot_mid": 100_000.0 + np.arange(n),
        "usdt_basis": np.full(n, float(basis)),
        "bn_spot_bid_sz": np.full(n, 2.0),
        "bn_spot_ask_sz": np.full(n, 3.0),
    })


def test_observations_land_on_the_100ms_grid():
    out = bucket_venue_l1(_venue(), 1786665600)
    assert (out["t_ms"] % 100 == 0).all()
    assert out["t_ms"].min() >= 0 and out["t_ms"].max() < 300_000


def test_the_first_observation_in_a_bucket_wins():
    """Same rule as s04_panel.py, so the two grids mean the same thing."""
    df = pd.DataFrame({
        "ts": [1786665600.00, 1786665600.05, 1786665600.10],
        "bn_spot_mid": [100.0, 999.0, 200.0],
        "usdt_basis": [0.0, 0.0, 0.0],
        "bn_spot_bid_sz": [1.0, 1.0, 1.0],
        "bn_spot_ask_sz": [1.0, 1.0, 1.0],
    })
    out = bucket_venue_l1(df, 1786665600).set_index("t_ms")
    assert out.loc[0, "spot"] == 100.0
    assert out.loc[100, "spot"] == 200.0


def test_observations_outside_the_window_are_dropped():
    df = _venue(n=4)
    df.loc[0, "ts"] = 1786665599.0        # before the open
    df.loc[3, "ts"] = 1786665901.0        # after the close
    assert len(bucket_venue_l1(df, 1786665600)) == 2


def test_the_gate_rejects_a_day_with_no_configured_offset():
    with pytest.raises(ClockGateError, match="2026-08-20"):
        require_offset("2026-08-20", None)


def test_the_gate_rejects_a_nan_offset():
    with pytest.raises(ClockGateError, match="2026-08-20"):
        require_offset("2026-08-20", float("nan"))


def test_the_gate_rejects_an_implausible_offset():
    """Seconds of drift is a broken clock, not a vantage difference."""
    with pytest.raises(ClockGateError, match="implausible"):
        require_offset("2026-08-20", 5.0)


def test_the_gate_accepts_a_plausible_offset():
    assert require_offset("2026-08-20", 0.074) == pytest.approx(0.074)


def test_bucketing_survives_float_error_without_overcorrecting():
    """Pins both sides of the boundary problem.

    An observation exactly 0.1 s after the open must land in bucket 100 despite
    float64 giving 99.9999 ms -- and one genuinely at 99.6 ms must still land
    in bucket 0, which rounding to the nearest millisecond would break.
    """
    open_ts = 1786665600
    df = pd.DataFrame({
        "ts": [open_ts + 0.1, open_ts + 0.0996, open_ts + 0.2],
        "bn_spot_mid": [1.0, 2.0, 3.0],
        "usdt_basis": [0.0, 0.0, 0.0],
        "bn_spot_bid_sz": [1.0, 1.0, 1.0],
        "bn_spot_ask_sz": [1.0, 1.0, 1.0],
    })
    out = bucket_venue_l1(df, open_ts).set_index("t_ms")
    assert out.loc[100, "spot"] == 1.0, "exact 0.1 s must not fall back a bucket"
    assert out.loc[0, "spot"] == 2.0, "99.6 ms must not be rounded up a bucket"
    assert out.loc[200, "spot"] == 3.0


# --- the currency correction -------------------------------------------
#
# These markets settle on Chainlink BTC/USD; the venue quotes BTC/USDT. The
# uncorrected mid ran ~+$43 rich to the settlement feed, roughly 0.17 of a
# 300 s sigma and about four times the taker fee in probability terms, so the
# subtraction below is not cosmetic -- it is the difference between measuring
# an edge and manufacturing one.


def test_spot_is_the_mid_less_the_usdt_basis():
    """The pin: emitted `spot` is BTC/USD, and the inputs stay auditable."""
    df = pd.DataFrame({
        "ts": [1786665600.0, 1786665600.1, 1786665600.2],
        "bn_spot_mid": [100_000.0, 100_010.0, 100_020.0],
        "usdt_basis": [40.0, 41.5, 39.0],
        "bn_spot_bid_sz": [1.0, 1.0, 1.0],
        "bn_spot_ask_sz": [1.0, 1.0, 1.0],
    })
    out = bucket_venue_l1(df, 1786665600).set_index("t_ms")
    assert out.loc[0, "spot"] == pytest.approx(99_960.0)
    assert out.loc[100, "spot"] == pytest.approx(99_968.5)
    assert out.loc[200, "spot"] == pytest.approx(99_981.0)
    # the raw mid survives untouched, and the basis applied is recorded, so
    # the correction can be audited and undone row by row
    assert out.loc[100, "spot_usdt"] == pytest.approx(100_010.0)
    assert out.loc[100, "usdt_basis"] == pytest.approx(41.5)
    assert (out["spot_usdt"] - out["usdt_basis"]).values == pytest.approx(
        out["spot"].values)


def test_a_bucket_with_no_usable_basis_is_dropped_not_left_uncorrected():
    """A missing observation beats a $43-wrong one.

    Bucket 100's only observation has no basis, so that bucket must be absent
    from the output entirely -- NOT present carrying the raw BTC/USDT mid.
    """
    df = pd.DataFrame({
        "ts": [1786665600.0, 1786665600.1, 1786665600.2],
        "bn_spot_mid": [100_000.0, 100_010.0, 100_020.0],
        "usdt_basis": [40.0, np.nan, 39.0],
        "bn_spot_bid_sz": [1.0, 1.0, 1.0],
        "bn_spot_ask_sz": [1.0, 1.0, 1.0],
    })
    out = bucket_venue_l1(df, 1786665600)
    assert out["t_ms"].tolist() == [0, 200]
    assert 100_010.0 not in out["spot"].values


def test_a_later_usable_observation_still_represents_its_bucket():
    """Rejection is per observation, not per bucket.

    Both rows fall in bucket 0. The first has no basis, so the second -- a
    genuine USD observation 50 ms later -- represents the bucket. Only a
    bucket with no usable observation at all disappears.
    """
    df = pd.DataFrame({
        "ts": [1786665600.00, 1786665600.05],
        "bn_spot_mid": [100_000.0, 100_002.0],
        "usdt_basis": [np.nan, 40.0],
        "bn_spot_bid_sz": [1.0, 1.0],
        "bn_spot_ask_sz": [1.0, 1.0],
    })
    out = bucket_venue_l1(df, 1786665600).set_index("t_ms")
    assert out.loc[0, "spot"] == pytest.approx(99_962.0)


def test_a_stale_basis_is_rejected_where_the_capture_dates_it():
    """Optional columns, honored when present.

    stream_venue_l1 carries neither `usdt_basis_stale` nor `usdt_basis_ts`,
    so this is forward-compatible behaviour -- but a stale conversion rate is
    a wrong price just as surely as a missing one, and a null in a staleness
    column must never wave a row through.
    """
    df = pd.DataFrame({
        "ts": [1000.0, 1001.0, 1002.0, 1003.0],
        "bn_spot_mid": [100.0, 100.0, 100.0, 100.0],
        "usdt_basis": [40.0, 40.0, 40.0, 40.0],
        "usdt_basis_ts": [1000.0, 1001.0 - MAX_BASIS_AGE_S - 1.0, 1002.0,
                          np.nan],
        "bn_spot_bid_sz": [1.0, 1.0, 1.0, 1.0],
        "bn_spot_ask_sz": [1.0, 1.0, 1.0, 1.0],
    })
    assert basis_ok(df).tolist() == [True, False, True, False]

    flagged = df.drop(columns="usdt_basis_ts").assign(
        usdt_basis_stale=[0.0, 1.0, 0.0, np.nan])
    assert basis_ok(flagged).tolist() == [True, False, True, False]


def test_the_basis_gate_is_a_no_op_when_every_basis_is_good():
    assert basis_ok(_venue(n=5, basis=38.0)).all()
