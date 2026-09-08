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

from harness.build.spot_5m_100ms import (ClockGateError, bucket_venue_l1,
                                         require_offset)


def _venue(open_ts=1786665600, n=30, step=0.1):
    ts = open_ts + np.arange(n) * step
    return pd.DataFrame({
        "ts": ts,
        "bn_spot_mid": 100_000.0 + np.arange(n),
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
        "bn_spot_bid_sz": [1.0, 1.0, 1.0],
        "bn_spot_ask_sz": [1.0, 1.0, 1.0],
    })
    out = bucket_venue_l1(df, open_ts).set_index("t_ms")
    assert out.loc[100, "spot"] == 1.0, "exact 0.1 s must not fall back a bucket"
    assert out.loc[0, "spot"] == 2.0, "99.6 ms must not be rounded up a bucket"
    assert out.loc[200, "spot"] == 3.0
