"""Pins the London build's currency, its freshness rule and its grid checks.

The dangerous property of this panel is that its `spot` column is BTC/**USDT**
while `paths.SPOT`'s is BTC/**USD**, under the same name and the same dtype.
Nothing downstream can tell them apart, so the tests that matter most here are
the ones that pin what the column contains and what it deliberately does NOT
contain.
"""
import numpy as np
import pandas as pd
import pytest

from harness import paths
from harness.build.spot_5m_100ms_london import (GridPhaseError, bucket_london,
                                                check_grid_phase,
                                                observations,
                                                verify_float_bucketing)

OPEN = 1786665600                       # a real 300 s boundary: OPEN % 300 == 0
NS = 1_000_000_000


def _london(open_ts=OPEN, n=30, spot_n=1):
    """A slice of the source panel: exact 100 ms grid, one row per bucket."""
    ts_ns = open_ts * NS + np.arange(n, dtype="int64") * 100_000_000
    return pd.DataFrame({
        "ts_ns": ts_ns,
        "spot_bid": 100_000.0 + np.arange(n),
        "spot_bid_sz": np.full(n, 2.0),
        "spot_ask": 100_002.0 + np.arange(n),
        "spot_ask_sz": np.full(n, 3.0),
        "spot_n": np.full(n, spot_n, dtype="int64"),
    })


# --- the grid ---------------------------------------------------------------

def test_observations_land_on_the_100ms_grid():
    out = bucket_london(_london(), OPEN)
    assert (out["t_ms"] % 100 == 0).all()
    assert out["t_ms"].min() >= 0 and out["t_ms"].max() < 300_000


def test_observations_outside_the_window_are_dropped():
    df = _london(n=4)
    df.loc[0, "ts_ns"] = (OPEN - 1) * NS            # before the open
    df.loc[3, "ts_ns"] = (OPEN + 301) * NS          # after the close
    assert len(bucket_london(df, OPEN)) == 2


def test_the_first_observation_in_a_bucket_wins():
    """Same rule as the venue build, so the two panels mean the same thing."""
    df = pd.DataFrame({
        "ts_ns": [OPEN * NS, OPEN * NS + 50_000_000, OPEN * NS + 100_000_000],
        "spot_bid": [100.0, 999.0, 200.0],
        "spot_bid_sz": [1.0, 1.0, 1.0],
        "spot_ask": [100.0, 999.0, 200.0],
        "spot_ask_sz": [1.0, 1.0, 1.0],
        "spot_n": [1, 1, 1],
    })
    out = bucket_london(df, OPEN).set_index("t_ms")
    assert out.loc[0, "spot"] == 100.0
    assert out.loc[100, "spot"] == 200.0


def test_bucketing_survives_float_error_without_overcorrecting():
    """Every row of this source sits exactly on a boundary, so EPS is load
    bearing on every single row -- not on an occasional edge case.

    The second row is 0.4 ms short of the boundary and must NOT be rounded up,
    which is what pins the epsilon at bucket scale rather than at millisecond
    scale.
    """
    df = pd.DataFrame({
        "ts_ns": [OPEN * NS + 100_000_000,      # exactly 0.1 s after the open
                  OPEN * NS + 99_600_000,       # 99.6 ms -- still bucket 0
                  OPEN * NS + 200_000_000],
        "spot_bid": [1.0, 2.0, 3.0],
        "spot_bid_sz": [1.0, 1.0, 1.0],
        "spot_ask": [1.0, 2.0, 3.0],
        "spot_ask_sz": [1.0, 1.0, 1.0],
        "spot_n": [1, 1, 1],
    })
    out = bucket_london(df, OPEN).set_index("t_ms")
    assert out.loc[100, "spot"] == 1.0, "exact 0.1 s must not fall back a bucket"
    assert out.loc[0, "spot"] == 2.0, "99.6 ms must not be rounded up a bucket"
    assert out.loc[200, "spot"] == 3.0


def test_the_float_rule_agrees_with_exact_integer_arithmetic():
    """`ts_ns` is exact, so the inherited float rule can be checked, not trusted."""
    df = _london(n=3000)
    assert verify_float_bucketing(df["ts_ns"].to_numpy(), OPEN, 0.0) == 0


def test_a_disagreement_with_integer_arithmetic_raises():
    """The guard has to be able to fire, or it is decoration.

    EPS is 1e-5 BUCKETS = 1 microsecond, so an observation 500 ns BELOW a
    boundary is the one case where the float rule and exact arithmetic part
    company: the integer floor keeps it in the lower bucket, EPS pushes it up.
    That row cannot occur in this source (every `ts_ns` is a whole 100 ms), and
    the build must refuse it rather than emit a bucket it cannot justify.
    """
    ts_ns = np.array([OPEN * NS + 100_000_000 - 500], dtype="int64")
    with pytest.raises(GridPhaseError, match="int64"):
        verify_float_bucketing(ts_ns, OPEN, 0.0)


def test_the_grid_phase_check_accepts_an_aligned_source():
    ts_ns = OPEN * NS + np.arange(10, dtype="int64") * 100_000_000
    info = check_grid_phase(ts_ns, [OPEN, OPEN + 300], 0.0)
    assert info["aligned"] and info["ts_phase_ns"] == [0]


def test_the_grid_phase_check_rejects_an_offset_source():
    """A 37 ms phase would silently shift every observation by a third of a
    bucket, so it must raise rather than be absorbed."""
    ts_ns = OPEN * NS + 37_000_000 + np.arange(10, dtype="int64") * 100_000_000
    with pytest.raises(GridPhaseError, match="does not align"):
        check_grid_phase(ts_ns, [OPEN], 0.0)


# --- the freshness rule -----------------------------------------------------

def test_a_forward_filled_row_is_not_an_observation():
    """`spot_n == 0` is a carry, not a quote.

    Emitting it would peg `spot_age_ms` at zero and hide every outage in the
    source, including the 683 s Binance-spot gap on 2026-08-15.
    """
    df = _london(n=3)
    df.loc[1, "spot_n"] = 0
    out = bucket_london(df, OPEN)
    assert list(out["t_ms"]) == [0, 200]


def test_a_bucket_of_pure_carry_disappears_rather_than_being_invented():
    df = _london(n=3, spot_n=0)
    assert len(bucket_london(df, OPEN)) == 0


def test_a_non_finite_or_non_positive_quote_is_dropped():
    df = _london(n=4)
    df.loc[1, "spot_bid"] = np.nan
    df.loc[2, "spot_ask"] = 0.0
    assert list(bucket_london(df, OPEN)["t_ms"]) == [0, 300]


def test_observations_mask_is_the_conjunction_of_fresh_and_finite():
    df = _london(n=3)
    df.loc[0, "spot_n"] = 0
    df.loc[1, "spot_ask"] = np.inf
    assert list(observations(df)) == [False, False, True]


# --- the currency, which is the whole point ---------------------------------

def test_spot_is_the_raw_usdt_mid_and_is_NOT_usd():
    """The pin. `spot` here is BTC/USDT with NO basis subtracted.

    On `paths.SPOT` the same column is BTC/USD. The two differ by ~$43 and
    nothing downstream can tell them apart, so this test is the contract.
    """
    df = pd.DataFrame({
        "ts_ns": [OPEN * NS, OPEN * NS + 100_000_000],
        "spot_bid": [100_000.0, 100_010.0],
        "spot_bid_sz": [1.0, 1.0],
        "spot_ask": [100_002.0, 100_014.0],
        "spot_ask_sz": [1.0, 1.0],
        "spot_n": [1, 1],
    })
    out = bucket_london(df, OPEN).set_index("t_ms")
    assert out.loc[0, "spot"] == pytest.approx(100_001.0)
    assert out.loc[100, "spot"] == pytest.approx(100_012.0)


def test_spot_and_spot_usdt_are_the_same_number():
    """There is no basis in this source, so the two columns must not diverge."""
    out = bucket_london(_london(), OPEN)
    assert out["spot"].to_numpy() == pytest.approx(out["spot_usdt"].to_numpy())


def test_no_usdt_basis_column_is_emitted():
    """An ABSENT column raises at the first consumer that assumes a USD basis;
    a NaN one would make `spot_usdt - usdt_basis` evaluate to NaN in silence."""
    assert "usdt_basis" not in bucket_london(_london(), OPEN).columns


def test_the_schema_matches_what_load_episodes_reads():
    out = bucket_london(_london(), OPEN)
    assert list(out.columns) == ["open_ts", "t_ms", "spot", "spot_usdt",
                                 "bid_sz", "ask_sz"]


def test_the_l1_sizes_are_carried_through_for_the_imbalance_signal():
    """`I = ln(bid_sz / ask_sz)` needs both, so neither may be dropped."""
    out = bucket_london(_london(), OPEN)
    assert (out["bid_sz"] == 2.0).all() and (out["ask_sz"] == 3.0).all()


def test_the_panel_name_says_usdt():
    """A reader who only ever sees the filename must still be told."""
    assert "usdt" in paths.SPOT_LONDON.lower()
    assert paths.SPOT_LONDON == paths.SPOT_LONDON_USDT
    assert paths.SPOT_LONDON != paths.SPOT
