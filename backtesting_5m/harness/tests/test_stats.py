"""Pins the measurement layer and the discipline gates.

The gates exist because the architecture doc says they are applied
'inconsistently'. Making them a function every run calls is the fix.
"""
import numpy as np
import pandas as pd
import pytest

from harness.core import stats


def _markets(pnl, days=None, shares=10.0):
    """Default day labels are CONTIGUOUS blocks of ten markets.

    Cycling labels would spread any mutation across every day, so a test that
    flips the last third of markets would leave every period's mean unchanged
    and the sign gate could never fail. Blocks keep calendar order meaningful.
    """
    pnl = np.asarray(pnl, dtype="float64")
    if days is None:
        days = [f"day-{i // 10:03d}" for i in range(len(pnl))]
    return pd.DataFrame({"day": days, "pnl_net": pnl,
                         "shares": np.full(len(pnl), shares),
                         "n_fills": np.ones(len(pnl))})


def test_headline_reports_per_market_and_per_share():
    m = _markets([1.0, 2.0, 3.0], shares=10.0)
    h = stats.headline(m)
    assert h["pnl_per_market"] == pytest.approx(2.0)
    assert h["c_per_share"] == pytest.approx(100.0 * 6.0 / 30.0)
    assert h["n_markets"] == 3


def test_markets_without_a_settlement_are_excluded():
    m = _markets([1.0, 2.0, np.nan])
    assert stats.headline(m)["n_markets"] == 2


def test_a_strong_positive_edge_has_a_ci_above_zero():
    rng = np.random.default_rng(0)
    m = _markets(rng.normal(1.0, 0.2, 400))
    lo, hi = stats.day_blocked_ci(m, n_boot=2000, seed=0)
    assert lo > 0.0 and hi > lo


def test_noise_rarely_looks_significant():
    """A 95 % CI on pure noise must span zero for the large majority of draws.

    Not for every draw -- that is exactly what "95 %" means, and with 40 day
    blocks the standard error is small enough that an unlucky sample mean sits
    two standard errors from zero. Asserting a single seed spans zero would be
    asserting something false about one draw in twenty; the honest claim is
    about the rate.
    """
    spans = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        m = _markets(rng.normal(0.0, 1.0, 400))
        lo, hi = stats.day_blocked_ci(m, n_boot=500, seed=0)
        spans += bool(lo < 0.0 < hi)
    assert spans >= 17, f"only {spans}/20 noise samples spanned zero"


def test_the_bootstrap_resamples_days_not_markets():
    """A single day that is wholly responsible for the edge must widen the CI.

    Market-level resampling would hide it; day-blocked resampling cannot.
    """
    pnl = [0.0] * 200 + [5.0] * 20
    days = ["2026-08-14"] * 200 + ["2026-08-15"] * 20
    lo, _ = stats.day_blocked_ci(_markets(pnl, days), n_boot=2000, seed=0)
    assert lo <= 0.0


def test_the_bootstrap_is_reproducible():
    m = _markets(np.random.default_rng(2).normal(0.5, 1.0, 200))
    assert stats.day_blocked_ci(m, n_boot=500, seed=7) == \
        stats.day_blocked_ci(m, n_boot=500, seed=7)


def test_the_sign_gate_fails_when_a_period_flips():
    good = _markets([1.0] * 90)
    assert stats.gate_sign_survives_periods(good)["passed"] is True

    flipped = good.copy()
    flipped.loc[60:, "pnl_net"] = -1.0
    assert stats.gate_sign_survives_periods(flipped)["passed"] is False


def test_the_delete_top_n_gate_catches_an_edge_carried_by_ten_markets():
    pnl = [0.0] * 200 + [50.0] * 10
    assert stats.gate_delete_top_n(_markets(pnl), n=10)["passed"] is False

    rng = np.random.default_rng(3)
    assert stats.gate_delete_top_n(_markets(rng.normal(2.0, 0.3, 210)),
                                   n=10)["passed"] is True


def test_run_gates_reports_all_four():
    m = _markets(np.random.default_rng(4).normal(1.0, 0.2, 300))
    result = stats.run_gates(m, n_boot=500)
    assert set(result["gates"]) == {
        "sign_survives_periods", "ci_excludes_zero", "delete_top_10"}
    assert isinstance(result["passed"], bool)


def test_gates_on_an_empty_sample_do_not_claim_success():
    result = stats.run_gates(_markets([]), n_boot=100)
    assert result["passed"] is False
