"""The normal-QQ benchmark, pinned.

This model exists to prove the harness runs end to end, so the tests are
arithmetic wherever they can be: closed forms and hand-computed updates, never
a second copy of the implementation.

Blocks are loaded through `resolve_slots`, so these tests also prove the
investigation folder actually wins over `blocks/defaults`.
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(
    os.path.join(HERE, os.pardir, os.pardir, "backtesting_5m")))

from harness.core.provenance import load_slot, resolve_slots  # noqa: E402

RESOLVED = resolve_slots(HERE)
fair = load_slot(RESOLVED["fair"], "fair")
vol = load_slot(RESOLVED["vol"], "vol")
f = load_slot(RESOLVED["f"], "f")
link_mod = load_slot(RESOLVED["link"], "link")


def test_the_investigation_folder_supplies_all_four_model_slots():
    for slot in ("fair", "vol", "f", "link"):
        assert os.path.dirname(RESOLVED[slot]) == HERE


# --- link: Phi(QQ(z)) ------------------------------------------------------

def test_qq_is_zeroed_at_the_origin():
    """No constant term, so an at-the-money z maps to an at-the-money z."""
    assert link_mod.qq(0.0) == 0.0


def test_qq_at_one_is_the_sum_of_its_coefficients():
    assert link_mod.qq(1.0) == pytest.approx(sum(link_mod.QQ), rel=1e-12)


def test_qq_at_minus_one_alternates_by_parity():
    a5, a4, a3, a2, a1 = link_mod.QQ
    assert link_mod.qq(-1.0) == pytest.approx(-a5 + a4 - a3 + a2 - a1,
                                              rel=1e-12)


def test_qq_assigns_each_coefficient_to_its_own_power():
    a5, a4, a3, a2, a1 = link_mod.QQ
    expected = 32 * a5 + 16 * a4 + 8 * a3 + 4 * a2 + 2 * a1
    assert link_mod.qq(2.0) == pytest.approx(expected, rel=1e-12)


def test_link_is_one_half_at_the_strike():
    assert link_mod.link(0.0) == pytest.approx(0.5)


def test_link_is_the_normal_cdf_of_the_polynomial():
    z = 0.7
    expected = 0.5 * (1.0 + math.erf(link_mod.qq(z) / math.sqrt(2.0)))
    assert link_mod.link(z) == pytest.approx(expected, rel=1e-12)


def test_link_is_monotone_and_bounded():
    grid = np.linspace(-6.0, 6.0, 241)
    ps = [link_mod.link(float(z)) for z in grid]
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert all(b >= a for a, b in zip(ps, ps[1:]))


def test_link_saturates_without_overflowing_the_quintic():
    assert link_mod.link(1e6) == 1.0
    assert link_mod.link(-1e6) == 0.0


def test_link_propagates_a_missing_z_rather_than_calling_it_a_coin_flip():
    assert math.isnan(link_mod.link(float("nan")))


# --- f: the d2 standardisation ---------------------------------------------

def test_z_is_d2_with_the_minus_half_sigma_squared_drift():
    sigma_t = 0.005
    expected = (math.log(101_000.0 / 100_000.0) - 0.5 * sigma_t ** 2) / sigma_t
    assert f.standardise(101_000.0, 100_000.0, sigma_t) == pytest.approx(
        expected, rel=1e-12)


def test_z_at_the_strike_is_the_drift_term_alone_and_is_negative():
    """The half-sigma-squared term is what makes at-the-money pay under 0.5."""
    sigma_t = 0.005
    assert f.standardise(100_000.0, 100_000.0, sigma_t) == pytest.approx(
        -0.5 * sigma_t)


def test_z_is_negative_below_the_strike():
    assert f.standardise(99_000.0, 100_000.0, 0.005) < 0.0


def test_z_is_zero_when_sigma_is_not_positive():
    assert f.standardise(101_000.0, 100_000.0, 0.0) == 0.0
    assert f.standardise(101_000.0, 100_000.0, -1.0) == 0.0


def test_z_is_zero_when_sigma_is_missing():
    assert f.standardise(101_000.0, 100_000.0, float("nan")) == 0.0


def test_z_is_zero_for_a_level_the_logarithm_cannot_take():
    assert f.standardise(0.0, 100_000.0, 0.005) == 0.0
    assert f.standardise(-5.0, 100_000.0, 0.005) == 0.0


# --- episode helpers -------------------------------------------------------

def _episode(obs_t_ms, spot_values, strike=100_000.0, settle=None):
    """A synthetic episode carrying spot observed at `obs_t_ms`.

    Built through `build_episode` rather than by hand, so the causality shift
    and the carry-forward are the real ones: an observation in bucket k first
    becomes visible at decision index k+1, with age 0.
    """
    import pandas as pd

    from harness.core.episode import build_episode
    from harness.paths import N_BUCKET

    grid = np.arange(N_BUCKET) * 100
    obs = pd.DataFrame({"t_ms": grid, "bid": 0.49, "ask": 0.51,
                        "mid": 0.50, "n_src": 2})
    spot = pd.DataFrame({"t_ms": np.asarray(obs_t_ms, dtype="int64"),
                         "spot": np.asarray(spot_values, dtype="float64")})
    return build_episode("synthetic", 1786665600, "2026-08-14", strike,
                         settle, obs, spot=spot)


def _dense(rate, s0=100_000.0):
    """Spot in every bucket, growing by a constant log return per bucket."""
    from harness.paths import N_BUCKET

    steps = np.arange(N_BUCKET)
    return _episode(steps * 100, s0 * np.exp(rate * steps))


BUCKET_S = 0.1
YEAR_S = 365.25 * 24 * 3600
W = 60.0                    # the Chainlink TWAP window


def _tau_eff(i):
    """The effective time at index i. Pinned per regime by its own tests."""
    tau = 300.0 - i * BUCKET_S
    if tau >= W:
        return tau - 2.0 * W / 3.0
    return tau ** 3 / (3.0 * W * W)


# --- fair: E[settling TWAP] ------------------------------------------------

def test_s_is_missing_until_the_first_spot_becomes_visible():
    """Index 0 can see no bucket at all, so there is nothing to average."""
    s = fair.precompute(_dense(0.0))
    assert math.isnan(s[0])
    assert math.isfinite(s[1])


def test_s_seeds_at_the_first_observation_rather_than_at_zero():
    ep = _dense(4.5e-5)
    assert fair.precompute(ep)[1] == pytest.approx(100_000.0)


def test_s_lags_a_rising_spot():
    ep = _dense(4.5e-5)
    s = fair.precompute(ep)
    assert s[2000] < float(ep.spot[2000])


def test_s_only_moves_when_a_new_observation_arrives():
    """The panel carries the last spot forward; a carried value is not news.

    Spot is observed in buckets 0, 1 and 5, so decision indices 1, 2 and 6
    carry news and 3, 4, 5 carry only a repeat of bucket 1.
    """
    ep = _episode([0, 100, 500], [100_000.0, 100_010.0, 100_050.0])
    s = fair.precompute(ep)
    assert s[3] == s[2]
    assert s[4] == s[2]
    assert s[5] == s[2]
    assert s[6] != s[2]


def test_s_is_missing_when_no_spot_ever_arrives():
    from harness.paths import N_BUCKET

    ep = _episode([], [])
    assert np.isnan(fair.precompute(ep)).all()
    assert len(fair.precompute(ep)) == N_BUCKET


# --- vol: the EWMA of realised variance ------------------------------------

def test_sigma_is_missing_until_five_updates_have_landed():
    """Index 1 seeds; indices 2..6 are the first five returns."""
    sigma = vol.precompute(_dense(4.5e-5))
    assert np.isnan(sigma[:6]).all()
    assert math.isfinite(sigma[6])


def test_sigma_matches_the_closed_form_for_a_constant_return_path():
    """With a constant r every bucket the EWMA has a closed form:

        rv_k = (r^2 / dt) * (1 - (1 - alpha)^k)

    which is arithmetic the implementation never sees.
    """
    rate = 4.5e-5
    sigma = vol.precompute(_dense(rate))

    alpha = 1.0 - math.exp(-BUCKET_S / 100.0)
    per_sec = rate ** 2 / BUCKET_S

    for i in (6, 500, 2999):
        updates = i - 1
        rv = per_sec * (1.0 - (1.0 - alpha) ** updates)
        expected = math.sqrt(rv * YEAR_S) * math.sqrt(_tau_eff(i) / YEAR_S)
        assert sigma[i] == pytest.approx(expected, rel=1e-10)


def test_sigma_counts_observations_not_grid_ticks_and_uses_their_spacing():
    """Spot once a second: five updates take 51 indices, and dt is 1 s.

    If dt were hard-coded to the 100 ms bucket, both the warm-up index and the
    variance level would be wrong.
    """
    from harness.paths import N_BUCKET

    rate = 1.4e-4
    k = np.arange(0, N_BUCKET, 10)
    ep = _episode(k * 100, 100_000.0 * np.exp(rate * np.arange(len(k))))
    sigma = vol.precompute(ep)

    assert np.isnan(sigma[:51]).all()
    assert math.isfinite(sigma[51])

    alpha = 1.0 - math.exp(-1.0 / 100.0)
    rv = (rate ** 2 / 1.0) * (1.0 - (1.0 - alpha) ** 5)
    expected = math.sqrt(rv * YEAR_S) * math.sqrt(_tau_eff(51) / YEAR_S)
    assert sigma[51] == pytest.approx(expected, rel=1e-10)


def test_sigma_is_zero_not_missing_for_a_perfectly_flat_spot():
    sigma = vol.precompute(_dense(0.0))
    assert sigma[6] == 0.0
    assert sigma[2999] == 0.0


def test_sigma_is_missing_when_no_spot_ever_arrives():
    assert np.isnan(vol.precompute(_episode([], []))).all()


def test_sigma_is_neither_shrunk_nor_clamped_toward_a_prior():
    """A tiny path stays tiny and a violent one stays violent.

    Read back in annualised terms, because sigma_T at the last index is small
    for any path -- there is only 100 ms left to be volatile in. A clamp to
    anything like [0.15, 3.0] would crush both of these; a shrink toward a
    prior would pull the ratio well below the 1e6 the rates are apart.
    """
    i = 2999
    to_annual = math.sqrt(_tau_eff(i) / YEAR_S)
    quiet = vol.precompute(_dense(1e-8))[i] / to_annual
    wild = vol.precompute(_dense(1e-2))[i] / to_annual

    assert quiet < 1e-3
    assert wild > 100.0
    assert wild / quiet == pytest.approx(1e6, rel=1e-6)


# --- the whole model, through the real loop --------------------------------

quote_mod = load_slot(RESOLVED["quote"], "quote")
execution_mod = load_slot(RESOLVED["execution"], "execution")
fill_mod = load_slot(RESOLVED["fill"], "fill")
fees_mod = load_slot(RESOLVED["fees"], "fees")


def _run(ep, **kwargs):
    from harness.core.config import ExecConfig, QuoteParams
    from harness.core.loop import run_episode

    blocks = {
        "f": f.standardise, "link": link_mod.link,
        "quote": quote_mod.quotes, "execution": execution_mod.decide,
        "fill": fill_mod.resolve, "fees": fees_mod.FeeSchedule(),
        "s": fair.precompute(ep), "sigma": vol.precompute(ep),
    }
    params = QuoteParams(e_p=0.02, shares=10.0, max_pos=50.0)
    return run_episode(ep, blocks, params, ExecConfig(mode="both"),
                       seed=0, **kwargs)


def _drifting(rate):
    """A market whose spot walks steadily away from its strike."""
    from harness.paths import N_BUCKET

    steps = np.arange(N_BUCKET)
    path = 100_000.0 * np.exp(rate * steps)
    return _episode(steps * 100, path, strike=100_000.0,
                    settle=float(path[-1]))


def test_the_four_blocks_compose_through_the_run_loop():
    """Spot climbs away from the strike, so the model buys and settles UP."""
    out = _run(_drifting(4.5e-5))
    assert out["settled"] is True
    assert out["n_fills"] > 0
    assert math.isfinite(out["pnl_net"])
    assert out["max_abs_q"] <= 50.0
    assert all(fl["side"] == 1 for fl in out["fills"])
    assert out["pnl_gross"] > 0.0


def test_the_loop_stands_aside_until_the_vol_estimate_warms_up():
    """A NaN sigma must reach the loop as 'do not quote', not as a price."""
    ticks = _run(_drifting(4.5e-5), emit_ticks=True)["ticks"]
    assert all(math.isnan(t["eff_bid"]) for t in ticks[:6])
    assert math.isfinite(ticks[6]["eff_bid"])
    assert all(t["t_ms"] < 600 for t in ticks[:6])


def test_no_trade_can_be_dated_before_the_warm_up_ends():
    out = _run(_drifting(4.5e-5))
    assert min(fl["t_ms"] for fl in out["fills"]) >= 600


def test_a_spot_far_above_the_strike_prices_near_certainty():
    ticks = _run(_drifting(4.5e-5), emit_ticks=True)["ticks"]
    assert ticks[2500]["fair_p"] > 0.99


def test_a_spot_far_below_the_strike_prices_near_zero():
    ticks = _run(_drifting(-4.5e-5), emit_ticks=True)["ticks"]
    assert ticks[2500]["fair_p"] < 0.01


def test_a_market_with_no_spot_feed_trades_nothing_rather_than_guessing():
    from harness.paths import N_BUCKET

    ep = _episode([], [], settle=100_100.0)
    out = _run(ep, emit_ticks=True)
    assert out["n_fills"] == 0
    assert all(math.isnan(t["fair_p"]) for t in out["ticks"])
    assert len(out["ticks"]) == N_BUCKET


# --- the settling average, not the settling tick ---------------------------
#
# The market settles on a 60 s TWAP, so the quantity being predicted is an
# AVERAGE over [T-w, T], not the price at T. Two things follow, and these
# tests pin both: the variance of an average is smaller than the variance of
# its endpoint, and once the averaging window opens, part of the settlement is
# already known rather than forecast.

def _rv_closed(rate, updates):
    """The EWMA variance after `updates` constant returns, in closed form."""
    alpha = 1.0 - math.exp(-BUCKET_S / 100.0)
    return (rate ** 2 / BUCKET_S) * (1.0 - (1.0 - alpha) ** updates)


def _step_episode():
    """Spot flat at 100,000 until the averaging window opens, then 101,000.

    Bucket 2399 is the last at 100,000, so decision index 2400 -- the first
    index inside the window -- still sees the old level, and every later index
    sees the new one. That makes both the realised average and the smoothed
    spot exact rationals.
    """
    from harness.paths import N_BUCKET

    path = np.where(np.arange(N_BUCKET) < 2400, 100_000.0, 101_000.0)
    return _episode(np.arange(N_BUCKET) * 100, path, strike=100_000.0)


def test_the_effective_time_removes_two_thirds_of_the_window_before_averaging():
    """tau_eff = tau - 2w/3 while the whole average is still in the future.

    Dividing sigma^2 by the closed-form variance isolates the time weighting
    from the EWMA recursion, so this test moves only if the weighting moves.
    """
    rate = 4.5e-5
    sigma = vol.precompute(_dense(rate))
    for i in (6, 500, 2400):
        tau = 300.0 - i * BUCKET_S
        tau_eff = sigma[i] ** 2 / _rv_closed(rate, i - 1)
        assert tau_eff == pytest.approx(tau - 2.0 * W / 3.0, rel=1e-9)


def test_the_effective_time_is_cubic_once_the_averaging_has_begun():
    """Inside the window only the unelapsed tail is still random."""
    rate = 4.5e-5
    sigma = vol.precompute(_dense(rate))
    for i in (2700, 2900, 2999):
        tau = 300.0 - i * BUCKET_S
        tau_eff = sigma[i] ** 2 / _rv_closed(rate, i - 1)
        assert tau_eff == pytest.approx(tau ** 3 / (3.0 * W * W), rel=1e-9)


def test_the_two_effective_time_regimes_meet_where_the_window_opens():
    """tau = w gives w/3 from either branch, so sigma has no step in it."""
    rate = 4.5e-5
    sigma = vol.precompute(_dense(rate))
    tau_eff = sigma[2400] ** 2 / _rv_closed(rate, 2399)
    assert tau_eff == pytest.approx(W / 3.0, rel=1e-9)


def test_averaging_shrinks_sigma_hardest_near_expiry():
    """A tenth of a second from expiry the settling average is nearly fixed."""
    rate = 4.5e-5
    sigma = vol.precompute(_dense(rate))
    point = math.sqrt(_rv_closed(rate, 2998) * 0.1)
    assert sigma[2999] < 0.15 * point


def test_s_is_the_current_spot_while_the_average_is_still_in_the_future():
    """E[A] = S_t out there: the settling average is a martingale from here.

    The rising path is what discriminates. At this rate a 2 s smoother sits
    ~0.13 % under spot and a 60 s one ~3.9 % under it, so the tolerance admits
    a light denoise and rejects a lag that is impersonating the TWAP.
    """
    s = fair.precompute(_step_episode())
    assert s[2000] == pytest.approx(100_000.0, rel=1e-9)
    assert s[2400] == pytest.approx(100_000.0, rel=1e-9)

    ep = _dense(4.5e-5)
    assert fair.precompute(ep)[2000] == pytest.approx(
        float(ep.spot[2000]), rel=5e-3)


def test_s_smooths_the_spot_with_a_short_halflife():
    ep = _dense(4.5e-5)
    s = fair.precompute(ep)
    a = 1.0 - math.exp(-BUCKET_S * math.log(2.0) / fair.SPOT_SMOOTH_HALFLIFE_S)

    s2 = a * float(ep.spot[2]) + (1.0 - a) * 100_000.0
    s3 = a * float(ep.spot[3]) + (1.0 - a) * s2
    assert s[2] == pytest.approx(s2, rel=1e-12)
    assert s[3] == pytest.approx(s3, rel=1e-12)


def test_s_weights_the_realised_average_by_the_elapsed_fraction():
    """Inside the window the elapsed part is KNOWN, so it enters at full weight.

    At index 2999 the window holds one bucket at 100,000 and 599 at 101,000,
    and the 2 s smoother has had 60 s to reach 101,000, so every term is exact.
    """
    s = fair.precompute(_step_episode())
    twap = (100_000.0 + 599 * 101_000.0) / 600.0
    frac = 0.1 / W
    assert s[2999] == pytest.approx((1.0 - frac) * twap + frac * 101_000.0,
                                    rel=1e-9)


def test_s_is_pulled_toward_the_realised_average_and_away_from_spot():
    """Halfway through the window a fresh jump is only half believed.

    At index 2700 the window holds 1 bucket at 100,000 and 300 at 101,000, and
    tau/w is exactly a half.
    """
    s = fair.precompute(_step_episode())
    twap = (100_000.0 + 300 * 101_000.0) / 301.0
    assert s[2700] == pytest.approx(0.5 * twap + 0.5 * 101_000.0, rel=1e-6)
