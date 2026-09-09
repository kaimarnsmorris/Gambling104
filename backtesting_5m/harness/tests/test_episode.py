"""Pins the causality shift.

The panel README's trap: a row labelled t_ms holds an observation drawn from
[t_ms, t_ms+100), so it was NOT knowable at t_ms. Decision index i may only
see buckets k <= i-1. These tests are the enforcement.
"""
import numpy as np
import pandas as pd
import pytest

from harness.core.episode import build_episode, shift_to_decision_grid
from harness.paths import N_BUCKET


def test_a_bucket_is_not_visible_at_its_own_index():
    values = np.array([np.nan] * N_BUCKET)
    present = np.zeros(N_BUCKET, dtype=bool)
    values[0], present[0] = 0.40, True
    carried, age = shift_to_decision_grid(values, present)
    assert np.isnan(carried[0]), "bucket 0 leaked into decision index 0"
    assert carried[1] == 0.40


def test_a_fresh_observation_has_zero_age_when_first_usable():
    values = np.full(N_BUCKET, np.nan)
    present = np.zeros(N_BUCKET, dtype=bool)
    values[7], present[7] = 0.44, True
    carried, age = shift_to_decision_grid(values, present)
    assert carried[8] == 0.44 and age[8] == 0.0
    assert carried[9] == 0.44 and age[9] == 100.0


def test_gaps_carry_the_last_quote_with_a_growing_age(obs_sparse):
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_100.0,
                       obs_sparse, None, None)
    assert np.isnan(ep.bid[0])
    assert ep.bid[1] == pytest.approx(0.40) and ep.book_age_ms[1] == 0.0
    assert ep.bid[2] == pytest.approx(0.41) and ep.book_age_ms[2] == 0.0
    # bucket 5 is not usable until index 6; indices 3..5 still hold bucket 1
    assert ep.bid[5] == pytest.approx(0.41) and ep.book_age_ms[5] == 300.0
    assert ep.bid[6] == pytest.approx(0.45) and ep.book_age_ms[6] == 0.0


def test_has_book_is_false_before_the_first_observation():
    obs = pd.DataFrame(
        {"t_ms": [400], "bid": [0.4], "ask": [0.5], "mid": [0.45], "n_src": [1]})
    ep = build_episode("m", 1786665600, "2026-08-14", 1.0, 2.0, obs, None, None)
    assert not ep.has_book[:5].any()
    assert ep.has_book[5:].all()


def test_no_future_leak_property():
    """Perturbing bucket k must never change any decision index <= k.

    This is the property that makes lookahead unrepresentable rather than
    merely discouraged. If it ever fails, every result in the repo is void.
    """
    rng = np.random.default_rng(0)
    present = rng.random(N_BUCKET) < 0.9
    values = np.where(present, rng.random(N_BUCKET), np.nan)
    base, _ = shift_to_decision_grid(values, present)
    for k in (0, 1, 37, 1500, N_BUCKET - 1):
        if not present[k]:
            continue
        bumped = values.copy()
        bumped[k] += 10.0
        after, _ = shift_to_decision_grid(bumped, present)
        np.testing.assert_array_equal(
            np.nan_to_num(base[: k + 1], nan=-1.0),
            np.nan_to_num(after[: k + 1], nan=-1.0),
            err_msg=f"bucket {k} leaked backwards")


def test_the_fair_column_obeys_the_same_no_future_leak_property(obs_sparse):
    """`s` is the alpha column, so it needs the property `bid` has.

    The export labels a value at t_ms with the bucket it was drawn from, the
    same convention the panel uses, so an unshifted `s` would put a future
    observation at the decision index that acts on it -- lookahead in exactly
    the column a strategy is built from.
    """
    rng = np.random.default_rng(3)
    values = 100_000.0 + rng.random(N_BUCKET) * 100.0

    def fair_of(arr):
        return build_episode("m", 1786665600, "2026-08-14", 100_000.0,
                             100_100.0, obs_sparse, None, arr).s

    base = fair_of(values)
    for k in (0, 1, 37, 1500, N_BUCKET - 1):
        bumped = values.copy()
        bumped[k] += 500.0
        after = fair_of(bumped)
        np.testing.assert_array_equal(
            np.nan_to_num(base[: k + 1], nan=-1.0),
            np.nan_to_num(after[: k + 1], nan=-1.0),
            err_msg=f"fair bucket {k} leaked backwards")


def test_the_fair_column_is_shifted_by_one_bucket_by_default(obs_sparse):
    values = np.arange(N_BUCKET, dtype="float64")
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_100.0,
                       obs_sparse, None, values)
    assert np.isnan(ep.s[0]), "bucket 0 leaked into decision index 0"
    assert ep.s[1] == pytest.approx(0.0)
    assert ep.s[2999] == pytest.approx(2998.0)


def test_fair_is_causal_opts_out_of_the_shift(obs_sparse):
    """Only legitimate once the export's timestamp contract is confirmed."""
    values = np.arange(N_BUCKET, dtype="float64")
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_100.0,
                       obs_sparse, None, values, fair_is_causal=True)
    np.testing.assert_array_equal(ep.s, values)


def test_settlement_is_absent_when_no_successor_exists(obs_sparse):
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, None,
                       obs_sparse, None, None)
    assert ep.settle is None and ep.winner_up is None


def test_winner_up_is_true_on_a_tie(obs_sparse):
    """settle >= strike, ties Up -- the panel README's rule."""
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_000.0,
                       obs_sparse, None, None)
    assert ep.winner_up is True


# --- the warm-up region ------------------------------------------------
#
# An Episode is one market, so a stateful estimator used to restart every
# 300 s. Warm-up gives `precompute` the seconds before the open on the same
# grid -- under the same causality shift, and without touching a single value
# the trading loop reads.


def _warm(t_ms, spot=None, usdt=None):
    out = {"t_ms": t_ms}
    if spot is not None:
        out["spot"] = spot
    if usdt is not None:
        out["spot_usdt"] = usdt
    return pd.DataFrame(out)


def test_no_warm_up_is_requested_by_default(obs_sparse):
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_100.0,
                       obs_sparse, None, None)
    assert ep.warmup_n == 0 and ep.warmup_s == 0.0
    assert ep.has_warmup is False
    assert ep.warmup_spot.shape == (0,)
    assert ep.warmup_chainlink.shape == (0,)


def test_the_in_window_arrays_are_untouched_by_warm_up(obs_sparse):
    """The whole design rests on this: warm-up cannot move a traded value.

    The warm-up region is shifted on its own grid rather than concatenated
    with the window, so every array the loop reads is bit for bit what it was
    before warm-up existed.
    """
    spot = _warm([0, 100, 200], spot=[100.0, 101.0, 102.0])
    cold = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_100.0,
                         obs_sparse, spot, None)
    warm = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_100.0,
                         obs_sparse, spot, None,
                         warmup_spot=_warm([0, 5000, 899_900],
                                           spot=[1.0, 2.0, 3.0]),
                         warmup_s=900.0, has_warmup=True)
    for name in ("bid", "ask", "mid", "book_age_ms", "n_src", "s",
                 "spot", "spot_age_ms", "spot_usdt", "chainlink",
                 "chainlink_age_ms"):
        np.testing.assert_array_equal(
            np.nan_to_num(getattr(cold, name), nan=-1.0, posinf=-2.0),
            np.nan_to_num(getattr(warm, name), nan=-1.0, posinf=-2.0),
            err_msg="warm-up moved the in-window column " + name)
    np.testing.assert_array_equal(cold.has_book, warm.has_book)
    np.testing.assert_array_equal(cold.has_spot, warm.has_spot)


def test_a_warm_up_bucket_is_not_visible_at_its_own_warm_up_index(obs_sparse):
    """Identical causality treatment: warm-up index j sees buckets <= j-1."""
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_100.0,
                       obs_sparse, None, None,
                       warmup_spot=_warm([0, 100], spot=[50.0, 51.0]),
                       warmup_s=900.0, has_warmup=True)
    assert ep.warmup_n == 9000
    assert np.isnan(ep.warmup_spot[0]), "warm-up bucket 0 leaked into index 0"
    assert ep.warmup_spot[1] == pytest.approx(50.0)
    assert ep.warmup_spot_age_ms[1] == 0.0
    assert ep.warmup_spot[2] == pytest.approx(51.0)
    assert ep.warmup_spot[3] == pytest.approx(51.0)
    assert ep.warmup_spot_age_ms[3] == 100.0


def test_the_last_pre_open_bucket_never_reaches_the_window(obs_sparse):
    """The seam is deliberately lossy, and this pins which way.

    An observation in the final warm-up bucket is knowable before the open,
    but carrying it into in-window index 0 would change what the loop sees.
    So it is dropped, and index 0 stays NaN exactly as it always was.
    """
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_100.0,
                       obs_sparse, None, None,
                       warmup_spot=_warm([899_900], spot=[77.0]),
                       warmup_s=900.0, has_warmup=True)
    assert np.isnan(ep.spot[0]) and np.isnan(ep.spot[1])
    assert np.isnan(ep.warmup_spot[-1]), "its own index cannot see it either"
    assert np.isfinite(ep.warmup_spot).sum() == 0


def test_warm_up_carries_the_raw_mid_and_the_oracle_too(obs_sparse):
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_100.0,
                       obs_sparse, None, None,
                       warmup_spot=_warm([0], spot=[10.0], usdt=[50.0]),
                       warmup_chainlink=pd.DataFrame({"t_ms": [0],
                                                      "px": [9.5]}),
                       warmup_s=900.0, has_warmup=True)
    assert ep.warmup_spot[1] == pytest.approx(10.0)
    assert ep.warmup_spot_usdt[1] == pytest.approx(50.0)
    assert ep.warmup_chainlink[1] == pytest.approx(9.5)
    assert ep.warmup_chainlink_age_ms[1] == 0.0


def test_a_requested_warm_up_with_no_data_is_nan_not_short(obs_sparse):
    """Never truncate, never pad. The shape is the contract."""
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_100.0,
                       obs_sparse, None, None, warmup_s=300.0,
                       has_warmup=True)
    assert ep.warmup_n == 3000
    assert ep.warmup_spot.shape == (3000,)
    assert ep.warmup_chainlink.shape == (3000,)
    assert not np.isfinite(ep.warmup_spot).any()
    assert not np.isfinite(ep.warmup_chainlink).any()
    assert ep.has_warmup is True, "the region exists; the feed was down"


def test_has_warmup_is_a_boundary_flag_not_a_data_flag(obs_sparse):
    """Two all-NaN regions that mean different things.

    `has_warmup` False says no history could exist here at all -- the start of
    the sample. True with an empty array says history exists and this feed was
    absent through it. A block that conflates them treats the start of the
    sample as an outage.
    """
    edge = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_100.0,
                         obs_sparse, None, None, warmup_s=300.0,
                         has_warmup=False)
    outage = build_episode("m", 1786665600, "2026-08-14", 100_000.0,
                           100_100.0, obs_sparse, None, None, warmup_s=300.0,
                           has_warmup=True)
    assert not np.isfinite(edge.warmup_chainlink).any()
    assert not np.isfinite(outage.warmup_chainlink).any()
    assert edge.has_warmup is False and outage.has_warmup is True


def test_has_warmup_cannot_be_true_without_a_region(obs_sparse):
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_100.0,
                       obs_sparse, None, None, warmup_s=0.0, has_warmup=True)
    assert ep.has_warmup is False and ep.warmup_n == 0


def test_warmup_tte_counts_down_to_the_open(obs_sparse):
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_100.0,
                       obs_sparse, None, None, warmup_s=900.0,
                       has_warmup=True)
    assert ep.warmup_tte_s(0) == pytest.approx(900.0)
    assert ep.warmup_tte_s(ep.warmup_n - 1) == pytest.approx(0.1)


def test_warm_up_has_the_no_future_leak_property_on_its_own_grid():
    """The same property the window has, enforced on the warm-up region."""
    rng = np.random.default_rng(11)
    n = 900
    t_ms = np.arange(n) * 100
    values = 100_000.0 + rng.random(n) * 50.0
    obs = pd.DataFrame({"t_ms": [0], "bid": [0.4], "ask": [0.5],
                        "mid": [0.45], "n_src": [1]})

    def warm_of(arr):
        return build_episode(
            "m", 1786665600, "2026-08-14", 100_000.0, 100_100.0, obs,
            None, None,
            warmup_spot=pd.DataFrame({"t_ms": t_ms, "spot": arr}),
            warmup_s=90.0, has_warmup=True).warmup_spot

    base = warm_of(values)
    for k in (0, 1, 37, n - 1):
        bumped = values.copy()
        bumped[k] += 500.0
        after = warm_of(bumped)
        np.testing.assert_array_equal(
            np.nan_to_num(base[: k + 1], nan=-1.0),
            np.nan_to_num(after[: k + 1], nan=-1.0),
            err_msg="warm-up bucket %d leaked backwards" % k)
