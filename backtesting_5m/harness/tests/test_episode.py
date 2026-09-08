"""Pins the causality shift.

The panel README's trap: a row labelled t_ms holds an observation drawn from
[t_ms, t_ms+100), so it was NOT knowable at t_ms. Decision index i may only
see buckets k <= i-1. These tests are the enforcement.
"""
import numpy as np
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
    obs = __import__("pandas").DataFrame(
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


def test_settlement_is_absent_when_no_successor_exists(obs_sparse):
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, None,
                       obs_sparse, None, None)
    assert ep.settle is None and ep.winner_up is None


def test_winner_up_is_true_on_a_tie(obs_sparse):
    """settle >= strike, ties Up -- the panel README's rule."""
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_000.0,
                       obs_sparse, None, None)
    assert ep.winner_up is True
