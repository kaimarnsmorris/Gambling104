"""Pins latency behaviour.

Two properties matter. The taker path can never be faster than the venue's
250 ms non-cancellable hold, and a market's draws depend only on its own id
and the run seed -- never on how many workers ran or in what order.
"""
import numpy as np
import pytest

from harness.core.latency import LatencyModel, episode_rng


def test_defaults_match_the_spec():
    m = LatencyModel()
    assert (m.place_ms, m.cancel_ms, m.take_ms) == (100.0, 100.0, 200.0)
    assert m.taker_lock_ms == 250.0


def test_taker_never_beats_the_venue_lock():
    """The 250 ms hold is a venue mechanic, not a latency assumption."""
    m = LatencyModel(take_ms=50.0, jitter_frac=0.5)
    rng = np.random.default_rng(0)
    draws = [m.draw(rng, "take") for _ in range(200)]
    assert min(draws) >= 250.0


def test_jitter_is_off_by_default():
    m = LatencyModel()
    rng = np.random.default_rng(0)
    assert {m.draw(rng, "place") for _ in range(20)} == {100.0}


def test_jitter_spreads_but_stays_positive():
    m = LatencyModel(jitter_frac=0.3)
    rng = np.random.default_rng(1)
    draws = np.array([m.draw(rng, "place") for _ in range(500)])
    assert draws.std() > 0.0 and draws.min() > 0.0


def test_cancel_is_slower_inside_a_move():
    """Measured 28 ms quiet against 73-165 ms in a move -- the adverse
    selection mechanism itself."""
    m = LatencyModel(cancel_ms=28.0, move_cancel_ms=120.0)
    rng = np.random.default_rng(0)
    assert m.draw(rng, "cancel", in_move=True) > m.draw(rng, "cancel")


def test_the_same_market_and_seed_give_identical_draws():
    a = [LatencyModel(jitter_frac=0.3).draw(episode_rng("mkt-1", 7), "place")
         for _ in range(3)]
    b = [LatencyModel(jitter_frac=0.3).draw(episode_rng("mkt-1", 7), "place")
         for _ in range(3)]
    assert a == b


def test_different_markets_do_not_share_a_stream():
    assert (LatencyModel(jitter_frac=0.3).draw(episode_rng("mkt-1", 7), "place")
            != LatencyModel(jitter_frac=0.3).draw(episode_rng("mkt-2", 7), "place"))


def test_delay_index_rounds_up_to_whole_buckets():
    m = LatencyModel()
    assert m.delay_idx(0.0) == 0
    assert m.delay_idx(1.0) == 1
    assert m.delay_idx(100.0) == 1
    assert m.delay_idx(250.0) == 3
