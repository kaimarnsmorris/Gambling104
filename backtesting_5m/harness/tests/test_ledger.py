"""Pins the ledger's accounting and the 10 s markout.

delta_quality_c is the adverse-selection measure: where the mid went in the
10 s after we traded, signed so that positive is good and expressed in cents.
"""
import numpy as np
import pytest

from harness.blocks.defaults.fees import Liquidity
from harness.core.ledger import markout
from harness.core.types import Side


def test_a_buy_is_scored_by_where_the_mid_went(flat_episode):
    ep = flat_episode
    ep.mid[200:] = 0.60
    mid_t10, dq, settled = markout(ep, 100, Side.BUY, 0.50)
    assert mid_t10 == pytest.approx(0.60)
    assert dq == pytest.approx(10.0)      # cents
    assert settled is False


def test_a_sell_is_scored_with_the_opposite_sign(flat_episode):
    ep = flat_episode
    ep.mid[200:] = 0.60
    _, dq, _ = markout(ep, 100, Side.SELL, 0.50)
    assert dq == pytest.approx(-10.0)


def test_the_markout_horizon_is_ten_seconds(flat_episode):
    """100 decision indices at 100 ms each."""
    ep = flat_episode
    ep.mid[199] = 0.90
    ep.mid[200] = 0.60
    mid_t10, _, _ = markout(ep, 100, Side.BUY, 0.50)
    assert mid_t10 == pytest.approx(0.60)


def test_a_fill_near_expiry_marks_out_against_settlement(flat_episode):
    """Past the window there is no mid, so the outcome is the reference."""
    mid_t10, dq, settled = markout(flat_episode, 2950, Side.BUY, 0.50)
    assert settled is True
    assert mid_t10 == pytest.approx(1.0)      # flat_episode settles UP
    assert dq == pytest.approx(50.0)


def test_markout_is_absent_when_the_outcome_is_unknown(flat_episode):
    from dataclasses import replace
    ep = replace(flat_episode, winner_up=None, settle=None)
    mid_t10, dq, settled = markout(ep, 2950, Side.BUY, 0.50)
    assert np.isnan(mid_t10) and np.isnan(dq)
