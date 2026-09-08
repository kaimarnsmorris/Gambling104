"""Pins the fill models.

A resting order fills at ITS OWN price when the book crosses it. Adverse
selection does not come from a worse fill price -- it comes from the cancel
losing the race, which is why `cancel_at` is an index and not a flag.

None of this is measurable from the panel: there is no depth, no trade tape
and no queue. These models are assumptions, which is why runs must report a
range across them rather than a single maker number.
"""
import numpy as np
import pytest

from harness.blocks.defaults.fees import Liquidity
from harness.blocks.defaults.fill import resolve
from harness.core.types import Order, Side


def _order(side=Side.BUY, price=0.49, live_from=0, cancel_at=None,
           liquidity=Liquidity.MAKER, expires_at=None):
    return Order(order_id=1, side=side, price=price, shares=10.0,
                 liquidity=liquidity, live_from=live_from,
                 cancel_at=cancel_at, reason="test", expires_at=expires_at)


def test_a_resting_bid_fills_when_the_ask_reaches_it(flat_episode):
    ep = flat_episode
    ep.ask[5] = 0.49
    fills = resolve([_order(price=0.49)], ep, 5, {})
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(0.49)
    assert fills[0].liquidity == Liquidity.MAKER


def test_a_resting_bid_does_not_fill_above_the_ask(flat_episode):
    assert resolve([_order(price=0.45)], flat_episode, 5, {}) == []


def test_an_order_is_dead_before_it_is_live(flat_episode):
    ep = flat_episode
    ep.ask[5] = 0.49
    assert resolve([_order(price=0.49, live_from=6)], ep, 5, {}) == []


def test_a_cancelled_order_still_fills_inside_the_latency_window(flat_episode):
    """The whole adverse-selection mechanism, in one test.

    We decided to pull at index 4; the cancel lands at 7. A cross at 5 fills
    us anyway -- and it is exactly when the book is moving against us that
    the cross happens.
    """
    ep = flat_episode
    ep.ask[5] = 0.49
    fills = resolve([_order(price=0.49, cancel_at=7)], ep, 5, {})
    assert len(fills) == 1


def test_a_cancel_that_lands_first_prevents_the_fill(flat_episode):
    ep = flat_episode
    ep.ask[8] = 0.49
    assert resolve([_order(price=0.49, cancel_at=7)], ep, 8, {}) == []


def test_a_resting_ask_fills_when_the_bid_reaches_it(flat_episode):
    ep = flat_episode
    ep.bid[5] = 0.51
    fills = resolve([_order(side=Side.SELL, price=0.51)], ep, 5, {})
    assert len(fills) == 1


def test_penetration_requires_the_book_to_trade_through(flat_episode):
    """The conservative arm: a touch is not enough, it must go past us."""
    ep = flat_episode
    ep.ask[5] = 0.49
    assert resolve([_order(price=0.49)], ep, 5, {"penetration": 0.01}) == []
    ep.ask[5] = 0.48
    assert len(resolve([_order(price=0.49)], ep, 5, {"penetration": 0.01})) == 1


def test_a_taker_order_fills_at_the_book_not_at_its_limit(flat_episode):
    """Marketable orders pay the book. The limit only protects the worst case."""
    ep = flat_episode
    fills = resolve([_order(price=0.55, liquidity=Liquidity.TAKER)], ep, 5, {})
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(ep.ask[5])


def test_a_taker_order_does_not_fill_through_its_limit(flat_episode):
    """The book ran away during the 250 ms lock -- the limit saves us."""
    ep = flat_episode
    ep.ask[5] = 0.60
    assert resolve([_order(price=0.55, liquidity=Liquidity.TAKER)], ep, 5, {}) == []


def test_an_order_does_not_trade_past_its_expiry(flat_episode):
    """A marketable order is live for the tick it arrives on, and no longer.

    The engine stamps `expires_at = live_from + 1` on every cross: the venue
    held it through the lock, and if the book has moved away by the time it
    arrives there is nothing left to execute against. Without this it goes on
    holding per-side cap room -- and trades, whenever the book eventually
    comes back through a limit nobody meant to leave standing.
    """
    ep = flat_episode
    order = _order(price=0.55, liquidity=Liquidity.TAKER,
                   live_from=4, expires_at=5)
    assert len(resolve([order], ep, 4, {})) == 1
    assert resolve([order], ep, 5, {}) == []
    assert order.is_dead(5) and not order.is_dead(4)


def test_no_fill_against_a_missing_book(flat_episode):
    ep = flat_episode
    ep.ask[5] = np.nan
    assert resolve([_order(price=0.49)], ep, 5, {}) == []
