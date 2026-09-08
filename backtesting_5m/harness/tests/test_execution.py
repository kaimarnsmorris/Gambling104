"""Pins the execution policy.

Policy owns what orders should exist -- post, cross, cancel, and the gates
that stop us trading on information we should not trust. It owns no timing;
the engine turns decisions into live_from/cancel_at indices.
"""
import numpy as np

from harness.blocks.defaults.execution import decide
from harness.blocks.defaults.fees import Liquidity
from harness.core.config import ExecConfig, QuoteParams
from harness.core.types import Order, Side

PARAMS = QuoteParams(shares=10.0, max_pos=50.0)


def _live(price, side=Side.BUY, oid=1):
    return Order(order_id=oid, side=side, price=price, shares=10.0,
                 liquidity=Liquidity.MAKER, live_from=0, cancel_at=None)


def test_maker_mode_posts_both_sides(flat_episode):
    place, cancel = decide(100, 0.48, 0.52, 0.0, flat_episode, [],
                           ExecConfig(mode="maker"), PARAMS)
    assert {o.side for o in place} == {Side.BUY, Side.SELL}
    assert all(o.liquidity == Liquidity.MAKER for o in place)


def test_maker_mode_never_crosses_the_book(flat_episode):
    """Our bid must stay below the ask or it is not a maker order."""
    place, _ = decide(100, 0.60, 0.70, 0.0, flat_episode, [],
                      ExecConfig(mode="maker"), PARAMS)
    buys = [o for o in place if o.side == Side.BUY]
    assert all(o.price < flat_episode.ask[100] for o in buys)


def test_taker_mode_lifts_only_when_the_ask_is_below_our_bid(flat_episode):
    """The crossing rule: eff_bid >= market ask means the book is cheap."""
    place, _ = decide(100, 0.52, 0.60, 0.0, flat_episode, [],
                      ExecConfig(mode="taker"), PARAMS)
    assert len(place) == 1
    assert place[0].side == Side.BUY
    assert place[0].liquidity == Liquidity.TAKER


def test_taker_mode_stands_still_when_the_book_is_fair(flat_episode):
    place, _ = decide(100, 0.48, 0.52, 0.0, flat_episode, [],
                      ExecConfig(mode="taker"), PARAMS)
    assert place == []


def test_a_stale_book_suppresses_all_trading(flat_episode):
    ep = flat_episode
    ep.book_age_ms[100] = 5000.0
    place, cancel = decide(100, 0.48, 0.52, 0.0, ep, [_live(0.48)],
                           ExecConfig(mode="maker", max_book_age_ms=1000.0),
                           PARAMS)
    assert place == []
    assert cancel == [1], "stale book must also pull resting orders"


def test_a_missing_book_suppresses_all_trading(flat_episode):
    ep = flat_episode
    ep.has_book[100] = False
    place, _ = decide(100, 0.48, 0.52, 0.0, ep, [],
                      ExecConfig(mode="maker"), PARAMS)
    assert place == []


def test_the_position_cap_binds_on_the_maker_path(flat_episode):
    place, _ = decide(100, 0.48, 0.52, 50.0, flat_episode, [],
                      ExecConfig(mode="maker"), PARAMS)
    assert all(o.side == Side.SELL for o in place), "long at the cap: sell only"


def test_the_position_cap_binds_on_the_taker_path_too(flat_episode):
    """The 2026-05-20 taker-cap-bypass lesson, as a test."""
    place, _ = decide(100, 0.52, 0.60, 50.0, flat_episode, [],
                      ExecConfig(mode="taker"), PARAMS)
    assert place == []


def test_a_live_cross_suppresses_the_next_one_on_that_side(flat_episode):
    """A taker order in flight is an intention already expressed."""
    live = [Order(order_id=1, side=Side.BUY, price=0.52, shares=10.0,
                  liquidity=Liquidity.TAKER, live_from=2, cancel_at=None)]
    place, cancel = decide(100, 0.52, 0.60, 0.0, flat_episode, live,
                           ExecConfig(mode="taker"), PARAMS)
    assert place == [], "the same cross was re-emitted while one was in flight"
    assert cancel == [], "a taker order cannot be cancelled anyway"


def test_in_flight_shares_count_towards_the_cap(flat_episode):
    """max_pos of one lot with one lot already in flight on that side.

    The stale sell at 0.60 is cancelled, but the cancel has not landed, so it
    can still fill. Replacing it now would put two lots of short exposure in
    flight against a cap of one. The replacement waits; the free side does not.
    """
    params = QuoteParams(shares=1.0, max_pos=1.0)
    live = [Order(order_id=1, side=Side.SELL, price=0.60, shares=1.0,
                  liquidity=Liquidity.MAKER, live_from=0, cancel_at=None)]
    place, cancel = decide(100, 0.45, 0.52, 0.0, flat_episode, live,
                           ExecConfig(mode="maker"), params)
    assert cancel == [1], "the stale sell is still cancelled"
    assert [o.side for o in place] == [Side.BUY], "the short side is full"


def test_a_resting_order_at_the_right_price_is_left_alone(flat_episode):
    place, cancel = decide(100, 0.48, 0.52, 0.0, flat_episode,
                           [_live(0.48, Side.BUY, 1), _live(0.52, Side.SELL, 2)],
                           ExecConfig(mode="maker"), PARAMS)
    assert place == [] and cancel == []


def test_a_resting_order_at_a_stale_price_is_replaced(flat_episode):
    place, cancel = decide(100, 0.45, 0.52, 0.0, flat_episode,
                           [_live(0.48, Side.BUY, 1), _live(0.52, Side.SELL, 2)],
                           ExecConfig(mode="maker"), PARAMS)
    assert cancel == [1]
    assert [o.side for o in place] == [Side.BUY]


def test_trading_stops_outside_the_tte_window(flat_episode):
    place, _ = decide(2990, 0.48, 0.52, 0.0, flat_episode, [],
                      ExecConfig(mode="maker", min_tte_s=15.0), PARAMS)
    assert place == []
