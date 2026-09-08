"""Pins the execution policy.

Policy owns what orders should exist -- post, cross, cancel, and the gates
that stop us trading on information we should not trust. It owns no timing;
the engine turns decisions into live_from/cancel_at indices.

There is one policy, not a menu: it rests on both sides and crosses the book
in the same pass, with the fee schedule inside the thresholds rather than
applied to the PnL afterwards.
"""
import pytest

from harness.blocks.defaults.execution import (decide, maker_fee,
                                               resting_price, resting_quote,
                                               taker_fee)
from harness.blocks.defaults.fees import FeeSchedule, Liquidity
from harness.core.config import ExecConfig, QuoteParams
from harness.core.types import Order, Side

PARAMS = QuoteParams(shares=10.0, max_pos=50.0)
FEES = FeeSchedule()
TICK = 0.01


def _live(price, side=Side.BUY, oid=1):
    return Order(order_id=oid, side=side, price=price, shares=10.0,
                 liquidity=Liquidity.MAKER, live_from=0, cancel_at=None)


def _makers(place):
    return [o for o in place if o.liquidity == Liquidity.MAKER]


def _takers(place):
    return [o for o in place if o.liquidity == Liquidity.TAKER]


def test_the_policy_posts_both_sides(flat_episode):
    place, cancel = decide(100, 0.48, 0.52, 0.0, flat_episode, [],
                           ExecConfig(), PARAMS)
    assert {o.side for o in _makers(place)} == {Side.BUY, Side.SELL}


def test_a_resting_order_never_crosses_the_book(flat_episode):
    """Our bid must stay below the ask or it is not a maker order."""
    place, _ = decide(100, 0.60, 0.70, 0.0, flat_episode, [],
                      ExecConfig(), PARAMS)
    buys = [o for o in _makers(place) if o.side == Side.BUY]
    assert all(o.price < flat_episode.ask[100] for o in buys)


# -- the maker rebate tightens the quote -------------------------------------

def test_the_rebate_moves_the_resting_pair_inwards():
    """A maker fee is NEGATIVE, so a bid posts higher and an ask posts lower.

    The all-in cost of a maker buy at P is P + fee(P), and it is that which
    has to clear eff_bid -- so we may post above fair by the rebate.
    """
    eff_bid, eff_ask = 0.48, 0.52
    bid = resting_price(FEES, eff_bid, Side.BUY)
    ask = resting_price(FEES, eff_ask, Side.SELL)

    rebate_bid = -maker_fee(FEES, bid)
    rebate_ask = -maker_fee(FEES, ask)
    assert rebate_bid > 0.0 and rebate_ask > 0.0

    assert bid > eff_bid
    assert ask < eff_ask
    assert bid - eff_bid == pytest.approx(rebate_bid, abs=1e-6)
    assert eff_ask - ask == pytest.approx(rebate_ask, abs=1e-6)


def test_the_resting_price_is_a_self_consistent_fixed_point():
    """P + maker_fee(P) is the all-in cost, and it must land back on eff_bid.

    Two passes of the iteration, because the map contracts by 0.014 a pass:
    the residual here is ~1e-7 against a 0.01 tick.
    """
    for eff_bid in (0.05, 0.2, 0.48, 0.5, 0.9):
        bid = resting_price(FEES, eff_bid, Side.BUY)
        assert bid + maker_fee(FEES, bid) == pytest.approx(eff_bid, abs=TICK)
        assert bid + maker_fee(FEES, bid) == pytest.approx(eff_bid, abs=1e-6)

    for eff_ask in (0.05, 0.2, 0.52, 0.5, 0.9):
        ask = resting_price(FEES, eff_ask, Side.SELL)
        assert ask - maker_fee(FEES, ask) == pytest.approx(eff_ask, abs=TICK)
        assert ask - maker_fee(FEES, ask) == pytest.approx(eff_ask, abs=1e-6)


def test_the_cent_grid_absorbs_the_whole_maker_rebate():
    """0.35 c/share at its widest, snapped conservatively onto a 1 c grid.

    So an on-grid eff_bid -- which is all `quote.py` ever produces -- posts at
    exactly eff_bid. The adjustment is real and correctly signed; this venue's
    tick is simply too coarse to express it, which is why fee-awareness bites
    on the taker threshold and not here.
    """
    assert -maker_fee(FEES, 0.5) < TICK
    assert resting_quote(FEES, 0.48, Side.BUY, TICK) == pytest.approx(0.48)
    assert resting_quote(FEES, 0.52, Side.SELL, TICK) == pytest.approx(0.52)


# -- the taker threshold is fee-adjusted -------------------------------------

def test_no_cross_when_the_edge_is_inside_the_taker_fee(flat_episode):
    """The defect the unified model exists to fix.

    eff_bid 0.52 against an ask of 0.51 is a 1 c edge and the old policy
    crossed it -- into a ~1.6 c fee, a loss by construction whatever the
    forecast said.
    """
    ep = flat_episode
    ep.ask[100] = 0.51
    fee = taker_fee(FEES, 0.51)
    assert ep.ask[100] > 0.52 - fee, "fixture must sit inside the fee band"

    place, _ = decide(100, 0.52, 0.60, 0.0, ep, [], ExecConfig(), PARAMS)
    assert _takers(place) == [], "crossed on an edge smaller than the fee"


def test_a_cross_fires_once_the_book_clears_the_taker_fee(flat_episode):
    ep = flat_episode
    ep.ask[100] = 0.50
    assert ep.ask[100] <= 0.52 - taker_fee(FEES, 0.50)

    place, _ = decide(100, 0.52, 0.60, 0.0, ep, [], ExecConfig(), PARAMS)
    takers = _takers(place)
    assert [o.side for o in takers] == [Side.BUY]


def test_a_sell_cross_fires_only_above_the_fee_adjusted_ask(flat_episode):
    ep = flat_episode
    ep.bid[100] = 0.53                     # eff_ask 0.52 + fee 0.0163 = 0.5363
    place, _ = decide(100, 0.40, 0.52, 0.0, ep, [], ExecConfig(), PARAMS)
    assert _takers(place) == []

    ep.bid[100] = 0.56
    place, _ = decide(100, 0.40, 0.52, 0.0, ep, [], ExecConfig(), PARAMS)
    assert [o.side for o in _takers(place)] == [Side.SELL]


def test_the_policy_stands_still_when_the_book_is_fair(flat_episode):
    place, _ = decide(100, 0.48, 0.52, 0.0, flat_episode, [],
                      ExecConfig(), PARAMS)
    assert _takers(place) == []


def test_making_and_taking_happen_together(flat_episode):
    """One policy, not two. A cheap ask is lifted while the ask is rested."""
    ep = flat_episode
    ep.bid[100], ep.ask[100] = 0.20, 0.21

    place, _ = decide(100, 0.48, 0.52, 0.0, ep, [], ExecConfig(), PARAMS)
    takers, makers = _takers(place), _makers(place)
    assert [o.side for o in takers] == [Side.BUY], "the cheap ask was not lifted"
    assert [o.side for o in makers] == [Side.SELL], "we stopped quoting"


def test_the_fee_schedule_on_the_config_drives_the_threshold(flat_episode):
    """Zero the schedule and the same book becomes crossable.

    Which is the whole point: the threshold IS the fee, so a run parameter
    that changes the fee has to change the decision.
    """
    ep = flat_episode
    ep.ask[100] = 0.51
    free = ExecConfig(fees=FeeSchedule(base_fee_rate=0.0))

    charged, _ = decide(100, 0.52, 0.60, 0.0, ep, [], ExecConfig(), PARAMS)
    waived, _ = decide(100, 0.52, 0.60, 0.0, ep, [], free, PARAMS)
    assert _takers(charged) == []
    assert [o.side for o in _takers(waived)] == [Side.BUY]


# -- gates -------------------------------------------------------------------

def test_a_stale_book_suppresses_all_trading(flat_episode):
    ep = flat_episode
    ep.book_age_ms[100] = 5000.0
    place, cancel = decide(100, 0.48, 0.52, 0.0, ep, [_live(0.48)],
                           ExecConfig(max_book_age_ms=1000.0), PARAMS)
    assert place == []
    assert cancel == [1], "stale book must also pull resting orders"


def test_a_missing_book_suppresses_all_trading(flat_episode):
    ep = flat_episode
    ep.has_book[100] = False
    place, cancel = decide(100, 0.48, 0.52, 0.0, ep, [_live(0.48)],
                           ExecConfig(), PARAMS)
    assert place == []
    assert cancel == [1], "a missing book must also pull resting orders"


def test_the_position_cap_binds_on_the_maker_path(flat_episode):
    place, _ = decide(100, 0.48, 0.52, 50.0, flat_episode, [],
                      ExecConfig(), PARAMS)
    assert all(o.side == Side.SELL for o in place), "long at the cap: sell only"


def test_the_position_cap_binds_on_the_taker_path_too(flat_episode):
    """The 2026-05-20 taker-cap-bypass lesson, as a test."""
    ep = flat_episode
    ep.ask[100] = 0.20                      # comfortably through the fee
    place, _ = decide(100, 0.52, 0.60, 50.0, ep, [], ExecConfig(), PARAMS)
    assert [o.side for o in place] == [Side.SELL], "bought at the long cap"


def test_the_cap_counts_in_flight_shares_per_side_on_the_taker_path(
        flat_episode):
    """A signed net cap lets a resting SELL pay for another BUY.

    q=0, cap=1 lot, one lot of sell already in flight. Netting would read
    0 - 1 + 1 <= 1 and cross again; per side it reads 0 + 0 + 1 <= 1 for the
    buy but the sell side is full, and neither may double up.
    """
    params = QuoteParams(shares=1.0, max_pos=1.0)
    ep = flat_episode
    ep.ask[100] = 0.20
    live = [Order(order_id=1, side=Side.SELL, price=0.52, shares=1.0,
                  liquidity=Liquidity.MAKER, live_from=0, cancel_at=None),
            Order(order_id=2, side=Side.BUY, price=0.20, shares=1.0,
                  liquidity=Liquidity.TAKER, live_from=0, cancel_at=None)]
    place, _ = decide(100, 0.52, 0.60, 0.0, ep, live, ExecConfig(), params)
    assert place == [], "in-flight size on both sides, yet it added more"


def test_a_live_cross_suppresses_the_next_one_on_that_side(flat_episode):
    """A taker order in flight is an intention already expressed."""
    ep = flat_episode
    ep.ask[100] = 0.20
    live = [Order(order_id=1, side=Side.BUY, price=0.52, shares=10.0,
                  liquidity=Liquidity.TAKER, live_from=2, cancel_at=None)]
    place, cancel = decide(100, 0.52, 0.60, 0.0, ep, live,
                           ExecConfig(), PARAMS)
    assert _takers(place) == [], "the same cross was re-emitted in flight"
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
                           ExecConfig(), params)
    assert cancel == [1], "the stale sell is still cancelled"
    assert [o.side for o in place] == [Side.BUY], "the short side is full"


def test_a_resting_order_at_the_right_price_is_left_alone(flat_episode):
    place, cancel = decide(100, 0.48, 0.52, 0.0, flat_episode,
                           [_live(0.48, Side.BUY, 1), _live(0.52, Side.SELL, 2)],
                           ExecConfig(), PARAMS)
    assert place == [] and cancel == []


def test_a_resting_order_at_a_stale_price_is_replaced(flat_episode):
    place, cancel = decide(100, 0.45, 0.52, 0.0, flat_episode,
                           [_live(0.48, Side.BUY, 1), _live(0.52, Side.SELL, 2)],
                           ExecConfig(), PARAMS)
    assert cancel == [1]
    assert [o.side for o in place] == [Side.BUY]


def test_an_in_flight_cross_is_not_treated_as_a_stale_quote(flat_episode):
    """A marketable order cannot be cancelled, so it is not a quote to requote.

    Sweeping every live order into the requote comparison would emit a cancel
    the venue will refuse for an order it is holding.
    """
    live = [Order(order_id=1, side=Side.BUY, price=0.52, shares=10.0,
                  liquidity=Liquidity.TAKER, live_from=2, cancel_at=None)]
    _, cancel = decide(100, 0.48, 0.52, 0.0, flat_episode, live,
                       ExecConfig(), PARAMS)
    assert cancel == []


def test_trading_stops_outside_the_tte_window(flat_episode):
    place, _ = decide(2990, 0.48, 0.52, 0.0, flat_episode, [],
                      ExecConfig(min_tte_s=15.0), PARAMS)
    assert place == []


def test_trading_stops_before_the_max_tte(flat_episode):
    place, _ = decide(0, 0.48, 0.52, 0.0, flat_episode, [],
                      ExecConfig(max_tte_s=200.0), PARAMS)
    assert place == []
