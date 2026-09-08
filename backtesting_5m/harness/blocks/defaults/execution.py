"""What orders should exist right now.

Owns policy only -- post, cross, cancel, and the gates. It owns no timing:
the engine converts these decisions into live_from / cancel_at indices using
the latency model, which is why latency can be swept without touching policy.

Gates, all of which bind:
  * a book older than max_book_age_ms is not a quote, so we neither post
    against it nor leave orders resting on it
  * the position cap binds on the TAKER path as well as the maker path
    (the 2026-05-20 taker-cap-bypass lesson), and it binds on IN-FLIGHT size,
    not only on realised inventory: a taker cross cannot be cancelled while
    the venue holds it, so a cap tested against `q` alone is re-breached on
    every requote until the first fill lands
  * one live unfilled cross per side -- a taker order still in flight is an
    intention already expressed, not a reason to express it again
  * a maker order that would cross the book is not a maker order
"""
import math

from harness.blocks.defaults.fees import Liquidity
from harness.core.types import OrderRequest, Side


def decide(i, eff_bid, eff_ask, q, ep, live_orders, execn, params):
    """Return (to_place, to_cancel) at decision index i."""
    to_place, to_cancel = [], []

    tte = ep.tte_s(i)
    tradable = (
        bool(ep.has_book[i])
        and ep.book_age_ms[i] <= execn.max_book_age_ms
        and execn.min_tte_s <= tte <= execn.max_tte_s
    )

    if not tradable:
        return [], [o.order_id for o in live_orders]

    cap = params.max_pos if params.max_pos > 0.0 else math.inf
    can_buy = q < cap
    can_sell = q > -cap

    # In-flight size counts. Orders already live may still fill, so the room
    # for one more lot is measured against the worst case on that side: every
    # live order on it filling as well.
    pending_buy = sum(o.shares for o in live_orders if o.side == Side.BUY)
    pending_sell = sum(o.shares for o in live_orders if o.side == Side.SELL)
    room_buy = can_buy and q + pending_buy + params.shares <= cap
    room_sell = can_sell and q - pending_sell - params.shares >= -cap

    book_bid, book_ask = ep.bid[i], ep.ask[i]

    if execn.mode == "taker":
        return _crosses(eff_bid, eff_ask, book_bid, book_ask, live_orders,
                        params, room_buy, room_sell), to_cancel

    # maker (and the maker half of "both"): rest inside our own valuation,
    # never through the book
    want = {}
    if can_buy and math.isfinite(book_ask) and eff_bid < book_ask:
        want[Side.BUY] = eff_bid
    if can_sell and math.isfinite(book_bid) and eff_ask > book_bid:
        want[Side.SELL] = eff_ask

    for o in live_orders:
        if want.get(o.side) != o.price:
            to_cancel.append(o.order_id)

    resting = {o.side: o.price for o in live_orders
               if o.order_id not in to_cancel}
    room = {Side.BUY: room_buy, Side.SELL: room_sell}
    for side, price in want.items():
        if resting.get(side) == price:
            continue
        if not room[side]:
            # a replacement quote would be additional in-flight size until the
            # cancel lands; wait for it rather than double the exposure
            continue
        to_place.append(OrderRequest(side, price, params.shares,
                                     Liquidity.MAKER, "quote"))

    if execn.mode == "both":
        to_place.extend(_crosses(eff_bid, eff_ask, book_bid, book_ask,
                                 live_orders, params, room_buy, room_sell))

    return to_place, to_cancel


def _crosses(eff_bid, eff_ask, book_bid, book_ask, live_orders, params,
             room_buy, room_sell):
    """Marketable orders, one live unfilled cross per side at most.

    The taker path has no cancel: the venue holds a marketable order for its
    lock window and will not release it. So a cross that is still in flight
    must suppress the next one on that side, or the same intention is sent
    once per requote until the first fill finally lands.
    """
    out = []
    live_sides = {o.side for o in live_orders}

    if (room_buy and Side.BUY not in live_sides
            and math.isfinite(book_ask) and eff_bid >= book_ask):
        out.append(OrderRequest(Side.BUY, eff_bid, params.shares,
                                Liquidity.TAKER, "cross_bid"))
    if (room_sell and Side.SELL not in live_sides
            and math.isfinite(book_bid) and eff_ask <= book_bid):
        out.append(OrderRequest(Side.SELL, eff_ask, params.shares,
                                Liquidity.TAKER, "cross_ask"))
    return out
