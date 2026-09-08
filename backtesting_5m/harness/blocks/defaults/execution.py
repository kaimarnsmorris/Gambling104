"""What orders should exist right now.

Owns policy only -- post, cross, cancel, and the gates. It owns no timing:
the engine converts these decisions into live_from / cancel_at indices using
the latency model, which is why latency can be swept without touching policy.

Gates, all of which bind:
  * a book older than max_book_age_ms is not a quote, so we neither post
    against it nor leave orders resting on it
  * the position cap binds on the TAKER path as well as the maker path
    (the 2026-05-20 taker-cap-bypass lesson)
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

    book_bid, book_ask = ep.bid[i], ep.ask[i]

    if execn.mode == "taker":
        # cross only when the book is on the wrong side of our own valuation
        if can_buy and math.isfinite(book_ask) and eff_bid >= book_ask:
            to_place.append(OrderRequest(Side.BUY, eff_bid, params.shares,
                                         Liquidity.TAKER, "cross_bid"))
        if can_sell and math.isfinite(book_bid) and eff_ask <= book_bid:
            to_place.append(OrderRequest(Side.SELL, eff_ask, params.shares,
                                         Liquidity.TAKER, "cross_ask"))
        return to_place, to_cancel

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
    for side, price in want.items():
        if resting.get(side) != price:
            to_place.append(OrderRequest(side, price, params.shares,
                                         Liquidity.MAKER, "quote"))

    if execn.mode == "both":
        if can_buy and math.isfinite(book_ask) and eff_bid >= book_ask:
            to_place.append(OrderRequest(Side.BUY, eff_bid, params.shares,
                                         Liquidity.TAKER, "cross_bid"))
        if can_sell and math.isfinite(book_bid) and eff_ask <= book_bid:
            to_place.append(OrderRequest(Side.SELL, eff_ask, params.shares,
                                         Liquidity.TAKER, "cross_ask"))

    return to_place, to_cancel
