"""What orders should exist right now.

ONE policy, not a menu. We rest on both sides AND lift the book whenever it is
through our threshold: making and taking are simultaneous, never alternatives,
and there is no `mode` to choose between them.

Fees are part of the DECISION, not an adjustment applied to the PnL after the
fact. The schedule is dollars per share and a price here is a probability, so
the two are directly comparable and a fee can simply be added to a price:

    fee_per_share(p) = 0.07 * p * (1 - p)
    a taker PAYS   +fee * (1 - 0.0833)    -- positive, up to 1.60 c/share
    a maker is PAID -fee * 0.20           -- NEGATIVE, up to 0.35 c/share

This is the defect the unified model exists to fix. The old taker arm crossed
on a 1 c edge against a ~1.6 c fee, which loses by construction no matter how
good the forecast is.

Resting, a FIXED POINT, because the fee depends on the price we are solving
for:

    resting_bid = eff_bid - maker_fee(resting_bid)   # maker_fee < 0: HIGHER
    resting_ask = eff_ask + maker_fee(resting_ask)   # LOWER

The rebate buys tightness. The all-in cost of a maker buy at P is P + fee(P),
and it is that, not P, which has to clear eff_bid -- so we may post ABOVE our
fair bid by the rebate and still pay no more than fair. The ask is the mirror
image, which is where its opposite sign comes from.

Solved by iterating p <- eff_bid - maker_fee(p) from p = eff_bid, TWICE. The
derivative of that map is 0.014 * (1 - 2p), so |g'| <= 0.014 everywhere on
(0, 1) and every pass shrinks the error by at least that factor. The starting
error is at most the rebate itself, 0.0035, so one pass already lands inside a
tick and two leave a residual under 2e-7 -- four orders of magnitude inside
the 0.01 grid, and far inside the rounding that follows. It is not closed
form; iterating it to machine precision would be theatre.

Crossing needs NO fixed point, and that asymmetry with the maker side is
deliberate rather than an oversight:

    buy  when  book_ask <= eff_bid - taker_fee(book_ask)
    sell when  book_bid >= eff_ask + taker_fee(book_bid)

The price we would pay is the price already on the screen, so the fee is known
exactly -- evaluate it AT the book and there is nothing to solve. Note that
neither threshold is snapped: `book_bid`/`book_ask` come off the venue and are
therefore already on the cent grid, and `eff_bid`/`eff_ask` are compared to
them as valuations, not sent as prices. Rounding a valuation before comparing
it would only discard the sub-tick part of our own edge.

Snapping happens ONCE, here, and only on prices we actually send. `quote.py`
deliberately returns theoretical prices off the grid: the maker rebate is
sub-tick, so snapping there and again here would round an on-grid number twice
in the same direction and the fee adjustment could never move an order price.
Adjust first, snap last -- the conservative direction is unchanged.

Gates, all of which bind:
  * a book older than max_book_age_ms is not a quote, so we neither post
    against it nor leave orders resting on it
  * the position cap binds on the TAKER path as well as the maker path
    (the 2026-05-20 taker-cap-bypass lesson), it binds on IN-FLIGHT size and
    it binds PER SIDE. Signed net inventory is the wrong test: a resting sell
    would offset a long, re-open the buy side and breach the cap on the maker
    path. And a taker cross cannot be cancelled while the venue holds it, so a
    cap tested against `q` alone is re-breached on every requote until the
    first fill lands.
  * one live unfilled order per side gates the cross -- an intention already
    expressed is not a reason to express it again. Under the unified policy
    this costs at most one requote cycle: the crossing condition and the
    "never cross the book" maker guard are mutually exclusive on a side, so a
    resting order sitting on the side we want to lift is always already in
    to_cancel.
  * a maker order that would cross the book is not a maker order, and that is
    tested on the price we would actually send, AFTER the fee adjustment and
    the grid snap
"""
import math

from harness.blocks.defaults.fees import FeeSchedule, Liquidity
from harness.core.types import OrderRequest, Side

#: Passes of the resting-price fixed point. See the module docstring: the map
#: contracts by <= 0.014 per pass, so two are already 1e4 times finer than the
#: tick they get rounded onto.
_FIXED_POINT_PASSES = 2


def snap(p, side, tick):
    """A price onto the venue's cent grid, conservatively: bids DOWN, asks UP.

    The ONLY snap in the harness. `quote.py` hands us theoretical prices off
    the grid on purpose: snapping there and again here would round an on-grid
    number in the same direction twice, which is a no-op, and the whole maker
    rebate is sub-tick -- so the adjustment could never survive it. Snap once,
    on the number actually sent to the venue, and it does.
    """
    if side == Side.BUY:
        p = round(math.floor(p / tick + 1e-9) * tick, 10)
    else:
        p = round(math.ceil(p / tick - 1e-9) * tick, 10)
    return min(1.0, max(0.0, p))


def maker_fee(fees, p):
    """Signed USD per share for a maker fill at `p`. Negative: we are PAID."""
    return fees.charge(Liquidity.MAKER, 1.0, p)


def taker_fee(fees, p):
    """Positive USD per share for a taker fill at `p`."""
    return fees.charge(Liquidity.TAKER, 1.0, p)


def resting_price(fees, target, side):
    """The fee-adjusted price to rest at, off the grid.

    Solves P = target - maker_fee(P) for a bid and P = target + maker_fee(P)
    for an ask. `maker_fee` is negative, so a bid lands ABOVE `target` and an
    ask BELOW it: the rebate tightens the pair.
    """
    sign = 1.0 if side == Side.BUY else -1.0
    p = target
    for _ in range(_FIXED_POINT_PASSES):
        p = target - sign * maker_fee(fees, p)
    return p


def resting_quote(fees, target, side, tick):
    """The order price: `resting_price`, then ONE conservative snap.

    `target` is the unsnapped theoretical quote, so the sub-tick rebate can
    carry the price across a tick boundary before the snap sees it, and a bid
    within the rebate of the next cent posts a full tick higher than the raw
    quote would. Where it cannot reach the boundary the snap takes it back and
    nothing changes -- which is the correct, conservative outcome, not the
    universal one it used to be.
    """
    return snap(resting_price(fees, target, side), side, tick)


def decide(i, eff_bid, eff_ask, q, ep, live_orders, execn, params):
    """Return (to_place, to_cancel) at decision index i."""
    to_place, to_cancel = [], []

    # `ExecConfig.fees` is a run parameter; None means "the default schedule".
    # Policy needs the schedule at all because the thresholds ARE fee-adjusted.
    fees = execn.fees if execn.fees is not None else FeeSchedule()

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

    # In-flight size counts, PER SIDE. Orders already live may still fill, so
    # the room for one more lot is measured against the worst case on that
    # side: every live order on it filling as well. Netting the two sides
    # would let a resting sell offset a long and re-open the buy side.
    pending_buy = sum(o.shares for o in live_orders if o.side == Side.BUY)
    pending_sell = sum(o.shares for o in live_orders if o.side == Side.SELL)
    room_buy = can_buy and q + pending_buy + params.shares <= cap
    room_sell = can_sell and q - pending_sell - params.shares >= -cap

    book_bid, book_ask = ep.bid[i], ep.ask[i]

    # --- rest on both sides, inside our own fee-adjusted valuation ---------
    want = {}
    if can_buy and math.isfinite(book_ask):
        price = resting_quote(fees, eff_bid, Side.BUY, params.tick)
        if price < book_ask:            # a maker order never crosses the book
            want[Side.BUY] = price
    if can_sell and math.isfinite(book_bid):
        price = resting_quote(fees, eff_ask, Side.SELL, params.tick)
        if price > book_bid:
            want[Side.SELL] = price

    # Only maker orders are requoted. A marketable order cannot be cancelled
    # while the venue holds it, so it is not a stale quote to be replaced.
    resting_orders = [o for o in live_orders if o.liquidity == Liquidity.MAKER]
    for o in resting_orders:
        if want.get(o.side) != o.price:
            to_cancel.append(o.order_id)

    resting = {o.side: o.price for o in resting_orders
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

    # --- and cross, at the same instant, when the book is through the fee ---
    to_place.extend(_crosses(fees, eff_bid, eff_ask, book_bid, book_ask,
                             live_orders, params, room_buy, room_sell))

    return to_place, to_cancel


def _crosses(fees, eff_bid, eff_ask, book_bid, book_ask, live_orders, params,
             room_buy, room_sell):
    """Marketable orders, one live unfilled order per side at most.

    The taker path has no cancel: the venue holds a marketable order for its
    lock window and will not release it. So a cross that is still in flight
    must suppress the next one on that side, or the same intention is sent
    once per requote until the first fill finally lands.

    The fee is evaluated at the BOOK price rather than at a fixed point,
    because the book price is exactly what we would pay. Compare the maker
    side, where the price is the unknown and so the fee is too; the asymmetry
    is the point, not an omission.

    The THRESHOLD is unsnapped on both sides of the comparison: the book is
    already on the grid because the venue put it there, and eff_bid/eff_ask
    enter as valuations rather than as prices. Only the protective limit we
    send is snapped, and snapping it conservatively cannot cost the fill: the
    book price is on the grid and no worse than the valuation, so the nearest
    grid point on the conservative side of the valuation is still at or
    through the book.
    """
    out = []
    live_sides = {o.side for o in live_orders}

    if (room_buy and Side.BUY not in live_sides and math.isfinite(book_ask)
            and book_ask <= eff_bid - taker_fee(fees, book_ask)):
        out.append(OrderRequest(Side.BUY, snap(eff_bid, Side.BUY, params.tick),
                                params.shares, Liquidity.TAKER, "cross_bid"))
    if (room_sell and Side.SELL not in live_sides and math.isfinite(book_bid)
            and book_bid >= eff_ask + taker_fee(fees, book_bid)):
        out.append(OrderRequest(Side.SELL, snap(eff_ask, Side.SELL,
                                                params.tick),
                                params.shares, Liquidity.TAKER, "cross_ask"))
    return out
