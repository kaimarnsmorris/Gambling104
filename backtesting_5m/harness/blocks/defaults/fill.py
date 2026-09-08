"""Does a live order trade, and at what price.

DEFAULT: touch-fill. A resting order fills at its own price the moment the
book's L1 reaches it. Adverse selection is NOT modelled by degrading the fill
price -- a limit order fills at its limit. It is modelled by the cancel losing
the race: `cancel_at` is an index, and any cross before it fills us. Since
cancel latency rises inside a move (28 ms quiet against 73-165 ms in a move),
we are filled hardest exactly when we are most wrong.

`penetration` gives the conservative arm: require the book to trade THROUGH
the price by a margin, a cheap proxy for queue position we cannot observe.
Set it to 0 (the default) for the touch arm.

None of this is measurable from the panel. Report a range, never one number.
"""
import math

from harness.blocks.defaults.fees import Liquidity
from harness.core.types import Fill, Side


def resolve(orders, ep, i, params):
    """Fills generated at decision index `i`."""
    margin = float(params.get("penetration", 0.0))
    bid, ask = ep.bid[i], ep.ask[i]
    fills = []

    for o in orders:
        if not o.is_live(i):
            continue

        if o.liquidity == Liquidity.TAKER:
            # marketable: pay the book, but never through our own limit
            if o.side == Side.BUY:
                if math.isfinite(ask) and ask <= o.price:
                    fills.append(_fill(o, i, ask))
            else:
                if math.isfinite(bid) and bid >= o.price:
                    fills.append(_fill(o, i, bid))
            continue

        # resting: the book must come to us
        if o.side == Side.BUY:
            if math.isfinite(ask) and ask <= o.price - margin:
                fills.append(_fill(o, i, o.price))
        else:
            if math.isfinite(bid) and bid >= o.price + margin:
                fills.append(_fill(o, i, o.price))

    return fills


def _fill(o, i, price):
    return Fill(order_id=o.order_id, idx=i, side=o.side,
                liquidity=o.liquidity, price=float(price),
                shares=o.shares, reason=o.reason)
