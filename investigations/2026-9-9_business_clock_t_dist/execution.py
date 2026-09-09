"""Cross only. No resting orders.

The harness's default policy is unified on purpose -- it rests on both sides and
lifts the book whenever the book is through the fee-adjusted threshold, and it has
no mode to choose between them. That is the right default and this block does not
argue with it. It argues with applying it to THIS model.

WHAT THE MEASUREMENT SAYS. Over 400 markets of the 5 m book:

    passive fills          markout  -3.70 c/share
    cross at |p-mid|>=0.05  net     +0.55 c/share   (already net of the taker fee)
    cross at |p-mid|>=0.08  net     +2.37 c/share
    cross at |p-mid|>=0.12  net     +7.84 c/share

and on every tick of those markets the model's own log-loss beats the book's,
decisively inside the last 30 s (0.022 against 0.051). The forecast is good. The
passive fills are simply not the trades it is good at: a resting order trades when
the book runs through it, which is when we are wrong, while a cross happens when we
choose it, which is when we disagree and are right.

One subtlety worth recording, because it sent the diagnosis down a blind alley for
an hour. This model's `p` is OVERCONFIDENT -- where it says "much lower" it says
0.361 against a book at 0.481 and a realised 0.458 -- so log-loss, which punishes
magnitude, scores it behind the book at the fills. PnL does not care about
magnitude, only sign, and the sign is right. A model can be simultaneously
badly calibrated in the tails and profitable to trade, and those two facts are
measured by different statistics.

HOW. By wrapping the default rather than reimplementing it. Every gate the default
enforces stays: book staleness, the tte window, the requote cadence, the per-side
cap on resting AND in-flight size, the one-live-order-per-side rule, and the fact
that a marketable order inside the venue's lock window cannot be cancelled. All of
that is logic worth keeping exactly and nothing worth copying by hand.

Cancels pass through untouched. An order already resting still has to be pulled
when the book goes stale, and suppressing its cancel because we no longer intend to
rest would strand it on the book.
"""
from harness.blocks.defaults.execution import decide as _unified
from harness.blocks.defaults.fees import Liquidity


def decide(i, eff_bid, eff_ask, q, ep, live_orders, execn, params):
    """The unified policy's crossing half. Same gates, same prices, no resting."""
    to_place, to_cancel = _unified(i, eff_bid, eff_ask, q, ep, live_orders,
                                   execn, params)
    return [o for o in to_place if o.liquidity == Liquidity.TAKER], to_cancel
