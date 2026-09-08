"""The vocabulary of an order's life.

`execution` decides WHAT orders should exist; the engine decides WHEN they
become live, by turning latency into `live_from`, `cancel_at` and `expires_at`
indices; `fill` decides WHETHER a live order trades. Keeping those three apart
is what lets fill optimism be swept with policy held fixed.

A marketable order's life is BOUNDED, and both halves of that live here so no
caller has to special-case it:

  * it cannot be recalled while the venue still holds it -- crypto up/down
    takers sit in a ~250 ms lock (`latency.taker_lock_ms`, a floor on
    `live_from`) that no cancel reaches, so `is_cancellable` is False until
    `live_from`. A blanket "pull everything" that ignores this retracts
    exactly the crosses a moving book would have picked off, which flatters
    the result;
  * and it does not survive the tick it arrives on. The lock is 250 ms, so by
    `live_from` the book has had 300 ms to move away; a cross that finds it
    gone is over, not resting. Without `expires_at` such an order stays live
    for the rest of the episode and permanently consumes per-side cap room --
    which suppresses the resting quote on that side too, since the cap is
    tested against in-flight size.
"""
from dataclasses import dataclass
from enum import IntEnum

from harness.blocks.defaults.fees import Liquidity  # re-exported


class Side(IntEnum):
    BUY = 1
    SELL = -1


@dataclass(frozen=True)
class OrderRequest:
    """What the execution block asks for. Carries no timing."""
    side: Side
    price: float
    shares: float
    liquidity: Liquidity
    reason: str = ""


@dataclass(frozen=True)
class Order:
    """A request the engine has stamped with timing."""
    order_id: int
    side: Side
    price: float
    shares: float
    liquidity: Liquidity
    live_from: int
    cancel_at: int | None
    reason: str = ""
    latency_ms: float = 0.0      # the placement delay actually drawn
    #: first index at which the order is gone of its own accord, whether or
    #: not it traded. `None` is "until cancelled" -- a resting order. The
    #: engine stamps `live_from + 1` on every marketable order.
    expires_at: int | None = None

    def is_live(self, i: int) -> bool:
        if i < self.live_from:
            return False
        if self.expires_at is not None and i >= self.expires_at:
            return False
        return self.cancel_at is None or i < self.cancel_at

    def is_dead(self, i: int) -> bool:
        """Past every index at which this order could still trade.

        `is_live` answers "not now"; this answers "not ever again", which is
        what lets the engine drop the order and release the cap room it held.
        """
        if self.expires_at is not None and i >= self.expires_at:
            return True
        return self.cancel_at is not None and i >= self.cancel_at

    def is_cancellable(self, i: int) -> bool:
        """Can a cancel issued at decision index `i` still reach this order?

        Not while the venue holds a marketable order inside its lock window.
        See the module docstring: this is the ONE predicate both cancel paths
        -- the not-tradable branch in `execution.decide` and the model-outage
        pull in the engine -- go through, so they cannot diverge again.
        """
        return not (self.liquidity == Liquidity.TAKER and i < self.live_from)


def cancellable_ids(orders, i):
    """The ids among `orders` that a cancel issued at index `i` can reach.

    Both cancel-everything paths call this rather than sweeping up every live
    order: a marketable order still inside the venue's lock is beyond recall,
    and pretending otherwise is what let the simulation dodge its own crosses.
    """
    return [o.order_id for o in orders if o.is_cancellable(i)]


@dataclass(frozen=True)
class Fill:
    order_id: int
    idx: int
    side: Side
    liquidity: Liquidity
    price: float
    shares: float
    reason: str = ""
