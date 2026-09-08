"""The vocabulary of an order's life.

`execution` decides WHAT orders should exist; the engine decides WHEN they
become live, by turning latency into `live_from` and `cancel_at` indices;
`fill` decides WHETHER a live order trades. Keeping those three apart is what
lets fill optimism be swept with policy held fixed.
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

    def is_live(self, i: int) -> bool:
        if i < self.live_from:
            return False
        return self.cancel_at is None or i < self.cancel_at


@dataclass(frozen=True)
class Fill:
    order_id: int
    idx: int
    side: Side
    liquidity: Liquidity
    price: float
    shares: float
    reason: str = ""
