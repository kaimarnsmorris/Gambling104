"""How long our intentions take to reach the exchange.

Defaults are operator-set: 100 ms to place or cancel, 200 ms to take. The
venue additionally holds marketable crypto up/down orders ~250 ms and will not
let them be cancelled, so `taker_lock_ms` is a FLOOR on the taker path that no
parameter can undercut. It also reconciles the measured 276 ms taker fill as
26 ms POST flight plus the 250 ms lock.

Seeding is per-episode so that a run is byte-identical regardless of worker
count or completion order.
"""
import hashlib
import math
from dataclasses import dataclass

import numpy as np

from harness.paths import BUCKET_MS


def episode_rng(market_id: str, run_seed: int) -> np.random.Generator:
    """A generator determined only by the market and the run seed."""
    digest = hashlib.blake2b(
        f"{market_id}:{run_seed}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


@dataclass(frozen=True)
class LatencyModel:
    place_ms: float = 100.0
    cancel_ms: float = 100.0
    take_ms: float = 200.0
    taker_lock_ms: float = 250.0
    jitter_frac: float = 0.0
    move_cancel_ms: float | None = None

    def draw(self, rng, kind: str, in_move: bool = False) -> float:
        if kind == "place":
            base = self.place_ms
        elif kind == "cancel":
            base = (self.move_cancel_ms
                    if in_move and self.move_cancel_ms is not None
                    else self.cancel_ms)
        elif kind == "take":
            base = self.take_ms
        else:
            raise ValueError(f"unknown latency kind {kind!r}")

        if self.jitter_frac > 0.0:
            # lognormal keeps it positive and right-skewed, like a real wire
            base = float(base * rng.lognormal(0.0, self.jitter_frac))

        if kind == "take":
            base = max(base, self.taker_lock_ms)
        return float(base)

    @staticmethod
    def delay_idx(ms: float) -> int:
        """Whole 100 ms buckets a delay of `ms` costs, rounded up."""
        return int(math.ceil(ms / BUCKET_MS))
