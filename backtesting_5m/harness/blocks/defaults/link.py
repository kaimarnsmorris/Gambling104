"""z -> p. Default link.

The shipped chain uses a NIG survival function; this logistic default is a
stand-in with the right shape and bounds, so the harness runs before the NIG
parameters are wired in. Override by dropping a `link.py` into an
investigation folder.
"""
import math


def link(z: float) -> float:
    """P(up) given the standardised distance from the strike."""
    if z > 40.0:
        return 1.0
    if z < -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))
