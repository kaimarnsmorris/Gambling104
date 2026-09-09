"""sigma -- the scale that carries the tail.

Not the settlement standard deviation. It is that, times the fitted tail's own
scale, times the quartile ratio that reconciles the fitted shape with the fixed
one `link.py` applies. The whole per-tick tail is folded in here, because this
and `fair.py` are the only blocks the harness lets see an episode.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _chainlink_fv_export import columns_for


def precompute(ep, variant=None, scale=1.0):
    """`variant` travels as a signal param, never as ambient state.

    The harness caches this array on the CONTENT of the block files plus
    `signal_params`. Every variant of this model has byte-identical block files --
    only the export behind them differs -- so a variant selected from the
    environment puts nothing in that key and the second variant in a process is
    handed the first one's arrays. Passing it here is what makes the cache correct.
    Falling back to `FV_VARIANT` keeps a plain single-variant run a one-liner.

    `scale` is the manual vol adjustment, and it is a signal param so the harness
    can sweep it without rebuilding a single export. Scaling the settlement scale
    up shrinks |z| and pulls `p` toward a half, which is the lever on the
    overconfidence the tails show: the model says 0.361 where the book says 0.481
    and reality was 0.458. Note that it moves magnitude, not sign, so it changes
    which trades clear a threshold rather than which way they point.
    """
    return columns_for(ep.market_id, variant)[1] * float(scale)
