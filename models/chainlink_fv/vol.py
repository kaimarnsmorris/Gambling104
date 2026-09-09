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


def precompute(ep, variant=None):
    """`variant` travels as a signal param, never as ambient state.

    The harness caches this array on the CONTENT of the block files plus
    `signal_params`. Every variant of this model has byte-identical block files --
    only the export behind them differs -- so a variant selected from the
    environment puts nothing in that key and the second variant in a process is
    handed the first one's arrays. Passing it here is what makes the cache correct.
    Falling back to `FV_VARIANT` keeps a plain single-variant run a one-liner.
    """
    return columns_for(ep.market_id, variant)[1]
