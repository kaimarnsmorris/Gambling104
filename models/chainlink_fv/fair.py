"""s -- the fair settlement level, in USD, shifted so a strike-centred link is right.

The shift is `mu * sigma`: the fitted tail's location offset, which moves the
model's 50/50 level off the strike. `f` measures from the strike and cannot be
told to centre anywhere else, so the offset rides on the fair value instead.
`tailfold.harness_columns` is the one place that decides it.
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
    return columns_for(ep.market_id, variant)[0]
