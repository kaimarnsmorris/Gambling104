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


def precompute(ep):
    return columns_for(ep.market_id)[0]
