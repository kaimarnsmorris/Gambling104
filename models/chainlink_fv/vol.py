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


def precompute(ep):
    return columns_for(ep.market_id)[1]
