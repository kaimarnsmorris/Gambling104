"""fair block: the settlement level the model is indifferent at, in USD.

This is the harness's `E[A]`. It is strike-free by construction - y* is affine in
the strike - so the strike enters in `f.py` and nowhere else.
"""
from _fvexport import for_episode


def precompute(ep):
    return for_episode(ep)["s"]
