"""sigma -- the scale that turns (s - strike) into a standardised distance.

Default is a square-root-of-time scaling of a single per-market constant. It
is a stand-in with the right shape; override it with a real vol block.
"""
import numpy as np


SIGMA_AT_300S = 250.0      # USD of BTC, one standard deviation over 300 s


def precompute(ep):
    tte = np.maximum(np.array([ep.tte_s(i) for i in range(len(ep))]), 1e-6)
    return SIGMA_AT_300S * np.sqrt(tte / 300.0)
