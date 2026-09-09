"""Placeholder fair: s = strike, so p = 0.5 everywhere.

This exists ONLY to prove the harness runs end to end against the real panel
before the Gambling102 fair export is wired in. It has no forecasting content.
Any PnL it produces is a property of the quoting and fill machinery, not of a
model, and must never be reported as an edge.
"""
import numpy as np


def precompute(ep):
    return np.full(len(ep), ep.strike, dtype="float64")
