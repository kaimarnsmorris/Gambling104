"""A deliberately crude placeholder: s = strike, so p = 0.5 everywhere.

Exists to exercise the plumbing before a real fair export is wired in. It has
no forecasting content and must never appear in a reported result.
"""
import numpy as np


def precompute(ep):
    return np.full(len(ep), ep.strike, dtype="float64")
