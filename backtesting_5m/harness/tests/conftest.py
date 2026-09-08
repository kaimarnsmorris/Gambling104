"""Synthetic fixtures. Every number here is known by construction, so a
failure localises to the code rather than to the 20.9 M-row panel."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


@pytest.fixture
def obs_sparse():
    """Observations in buckets 0, 1 and 5 only. Everything else is a gap."""
    return pd.DataFrame({
        "t_ms": [0, 100, 500],
        "bid": [0.40, 0.41, 0.45],
        "ask": [0.42, 0.43, 0.47],
        "mid": [0.41, 0.42, 0.46],
        "n_src": [2, 2, 1],
    })


@pytest.fixture
def flat_episode():
    """A 3000-tick market quoted flat at 0.49/0.51, settling UP.

    Built directly rather than through the loader so engine tests do not
    depend on the loader's correctness.
    """
    from harness.core.episode import Episode
    from harness.paths import N_BUCKET

    n = N_BUCKET
    ones = np.ones(n)
    return Episode(
        market_id="synthetic-up",
        open_ts=1786665600,
        day="2026-08-14",
        strike=100_000.0,
        settle=100_100.0,
        winner_up=True,
        bid=ones * 0.49,
        ask=ones * 0.51,
        mid=ones * 0.50,
        book_age_ms=np.zeros(n),
        has_book=np.ones(n, dtype=bool),
        n_src=np.full(n, 2),
        s=ones * 100_000.0,
        sigma=ones * 50.0,
        spot=ones * 100_000.0,
        spot_age_ms=np.zeros(n),
        has_spot=np.ones(n, dtype=bool),
    )
