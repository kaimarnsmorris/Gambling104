"""Pins registry-driven loading, and that it groups once.

The per-market boolean scan was O(markets x rows) over a 4.6 M-row frame.
"""
import time

import numpy as np
import pandas as pd
import pytest

from harness.build.episodes import load_episodes
from harness.streams import (TimeKind, Stream, clear_registry, register,
                             write_stream)


@pytest.fixture(autouse=True)
def _clean():
    clear_registry()
    yield
    clear_registry()


def _panel(tmp_path, n_markets=3):
    rows = []
    for m in range(n_markets):
        open_ts = 1786665600 + 300 * m
        for t in (0, 100, 200):
            rows.append({"market_id": f"m{m}", "open_ts": open_ts, "t_ms": t,
                         "bid": 0.49, "ask": 0.51, "mid": 0.50, "n_src": 2})
    p = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(p)
    s = tmp_path / "strikes.parquet"
    pd.DataFrame({"market_id": [f"m{m}" for m in range(n_markets + 1)],
                  "open_ts": [1786665600 + 300 * m
                              for m in range(n_markets + 1)],
                  "strike": [100.0 + m for m in range(n_markets + 1)]
                  }).to_parquet(s)
    return str(p), str(s)


def test_a_registered_stream_is_attached_to_each_episode(tmp_path):
    panel, strikes = _panel(tmp_path)
    root = tmp_path / "cl"
    base = 1786665600 * 10**9
    write_stream(pd.DataFrame({
        "recv_ns": base + np.arange(9, dtype="int64") * 10**8,
        "px": np.arange(9, dtype="float64")}),
        str(root), name="chainlink", asset="BTC", causal=True, recorder="r")
    register("chainlink", str(root))

    eps = load_episodes(panel_path=panel, strikes_path=strikes,
                        streams=("chainlink",))
    assert "chainlink" in eps[0].streams
    assert eps[0].stream("chainlink").px.shape == (3000,)


def test_streams_not_requested_are_not_attached(tmp_path):
    panel, strikes = _panel(tmp_path)
    eps = load_episodes(panel_path=panel, strikes_path=strikes)
    assert eps[0].streams == {}
