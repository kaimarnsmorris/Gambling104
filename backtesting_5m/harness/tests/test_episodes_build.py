"""Pins episode assembly from panel-shaped frames.

Settlement is not stored anywhere: settle(N) IS the strike of the market
opening 300 s later. Verified exact on 7,330 of 7,330 back-to-back pairs.
"""
import numpy as np
import pandas as pd
import pytest

from harness.build.episodes import load_episodes, settlement_map


def _strikes():
    return pd.DataFrame({
        "market_id": ["a", "b", "c"],
        "open_ts": [1786665600, 1786665900, 1786666200],
        "strike": [100_000.0, 100_050.0, 99_900.0],
    })


def _panel():
    rows = []
    for mid, open_ts in (("a", 1786665600), ("b", 1786665900)):
        for t in (0, 100, 200):
            rows.append({"market_id": mid, "open_ts": open_ts, "t_ms": t,
                         "bid": 0.49, "ask": 0.51, "mid": 0.50, "n_src": 2})
    return pd.DataFrame(rows)


def test_settlement_is_the_next_markets_strike():
    m = settlement_map(_strikes())
    assert m[1786665600] == pytest.approx(100_050.0)
    assert m[1786665900] == pytest.approx(99_900.0)


def test_the_last_market_has_no_settlement():
    assert settlement_map(_strikes()).get(1786666200) is None


def test_episodes_are_built_with_settlement_and_winner(tmp_path):
    panel_p = tmp_path / "panel.parquet"
    strikes_p = tmp_path / "strikes.parquet"
    _panel().to_parquet(panel_p)
    _strikes().to_parquet(strikes_p)

    eps = load_episodes(panel_path=str(panel_p), strikes_path=str(strikes_p))
    by_id = {e.market_id: e for e in eps}
    assert by_id["a"].settle == pytest.approx(100_050.0)
    assert by_id["a"].winner_up is True       # 100050 >= 100000
    assert by_id["b"].winner_up is False      # 99900 < 100050


def test_each_episode_is_a_full_length_grid(tmp_path):
    panel_p, strikes_p = tmp_path / "p.parquet", tmp_path / "s.parquet"
    _panel().to_parquet(panel_p)
    _strikes().to_parquet(strikes_p)
    ep = load_episodes(panel_path=str(panel_p), strikes_path=str(strikes_p))[0]
    assert len(ep.bid) == 3000
    assert np.isnan(ep.bid[0]), "the causality shift still applies"


def test_max_markets_is_honoured(tmp_path):
    panel_p, strikes_p = tmp_path / "p.parquet", tmp_path / "s.parquet"
    _panel().to_parquet(panel_p)
    _strikes().to_parquet(strikes_p)
    assert len(load_episodes(panel_path=str(panel_p),
                             strikes_path=str(strikes_p),
                             max_markets=1)) == 1
