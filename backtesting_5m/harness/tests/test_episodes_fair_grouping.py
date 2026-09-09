"""Characterisation test for the per-market fair lookup in load_episodes.

Written against the boolean-scan implementation
(`rows = fair[fair["market_id"] == market_id]`) BEFORE it was replaced with a
group-once lookup, so it pins the existing behaviour rather than merely
agreeing with its own refactor. The fair fixture interleaves two markets'
rows rather than keeping them contiguous, because contiguous rows would still
pass under a grouping bug (e.g. one that silently sorted or reordered) that a
real fair export -- which is not sorted by market -- would expose.
"""
import pandas as pd
import pytest

from harness.build.episodes import load_episodes


def _strikes():
    return pd.DataFrame({
        "market_id": ["a", "b"],
        "open_ts": [1786665600, 1786665900],
        "strike": [100_000.0, 100_050.0],
    })


def _panel():
    rows = []
    for mid, open_ts in (("a", 1786665600), ("b", 1786665900)):
        for t in (0, 100, 200):
            rows.append({"market_id": mid, "open_ts": open_ts, "t_ms": t,
                         "bid": 0.49, "ask": 0.51, "mid": 0.50, "n_src": 2})
    return pd.DataFrame(rows)


def _interleaved_fair():
    # Rows for "a" and "b" are interleaved in frame order, not grouped
    # contiguously -- the shape a real fair export would actually have.
    return pd.DataFrame({
        "market_id": ["a", "b", "a", "b", "a"],
        "t_ms": [0, 50, 100, 150, 200],
        "s": [100_010.0, 100_060.0, 100_020.0, 100_070.0, 100_030.0],
    })


def test_fair_values_land_at_the_right_bucket_per_market(tmp_path):
    panel_p = tmp_path / "panel.parquet"
    strikes_p = tmp_path / "strikes.parquet"
    fair_p = tmp_path / "fair.parquet"
    _panel().to_parquet(panel_p)
    _strikes().to_parquet(strikes_p)
    _interleaved_fair().to_parquet(fair_p)

    # fair_is_causal=True opts out of the decision-grid shift so this test
    # pins the market lookup itself, not the (separately tested) shift.
    eps = load_episodes(panel_path=str(panel_p), strikes_path=str(strikes_p),
                        fair_path=str(fair_p), fair_is_causal=True)
    by_id = {e.market_id: e for e in eps}

    # bucket index = t_ms // BUCKET_MS(100)
    assert by_id["a"].s[0] == pytest.approx(100_010.0)
    assert by_id["a"].s[1] == pytest.approx(100_020.0)
    assert by_id["a"].s[2] == pytest.approx(100_030.0)

    assert by_id["b"].s[0] == pytest.approx(100_060.0)
    assert by_id["b"].s[1] == pytest.approx(100_070.0)

    # A market absent from `fair` altogether must yield the same all-NaN
    # `ep.s` the boolean mask produced for an empty match -- not a KeyError.
    strikes_c = pd.concat([_strikes(), pd.DataFrame({
        "market_id": ["c"], "open_ts": [1786666200], "strike": [99_900.0]})],
        ignore_index=True)
    strikes_c.to_parquet(strikes_p)
    panel_c = pd.concat([_panel(), pd.DataFrame([
        {"market_id": "c", "open_ts": 1786666200, "t_ms": t,
         "bid": 0.49, "ask": 0.51, "mid": 0.50, "n_src": 2}
        for t in (0, 100, 200)])], ignore_index=True)
    panel_c.to_parquet(panel_p)

    eps = load_episodes(panel_path=str(panel_p), strikes_path=str(strikes_p),
                        fair_path=str(fair_p), fair_is_causal=True)
    by_id = {e.market_id: e for e in eps}
    assert pd.isna(by_id["c"].s).all()
