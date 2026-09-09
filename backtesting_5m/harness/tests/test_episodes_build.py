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


# --- warm-up ------------------------------------------------------------
#
# An Episode is one market, so `precompute` used to see nothing before the
# open and every stateful estimator restarted every 300 s. The loader now
# assembles the seconds before the open out of the PRIOR markets' own rows.


def _wide_strikes(n=6, t0=1786665600):
    return pd.DataFrame({
        "market_id": ["m%d" % k for k in range(n)],
        "open_ts": [t0 + 300 * k for k in range(n)],
        "strike": [100_000.0 + k for k in range(n)],
    })


def _wide_panel(n=6, t0=1786665600):
    rows = []
    for k in range(n):
        for t in (0, 100, 200):
            rows.append({"market_id": "m%d" % k, "open_ts": t0 + 300 * k,
                         "t_ms": t, "bid": 0.49, "ask": 0.51, "mid": 0.50,
                         "n_src": 2})
    return pd.DataFrame(rows)


def _wide_spot(n=6, t0=1786665600):
    """One spot observation per second, per market, priced by absolute time."""
    rows = []
    for k in range(n):
        open_ts = t0 + 300 * k
        for sec in range(300):
            rows.append({"open_ts": open_ts, "t_ms": sec * 1000,
                         "spot": 100_000.0 + 300 * k + sec,
                         "spot_usdt": 100_040.0 + 300 * k + sec,
                         "usdt_basis": 40.0})
    return pd.DataFrame(rows)


def _wide_paths_at(tmp_path, t0, n=6):
    panel_p = tmp_path / "p.parquet"
    strikes_p = tmp_path / "s.parquet"
    spot_p = tmp_path / "sp.parquet"
    _wide_panel(n, t0).to_parquet(panel_p)
    _wide_strikes(n, t0).to_parquet(strikes_p)
    _wide_spot(n, t0).to_parquet(spot_p)
    return str(panel_p), str(strikes_p), str(spot_p)


def _wide_paths(tmp_path, n=6):
    panel_p = tmp_path / "p.parquet"
    strikes_p = tmp_path / "s.parquet"
    spot_p = tmp_path / "sp.parquet"
    _wide_panel(n).to_parquet(panel_p)
    _wide_strikes(n).to_parquet(strikes_p)
    _wide_spot(n).to_parquet(spot_p)
    return str(panel_p), str(strikes_p), str(spot_p)


def test_warm_up_is_off_unless_asked_for(tmp_path):
    """The default must be what this loader produced before warm-up existed."""
    panel_p, strikes_p, spot_p = _wide_paths(tmp_path)
    eps = load_episodes(panel_path=panel_p, strikes_path=strikes_p,
                        spot_path=spot_p)
    assert all(e.warmup_n == 0 and not e.has_warmup for e in eps)


def test_requesting_warm_up_leaves_every_in_window_array_unchanged(tmp_path):
    panel_p, strikes_p, spot_p = _wide_paths(tmp_path)
    cold = load_episodes(panel_path=panel_p, strikes_path=strikes_p,
                         spot_path=spot_p)
    warm = load_episodes(panel_path=panel_p, strikes_path=strikes_p,
                         spot_path=spot_p, warmup=True, warmup_s=900.0)
    assert [e.market_id for e in cold] == [e.market_id for e in warm]
    for a, b in zip(cold, warm):
        for name in ("bid", "ask", "mid", "book_age_ms", "spot",
                     "spot_age_ms", "spot_usdt", "s"):
            np.testing.assert_array_equal(
                np.nan_to_num(getattr(a, name), nan=-1.0, posinf=-2.0),
                np.nan_to_num(getattr(b, name), nan=-1.0, posinf=-2.0),
                err_msg="%s moved on %s" % (name, a.market_id))


def test_warm_up_is_the_prior_markets_spot_on_one_continuous_grid(tmp_path):
    """Three prior markets, relabelled onto a single 900 s grid.

    `_wide_spot` prices market k at 100000 + 300k + second, so the value at
    warm-up second `j` is exactly the value 900 - j seconds before the open --
    which is what pins that the relabelling arithmetic is right and not merely
    self-consistent.
    """
    panel_p, strikes_p, spot_p = _wide_paths(tmp_path)
    eps = {e.market_id: e for e in load_episodes(
        panel_path=panel_p, strikes_path=strikes_p, spot_path=spot_p,
        warmup=True, warmup_s=900.0)}
    ep = eps["m4"]
    assert ep.warmup_n == 9000 and ep.has_warmup
    # index 10 acts on warm-up bucket 9, i.e. 0.9 s into the region, which is
    # the observation stamped at second 0 of the market three back (m1).
    assert ep.warmup_spot[10] == pytest.approx(100_000.0 + 300 * 1 + 0)
    # one second later, still m1
    assert ep.warmup_spot[20] == pytest.approx(100_000.0 + 300 * 1 + 1)
    # the far end of the region belongs to m3, the market just before this one
    assert ep.warmup_spot[8990] == pytest.approx(
        100_000.0 + 300 * 3 + 298)
    assert ep.warmup_spot_usdt[10] == pytest.approx(100_040.0 + 300 * 1 + 0)


def test_early_markets_get_a_full_length_nan_region_and_no_flag(tmp_path):
    """The boundary. NaN and `has_warmup=False`, never a short array."""
    panel_p, strikes_p, spot_p = _wide_paths(tmp_path)
    eps = {e.market_id: e for e in load_episodes(
        panel_path=panel_p, strikes_path=strikes_p, spot_path=spot_p,
        warmup=True, warmup_s=900.0)}
    first = eps["m0"]
    assert first.warmup_n == 9000, "the shape is the contract"
    assert first.has_warmup is False
    assert not np.isfinite(first.warmup_spot).any()
    # m3 is the first open with a full 900 s of prior panel behind it
    assert eps["m3"].has_warmup is True


def test_an_absent_oracle_is_an_outage_not_a_boundary(tmp_path):
    """`has_warmup` True with an all-NaN oracle line is the distinction.

    No `rtds_path` here, so the Chainlink warm-up is empty -- exactly what an
    episode past the end of the oracle capture looks like. `has_warmup` still
    says the region is real, so a block can tell that apart from the start of
    the sample.
    """
    panel_p, strikes_p, spot_p = _wide_paths(tmp_path)
    ep = {e.market_id: e for e in load_episodes(
        panel_path=panel_p, strikes_path=strikes_p, spot_path=spot_p,
        warmup=True, warmup_s=900.0)}["m4"]
    assert ep.has_warmup is True
    assert np.isfinite(ep.warmup_spot).any()
    assert not np.isfinite(ep.warmup_chainlink).any()


def test_warm_up_ignores_the_day_filter_when_gathering_history(tmp_path):
    """A day filter says what to trade, not what happened before it.

    Only the last market is selected, and its warm-up still comes from the
    three markets before it -- which the filter excluded from the run.
    """
    panel_p, strikes_p, spot_p = _wide_paths(tmp_path)
    eps = load_episodes(panel_path=panel_p, strikes_path=strikes_p,
                        spot_path=spot_p, markets=["m5"],
                        warmup=True, warmup_s=900.0)
    assert len(eps) == 1 and eps[0].has_warmup
    assert np.isfinite(eps[0].warmup_spot).sum() > 8000


def test_a_shorter_warm_up_is_honoured(tmp_path):
    panel_p, strikes_p, spot_p = _wide_paths(tmp_path)
    ep = {e.market_id: e for e in load_episodes(
        panel_path=panel_p, strikes_path=strikes_p, spot_path=spot_p,
        warmup=True, warmup_s=300.0)}["m4"]
    assert ep.warmup_n == 3000 and ep.warmup_s == pytest.approx(300.0)
    assert ep.warmup_spot[10] == pytest.approx(100_000.0 + 300 * 3 + 0)


def test_a_day_filter_does_not_make_the_days_first_markets_look_early(
        tmp_path):
    """The other half of "warm-up ignores the day filter".

    Six markets straddling midnight, and only the later day selected. Its
    first selected market has three markets of real history behind it, so
    `has_warmup` must be True -- a `has_warmup` computed off the FILTERED
    panel would call it the start of the sample and hand a block a cold
    start on every day boundary in the run.
    """
    t0 = 1786752000 - 900          # 2026-08-14 23:45 UTC
    panel_p, strikes_p, spot_p = _wide_paths_at(tmp_path, t0)
    eps = load_episodes(panel_path=panel_p, strikes_path=strikes_p,
                        spot_path=spot_p, days=["2026-08-15"],
                        warmup=True, warmup_s=900.0)
    assert [e.market_id for e in eps] == ["m3", "m4", "m5"]
    assert eps[0].has_warmup is True
    assert np.isfinite(eps[0].warmup_spot).sum() > 8000


def test_an_incomplete_region_is_flagged_false_but_still_carried(tmp_path):
    """`has_warmup` says COMPLETE, not NON-EMPTY.

    m2 has two prior markets behind it and needs three, so its region
    straddles the start of the sample: the flag is False and the data that
    does exist is still there. Throwing it away would be worse -- a block may
    perfectly reasonably use a short history if it knows it is short -- and
    calling it True would let a block trust a region a third of which never
    existed.
    """
    panel_p, strikes_p, spot_p = _wide_paths(tmp_path)
    ep = {e.market_id: e for e in load_episodes(
        panel_path=panel_p, strikes_path=strikes_p, spot_path=spot_p,
        warmup=True, warmup_s=900.0)}["m2"]
    assert ep.has_warmup is False
    finite = int(np.isfinite(ep.warmup_spot).sum())
    assert 5000 < finite < 9000, finite
    # and what IS there is the tail of the region, nearest the open
    assert not np.isfinite(ep.warmup_spot[:2900]).any()
    assert np.isfinite(ep.warmup_spot[-1])
