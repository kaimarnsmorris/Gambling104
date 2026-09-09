"""Pins the public surface: one import, one call, a result object."""
import os

import pytest

import harness
from harness import ExecConfig, Output, QuoteParams, Sample, backtest


@pytest.fixture
def episodes(flat_episode):
    from dataclasses import replace
    out = []
    for k in range(6):
        ep = replace(flat_episode, market_id=f"m{k}",
                     open_ts=1786665600 + 300 * k,
                     day=f"2026-08-{14 + k % 3:02d}")
        ep.ask[:] = 0.20
        out.append(ep)
    return out


def test_the_public_names_are_importable_from_harness():
    for n in ("backtest", "Sample", "QuoteParams", "ExecConfig", "Output",
              "register", "Stream", "TimeKind"):
        assert hasattr(harness, n), n


def test_backtest_returns_a_result_object(tmp_path, episodes):
    r = backtest(investigation_dir=str(tmp_path), episodes=episodes,
                 quote=QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
                 execn=ExecConfig(), sample=Sample(),
                 output=Output())
    assert os.path.exists(os.path.join(r.run_dir, "summary.json"))
    assert r.summary["headline"]["n_markets"] == 6
    assert len(r.markets) == 6
    assert r.ledger is not None


def test_the_result_carries_the_drop_counts(tmp_path, episodes):
    r = backtest(investigation_dir=str(tmp_path), episodes=episodes,
                 quote=QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
                 execn=ExecConfig(),
                 sample=Sample(require=("chainlink",)),
                 output=Output())
    assert r.summary["sample"]["dropped"]["chainlink"] == 6
