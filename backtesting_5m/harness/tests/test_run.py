"""Pins the orchestrator: sample selection, outputs, provenance, gates."""
import json
import os

import numpy as np
import pytest

from harness.core.config import ExecConfig, Output, QuoteParams, Sample
from harness.core.run import run


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


def _run(tmp_path, episodes, **kw):
    kw.setdefault("quote", QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0))
    kw.setdefault("execn", ExecConfig(mode="taker"))
    kw.setdefault("sample", Sample())
    kw.setdefault("output", Output(plots=False))
    return run(str(tmp_path), episodes=episodes, **kw)


def test_a_run_writes_a_manifest_and_a_ledger(tmp_path, episodes):
    result = _run(tmp_path, episodes)
    run_dir = result["run_dir"]
    for name in ("manifest.json", "ledger.parquet", "markets.parquet",
                 "summary.json"):
        assert os.path.exists(os.path.join(run_dir, name)), name


def test_the_summary_carries_the_grid_bias_caveat(tmp_path, episodes):
    result = _run(tmp_path, episodes)
    summary = json.loads(
        open(os.path.join(result["run_dir"], "summary.json")).read())
    assert "grid_bias_usd_per_market" in summary["caveats"]
    assert summary["caveats"]["grid_bias_usd_per_market"] == pytest.approx(0.13)


def test_max_markets_limits_the_sample(tmp_path, episodes):
    result = _run(tmp_path, episodes, sample=Sample(max_markets=2))
    assert result["summary"]["headline"]["n_markets"] == 2


def test_a_day_filter_selects_only_those_days(tmp_path, episodes):
    result = _run(tmp_path, episodes, sample=Sample(days=("2026-08-14",)))
    assert result["markets"]["day"].unique().tolist() == ["2026-08-14"]


def test_a_market_filter_selects_only_those_markets(tmp_path, episodes):
    result = _run(tmp_path, episodes, sample=Sample(markets=("m0", "m3")))
    assert sorted(result["markets"]["market_id"]) == ["m0", "m3"]


def test_tick_output_is_written_only_when_asked(tmp_path, episodes):
    plain = _run(tmp_path, episodes)
    assert not os.path.exists(os.path.join(plain["run_dir"], "ticks.parquet"))

    ticked = _run(tmp_path, episodes,
                  output=Output(emit_ticks=True, tick_markets=("m0",),
                                plots=False))
    ticks_path = os.path.join(ticked["run_dir"], "ticks.parquet")
    assert os.path.exists(ticks_path)
    import pandas as pd
    assert pd.read_parquet(ticks_path)["market_id"].unique().tolist() == ["m0"]


def test_the_ledger_carries_liquidity_fees_and_markout(tmp_path, episodes):
    result = _run(tmp_path, episodes)
    ledger = result["ledger"]
    assert {"liquidity", "fee_usd", "mid_t10", "delta_quality_c"} <= set(
        ledger.columns)
    assert (ledger["fee_usd"] > 0).all(), "taker fees are positive"


def test_gates_run_on_every_result(tmp_path, episodes):
    result = _run(tmp_path, episodes)
    assert "gates" in result["summary"]
    assert set(result["summary"]["gates"]["gates"]) == {
        "sign_survives_periods", "ci_excludes_zero", "delete_top_10"}


def test_multiple_seeds_are_all_reported(tmp_path, episodes):
    result = _run(tmp_path, episodes, output=Output(seeds=(0, 1, 2),
                                                    plots=False))
    assert len(result["summary"]["per_seed"]) == 3


def test_the_frozen_blocks_travel_with_the_run(tmp_path, episodes):
    from harness.core.provenance import SLOTS
    result = _run(tmp_path, episodes)
    frozen = os.path.join(result["run_dir"], "blocks")
    assert sorted(os.listdir(frozen)) == sorted(f"{s}.py" for s in SLOTS)
