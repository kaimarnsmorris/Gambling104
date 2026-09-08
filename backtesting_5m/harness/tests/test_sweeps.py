"""Pins the sweeps that must attach to every headline.

Taker edge is known to lose significance between 200 ms and 500 ms, and every
maker number is conditional on an unmeasurable fill assumption. Reporting a
single number for either is the failure mode these sweeps exist to prevent.
"""
import os

import pandas as pd
import pytest

from harness.blocks.defaults.fees import apply_daily_minimum
from harness.core.config import ExecConfig, Output, QuoteParams, Sample
from harness.core.sweeps import FILL_ARMS, LATENCY_LADDER_MS, run_with_sweeps


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


def _sweep(tmp_path, episodes):
    return run_with_sweeps(
        str(tmp_path),
        quote=QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
        execn=ExecConfig(mode="taker"),
        sample=Sample(),
        output=Output(plots=False),
        episodes=episodes,
    )


def test_the_latency_ladder_is_reported_in_full(tmp_path, episodes):
    result = _sweep(tmp_path, episodes)
    got = [arm["latency_ms"] for arm in result["sweeps"]["latency"]]
    assert got == list(LATENCY_LADDER_MS)


def test_every_fill_arm_is_reported(tmp_path, episodes):
    result = _sweep(tmp_path, episodes)
    assert {a["arm"] for a in result["sweeps"]["fill"]} == set(FILL_ARMS)


def test_slower_latency_never_helps_a_taker(tmp_path, episodes):
    """A cheap book crossed later is crossed at a worse or equal price."""
    result = _sweep(tmp_path, episodes)
    ladder = {a["latency_ms"]: a["headline"]["pnl_per_market"]
              for a in result["sweeps"]["latency"]}
    assert ladder[500.0] <= ladder[0.0] + 1e-9


def test_the_sweeps_are_written_into_the_summary(tmp_path, episodes):
    import json
    result = _sweep(tmp_path, episodes)
    summary = json.loads(
        open(os.path.join(result["run_dir"], "summary.json")).read())
    assert "sweeps" in summary


def test_a_dust_day_is_not_paid_its_rebate():
    """Below $1.00/day per stream, the rebate simply does not arrive."""
    ledger = pd.DataFrame({
        "day": ["2026-08-14", "2026-08-15"],
        "liquidity": [0, 0],
        "fee_usd": [-0.50, -4.00],
    })
    out = apply_daily_minimum(ledger)
    assert out.loc[0, "fee_usd"] == pytest.approx(0.0), "dust day zeroed"
    assert out.loc[1, "fee_usd"] == pytest.approx(-4.00)


def test_taker_fees_are_untouched_by_the_minimum():
    ledger = pd.DataFrame({
        "day": ["2026-08-14"], "liquidity": [1], "fee_usd": [0.10]})
    assert apply_daily_minimum(ledger).loc[0, "fee_usd"] == pytest.approx(0.10)
