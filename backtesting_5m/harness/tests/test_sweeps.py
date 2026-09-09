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
from harness.core.sweeps import (FILL_ARMS, LATENCY_LADDER_MS,
                                  _resolve_fill_arm, run_with_sweeps)


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
        execn=ExecConfig(),
        sample=Sample(),
        output=Output(),
        episodes=episodes,
    )


def test_the_latency_ladder_is_reported_in_full(tmp_path, episodes):
    result = _sweep(tmp_path, episodes)
    got = [arm["latency_ms"] for arm in result["sweeps"]["latency"]]
    assert got == list(LATENCY_LADDER_MS)


def test_every_fill_arm_is_reported(tmp_path, episodes):
    result = _sweep(tmp_path, episodes)
    assert {a["arm"] for a in result["sweeps"]["fill"]} == set(FILL_ARMS)


def test_every_fill_arm_resolves_to_a_distinct_configuration():
    """The failure this file exists to prevent: two arms that are secretly
    the same run reported as two data points. Compare the actual triple each
    arm resolves to (fill params, cancel latency, move-cancel latency), not
    just the dict keys."""
    base = ExecConfig()
    triples = []
    for name, arm_spec in FILL_ARMS.items():
        resolved = _resolve_fill_arm(base, arm_spec)
        triples.append((
            name,
            tuple(sorted(resolved.fill_params.items())),
            resolved.latency.cancel_ms,
            resolved.latency.move_cancel_ms,
        ))
    configs = [t[1:] for t in triples]
    assert len(configs) == len(set(configs)), (
        f"duplicate effective fill-arm configuration(s): {triples}")


def test_optimistic_arm_has_zero_cancel_latency():
    """optimistic must be a true upper bound: you always pull in time, on
    both the ordinary and the in-move cancel path."""
    base = ExecConfig(latency=ExecConfig().latency.__class__(
        cancel_ms=100.0, move_cancel_ms=50.0))
    resolved = _resolve_fill_arm(base, FILL_ARMS["optimistic"])
    assert resolved.latency.cancel_ms == 0.0
    assert resolved.latency.move_cancel_ms is None


def test_adverse_lag_and_penetration_track_the_callers_latency():
    """These two arms differ only in penetration -- the cancel latency the
    caller configured must pass through unchanged."""
    base = ExecConfig(latency=ExecConfig().latency.__class__(
        cancel_ms=250.0, move_cancel_ms=75.0))
    for name in ("adverse_lag", "penetration"):
        resolved = _resolve_fill_arm(base, FILL_ARMS[name])
        assert resolved.latency.cancel_ms == 250.0
        assert resolved.latency.move_cancel_ms == 75.0


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


def test_dust_is_measured_per_seed_rather_than_pooled_across_them():
    """A seed is a replay of the same calendar, not another day of earnings.

    Grouping on `day` alone summed one day's rebate across every seed, so a
    3-seed ledger reported roughly three times the day's earnings and lifted
    genuine dust days over the floor -- an error that can only ever be too
    generous to the maker, which is the one direction this harness must never
    be wrong in.
    """
    ledger = pd.DataFrame({
        "day": ["2026-08-14", "2026-08-14"],
        "seed": [0, 1],
        "liquidity": [0, 0],
        "fee_usd": [-0.60, -0.60],
    })
    assert -ledger["fee_usd"].sum() > 1.0, "pooled, this day clears the floor"

    out = apply_daily_minimum(ledger)
    assert out["fee_usd"].tolist() == [0.0, 0.0], (
        "each seed earned $0.60 on this day, which is dust in both replays")


def test_a_paid_day_is_still_paid_in_every_seed():
    ledger = pd.DataFrame({
        "day": ["2026-08-14", "2026-08-14"],
        "seed": [0, 1],
        "liquidity": [0, 0],
        "fee_usd": [-4.00, -4.00],
    })
    assert apply_daily_minimum(ledger)["fee_usd"].tolist() == [-4.00, -4.00]
