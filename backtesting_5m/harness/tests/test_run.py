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
    kw.setdefault("execn", ExecConfig())
    kw.setdefault("sample", Sample())
    kw.setdefault("output", Output())
    return run(str(tmp_path), episodes=episodes, **kw)


def test_a_run_writes_a_manifest_and_a_ledger(tmp_path, episodes):
    result = _run(tmp_path, episodes)
    run_dir = result["run_dir"]
    for name in ("manifest.json", "ledger.parquet", "markets.parquet",
                 "summary.json"):
        assert os.path.exists(os.path.join(run_dir, name)), name


def test_the_summary_carries_every_standing_caveat(tmp_path, episodes):
    """All three, not just the grid bias.

    This test used to check the grid-bias key alone, which is how the
    clock-offset caveat came to be written and never reach an output file.
    """
    result = _run(tmp_path, episodes)
    summary = json.loads(
        open(os.path.join(result["run_dir"], "summary.json")).read())
    caveats = summary["caveats"]
    assert {"grid_bias_usd_per_market", "grid_bias_note",
            "maker_fills_are_modelled",
            "clock_offset_is_assumed_not_measured"} <= set(caveats)
    assert caveats["grid_bias_usd_per_market"] == pytest.approx(0.13)


def test_the_clock_offset_caveat_says_it_is_assumed_not_measured(tmp_path,
                                                                 episodes):
    """Spot-derived results must not ship without disclosing this.

    The offset cannot be recovered from these two streams -- they carry
    different events -- so it is configured, defaults to 0.0, and is bounded
    only by this repo's own 0-74 ms same-book drift measurements.
    """
    result = _run(tmp_path, episodes)
    caveats = json.loads(
        open(os.path.join(result["run_dir"], "summary.json")).read())["caveats"]
    note = caveats["clock_offset_is_assumed_not_measured"]
    assert "ASSUMPTION" in note and "not a measurement" in note
    assert caveats["clock_offset_default_s"] == pytest.approx(0.0)
    assert caveats["clock_offset_plausible_range_ms"] == [0.0, 74.0]
    # these synthetic days have no recorded offset, so the run must say so
    assert caveats["clock_offset_s_by_day"] is None
    assert "unknown" in caveats["clock_offset_unknown_to_this_run"]


def test_the_recorded_offsets_are_reported_for_the_days_that_have_them(
        tmp_path, episodes, monkeypatch):
    from dataclasses import replace as dc_replace

    from harness.core import run as run_module

    tsv = tmp_path / "venue_vantage_offsets.tsv"
    header = ["day", "offset_s", "n", "source"]
    row = ["2026-08-14", "0.031", "100", "configured"]
    tsv.write_text("\n".join(["\t".join(header),
                              "\t".join(row), ""]))
    monkeypatch.setattr(run_module, "VENUE_OFFSETS_TSV", str(tsv))

    result = _run(tmp_path, [dc_replace(episodes[0], day="2026-08-14")])
    caveats = result["summary"]["caveats"]
    assert caveats["clock_offset_s_by_day"] == {
        "2026-08-14": pytest.approx(0.031)}
    assert "clock_offset_unknown_to_this_run" not in caveats


def test_the_manifest_fingerprints_the_inputs_it_was_given(tmp_path, episodes):
    """spec 7 wants the panel and spot build identities in the manifest.

    run() never passed `inputs` to write_manifest, so `manifest["inputs"]` was
    [] on every shipped run and no run folder could be matched to the data it
    read.
    """
    panel = tmp_path / "panel.parquet"
    panel.write_bytes(b"not really a parquet, but it has an identity")

    result = _run(tmp_path, episodes, inputs=(str(panel),))
    manifest = json.loads(
        open(os.path.join(result["run_dir"], "manifest.json")).read())

    assert len(manifest["inputs"]) == 1
    got = manifest["inputs"][0]
    assert got["present"] is True
    assert len(got["sha256"]) == 64
    assert got["path"] == str(panel)


def test_the_manifest_records_an_input_that_is_missing(tmp_path, episodes):
    result = _run(tmp_path, episodes,
                  inputs=(str(tmp_path / "absent.parquet"),))
    manifest = json.loads(
        open(os.path.join(result["run_dir"], "manifest.json")).read())
    assert manifest["inputs"][0]["present"] is False


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
                  output=Output(emit_ticks=True, tick_markets=("m0",)))
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


def test_a_fee_schedule_on_the_config_is_honoured(tmp_path, episodes):
    """`ExecConfig.fees` was declared and never read: the run always built a
    default schedule from the fees block. It is a documented run parameter."""
    from harness.blocks.defaults.fees import FeeSchedule

    result = _run(tmp_path, episodes,
                  execn=ExecConfig(fees=FeeSchedule(base_fee_rate=0.0)))
    assert len(result["ledger"]), "nothing traded, so nothing is proven"
    assert (result["ledger"]["fee_usd"] == 0.0).all()


def test_configs_differing_only_in_fill_params_hash_differently(tmp_path,
                                                                episodes):
    """The shipped sweep's failure: `adverse_lag` and `penetration` differ ONLY
    in fill_params, so they produced run folders with the same hash suffix and
    byte-identical config blocks -- you could not tell which arm a folder was.
    """
    a = _run(tmp_path, episodes,
             execn=ExecConfig(fill_params={"penetration": 0.0}))
    b = _run(tmp_path, episodes,
             execn=ExecConfig(fill_params={"penetration": 0.01}))
    assert a["run_dir"].split("__")[-1] != b["run_dir"].split("__")[-1]


def test_the_hashed_config_carries_the_parameters_that_change_a_run(tmp_path,
                                                                    episodes):
    result = _run(tmp_path, episodes)
    manifest = json.loads(
        open(os.path.join(result["run_dir"], "manifest.json")).read())
    assert {"fill_params", "min_tte_s", "max_tte_s", "fees"} <= set(
        manifest["config"])
    assert manifest["config"]["fees"]["base_fee_rate"] == pytest.approx(0.07)


def test_gates_run_on_every_result(tmp_path, episodes):
    result = _run(tmp_path, episodes)
    assert "gates" in result["summary"]
    assert set(result["summary"]["gates"]["gates"]) == {
        "sign_survives_periods", "ci_excludes_zero", "delete_top_10"}


def test_multiple_seeds_are_all_reported(tmp_path, episodes):
    result = _run(tmp_path, episodes, output=Output(seeds=(0, 1, 2)))
    assert len(result["summary"]["per_seed"]) == 3


def test_the_frozen_blocks_travel_with_the_run(tmp_path, episodes):
    from harness.core.provenance import SLOTS
    result = _run(tmp_path, episodes)
    frozen = os.path.join(result["run_dir"], "blocks")
    assert sorted(os.listdir(frozen)) == sorted(f"{s}.py" for s in SLOTS)
