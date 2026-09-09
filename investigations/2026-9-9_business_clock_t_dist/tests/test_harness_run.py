"""One real run, end to end, through the public API.

This is the test the previous block layer could not have passed. Its `link` took a
tick index the harness never supplies, so the blocks were exercised only by tests
that called them the way the tests found convenient. Nothing here calls a block
directly: `backtest()` resolves the slots, builds the episodes and drives the loop,
and the assertions are about what came out the far end.
"""
from __future__ import annotations

import os
import pathlib

import pytest

MODEL_DIR = os.path.join(
    pathlib.Path(__file__).resolve().parents[3], "models", "chainlink_fv")

pytestmark = pytest.mark.slow


def test_the_model_directory_supplies_fair_vol_and_link_and_nothing_else():
    """A model that overrides less is a model with less to go wrong. `f` is not
    ours because the harness default -- (level - strike)/sigma -- is already
    exactly our convention, and quote/execution/fill/fees are the venue, not us."""
    from harness.core import provenance

    resolved = provenance.resolve_slots("no_such_investigation", model_dir=MODEL_DIR)
    ours = {slot for slot, path in resolved.items()
            if os.path.dirname(path) == MODEL_DIR}
    assert ours == {"fair", "vol", "link"}, ours
    for slot in ("f", "quote", "execution", "fill", "fees"):
        assert "defaults" in resolved[slot], slot


def test_a_short_run_selects_markets_quotes_and_fills(tmp_path):
    """The whole chain against real data. Not a smoke test: a run that selects
    markets and fills nothing is the documented way this fails silently, so both
    counts are asserted."""
    from harness import ExecConfig, Output, QuoteParams, Sample, backtest, paths
    from harness.build.episodes import load_episodes

    episodes = load_episodes(days=("2026-08-20",), max_markets=25,
                             fair_is_causal=True)
    assert episodes, "no episodes built for the test day"

    r = backtest(
        investigation_dir=str(tmp_path),
        model=MODEL_DIR,
        quote=QuoteParams(e_p=0.01, max_pos=50.0, shares=10.0),
        execn=ExecConfig(),
        sample=Sample(),
        output=Output(seeds=(0,)),
        episodes=episodes,
        inputs=(paths.PANEL, paths.STRIKES),
    )

    assert r.summary["sample"]["n_selected"] > 0, r.summary["sample"]["dropped"]
    assert r.summary["headline"]["n_fills"] > 0, (
        "selected %d markets and filled none -- the model quoted nothing, which "
        "means `s` or `sigma` came back NaN"
        % r.summary["sample"]["n_selected"])
    assert os.path.exists(os.path.join(r.run_dir, "summary.json"))


def test_the_run_folder_records_which_files_were_the_model(tmp_path):
    """Provenance is the reason a model directory is safe to share: the run freezes
    the resolved blocks, so `models/chainlink_fv` moving on later cannot rewrite what
    this run was."""
    import json

    from harness import ExecConfig, Output, QuoteParams, Sample, backtest
    from harness.build.episodes import load_episodes

    episodes = load_episodes(days=("2026-08-20",), max_markets=5,
                             fair_is_causal=True)
    r = backtest(investigation_dir=str(tmp_path), model=MODEL_DIR,
                 quote=QuoteParams(e_p=0.01, max_pos=50.0, shares=10.0),
                 execn=ExecConfig(), sample=Sample(), output=Output(seeds=(0,)),
                 episodes=episodes)
    manifest = json.loads(
        open(os.path.join(r.run_dir, "manifest.json")).read())
    blocks = json.dumps(manifest)
    for slot in ("fair", "vol", "link"):
        assert "chainlink_fv" in blocks and slot in blocks, slot
