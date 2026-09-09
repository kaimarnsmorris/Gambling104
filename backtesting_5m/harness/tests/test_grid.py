"""Pins that a sweep shares the signal it cannot vary, and only that.

`fair.precompute` and `vol.precompute` take the episode and the two
SignalBlocks and nothing else -- no quote parameter, no seed -- and profiled
at 59.5 ms/episode against 37.3 ms for the whole feed loop. `run()` used to
call them inside the seed loop, so a 3-seed run did 61 % of its work three
times over for identical arrays.

The saving is only safe if the cache key covers everything that can change
the arrays. Keyed too loosely it hands the next model the previous model's
`s` and reports it as a result, which is the failure this harness exists to
prevent -- so most of this file is about the key, not the speed.
"""
import os
from dataclasses import replace

import numpy as np
import pytest

from harness import (Arm, ExecConfig, Output, QuoteParams, Sample,
                     backtest, default_workers, grid_search, linear_grid)
from harness.core.grid import as_arms
from harness.core.run import config_dict
from harness.core.signals import (block_signature, normalise_params,
                                  precompute_signals)


class _Counting:
    """A stand-in SignalBlock that records how often it was asked."""

    def __init__(self, value):
        self.value = value
        self.calls = 0
        self.seen = []

    def precompute(self, ep, **kw):
        self.calls += 1
        self.seen.append(kw)
        return np.full(len(ep), self.value + kw.get("scale", 0.0))


def _modules():
    return {"fair": _Counting(1.0), "vol": _Counting(2.0)}


@pytest.fixture
def episodes(flat_episode):
    out = []
    for k in range(3):
        ep = replace(flat_episode, market_id=f"m{k}",
                     open_ts=1786665600 + 300 * k, day="2026-08-20")
        ep.ask[:] = 0.20
        out.append(ep)
    return out


# -- the cache key ----------------------------------------------------------

def test_the_same_episode_is_computed_once_per_signature(episodes):
    mods, cache = _modules(), {}
    precompute_signals(episodes, mods, "sig", cache)
    precompute_signals(episodes, mods, "sig", cache)
    assert mods["fair"].calls == 3, "the second pass must be free"
    assert mods["vol"].calls == 3


def test_a_different_signature_does_not_reuse_the_arrays(episodes):
    """The whole point of the key. A grid that swaps fair.py or vol.py must
    recompute; reusing here reports the old model's signal under the new
    model's name."""
    mods, cache = _modules(), {}
    precompute_signals(episodes, mods, "sig-a", cache)
    precompute_signals(episodes, mods, "sig-b", cache)
    assert mods["fair"].calls == 6


def test_signal_params_are_part_of_the_key(episodes):
    """A vol-scale sweep changes no file, so the file digest cannot see it.
    Without the params in the key every arm would get arm one's sigma."""
    mods, cache = _modules(), {}
    a = precompute_signals(episodes, mods, "sig", cache, {"scale": 1.0})
    b = precompute_signals(episodes, mods, "sig", cache, {"scale": 2.0})
    assert mods["vol"].calls == 6
    assert a["m0"][1][0] != b["m0"][1][0], "different scale, different sigma"


def test_signal_params_reach_the_block_as_keywords(episodes):
    mods = _modules()
    precompute_signals(episodes, mods, "sig", {}, {"scale": 3.0})
    assert mods["vol"].seen[0] == {"scale": 3.0}


def test_a_block_that_takes_only_the_episode_is_called_unchanged(episodes):
    """Opting in must be optional: the existing contract is `precompute(ep)`
    and an empty params dict has to keep calling it that way."""
    class _Strict:
        def precompute(self, ep):          # no **kwargs on purpose
            return np.zeros(len(ep))

    mods = {"fair": _Strict(), "vol": _Strict()}
    precompute_signals(episodes, mods, "sig", {})          # must not raise


def test_params_are_order_free_so_one_spelling_is_one_entry():
    assert (normalise_params({"scale": 2, "floor": 1})
            == normalise_params({"floor": 1, "scale": 2}))


def test_the_signature_follows_block_content_not_its_path(tmp_path):
    """Two investigations pointing at identical blocks should share an entry;
    one that edits its local vol.py must not keep the old arrays."""
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        os.makedirs(str(d))
        (d / "fair.py").write_text("def precompute(ep):\n    return 1\n")
        (d / "vol.py").write_text("def precompute(ep):\n    return 2\n")
    same = block_signature({"fair": str(a / "fair.py"),
                            "vol": str(a / "vol.py")})
    assert same == block_signature({"fair": str(b / "fair.py"),
                                    "vol": str(b / "vol.py")})

    (b / "vol.py").write_text("def precompute(ep):\n    return 999\n")
    assert block_signature({"fair": str(b / "fair.py"),
                            "vol": str(b / "vol.py")}) != same


def test_the_signature_ignores_the_slots_that_cannot_change_the_signal(
        tmp_path):
    """Only fair and vol produce `s` and `sigma`. Folding the per-tick slots
    into the key would defeat the cache on exactly the sweeps it exists for:
    varying execution or fees changes no signal at all."""
    os.makedirs(str(tmp_path / "x"))
    f, v = tmp_path / "x" / "fair.py", tmp_path / "x" / "vol.py"
    f.write_text("a\n")
    v.write_text("b\n")
    base = {"fair": str(f), "vol": str(v)}
    assert block_signature(base) == block_signature(
        {**base, "execution": "whatever.py", "fees": "other.py"})


def test_an_unreadable_block_never_shares_an_entry(tmp_path):
    """Degrade to "always recompute" rather than collide with a readable
    block -- a signature that silently matched is the wrong-signal bug."""
    assert (block_signature({"fair": str(tmp_path / "nope.py"),
                             "vol": str(tmp_path / "nope2.py")})
            != block_signature({"fair": None, "vol": None}))


# -- signal_params must reach the run folder's identity ---------------------

def test_signal_params_change_the_config_hash():
    """Two arms of a vol-scale sweep differ in NOTHING else -- same quote,
    same execution, same blocks. Left out of the config they would share a
    run folder and the second would overwrite the first, which is the
    failure `fill_params` already caused once."""
    from harness.blocks.defaults.fees import FeeSchedule
    args = (QuoteParams(), ExecConfig(), Sample(), Output(), FeeSchedule())
    a = config_dict(*args, signal_params={"scale": 1.0})
    b = config_dict(*args, signal_params={"scale": 2.0})
    assert a != b
    assert a["signal_params"] == {"scale": 1.0}


# -- the grid entry point ---------------------------------------------------

def test_an_unlabelled_grid_gets_positional_names():
    arms = as_arms([QuoteParams(e_p=0.01), QuoteParams(e_p=0.02)])
    assert [a.label for a in arms] == ["arm 0", "arm 1"]


def test_pairs_and_arms_are_both_accepted():
    arms = as_arms([("wide", QuoteParams(e_p=0.05)),
                    Arm(label="scaled", signal_params={"scale": 2.0})])
    assert [a.label for a in arms] == ["wide", "scaled"]
    assert arms[1].signal_params == (("scale", 2.0),)


def test_duplicate_labels_are_refused():
    """Labels name the runs in the report; two arms under one label plot as
    one line and quietly discard an arm."""
    with pytest.raises(ValueError, match="unique"):
        as_arms([("a", QuoteParams()), ("a", QuoteParams(e_p=0.1))])


def test_linear_grid_varies_one_field_and_names_the_arms():
    arms = linear_grid(QuoteParams(e_p=0.02), "e_z", [0.1, 0.2])
    assert [a.label for a in arms] == ["e_z=0.1", "e_z=0.2"]
    assert [a.quote.e_z for a in arms] == [0.1, 0.2]
    assert all(a.quote.e_p == 0.02 for a in arms), "the base is carried"


def test_default_workers_leaves_a_core_free():
    assert 1 <= default_workers() <= max(1, os.cpu_count() or 2)


def test_the_default_is_one_seed():
    """A seed changes only the latency draws, so a multi-seed grid pays N
    times for a dispersion estimate that ranking does not use. Rank on one,
    re-run the finalists with three."""
    assert Output().seeds == (0,)


def test_a_grid_runs_every_arm_and_shares_the_signal(tmp_path, episodes):
    """End to end: three arms differing only in `e_p`, which cannot touch
    `s` or `sigma`, so the signal is computed once per episode for the whole
    grid rather than once per arm."""
    cache = {}
    res = grid_search(
        grid=linear_grid(QuoteParams(shares=1.0), "e_p", [0.01, 0.02, 0.03]),
        investigation_dir=str(tmp_path), execn=ExecConfig(), sample=Sample(),
        output=Output(), episodes=episodes, signal_cache=cache)

    assert list(res) == ["e_p=0.01", "e_p=0.02", "e_p=0.03"]
    assert len({r["run_dir"] for r in res.values()}) == 3, (
        "each arm keeps its own manifest -- a grid is many identifiable runs")
    for r in res.values():
        assert os.path.exists(os.path.join(r["run_dir"], "summary.json"))
        assert r["summary"]["sample"]["n_selected"] == 3

    assert len(cache) == 3, (
        f"one entry per episode for the whole grid, got {len(cache)} -- "
        "nine would mean the signal was recomputed per arm")


def test_the_cache_is_reusable_across_two_grid_calls(tmp_path, episodes):
    """A session that sweeps, looks, then sweeps again should not pay twice.
    Handing the same dict back is the whole interface."""
    cache = {}
    common = dict(investigation_dir=str(tmp_path), execn=ExecConfig(),
                  sample=Sample(), output=Output(), episodes=episodes,
                  signal_cache=cache)
    grid_search(grid=[("a", QuoteParams(shares=1.0))], **common)
    keys = set(cache)
    grid_search(grid=[("b", QuoteParams(e_p=0.9, shares=1.0))], **common)
    assert set(cache) == keys, "the second grid added no entries"


def test_a_grid_and_a_single_backtest_agree(tmp_path, episodes):
    """The cache must be a speed change and nothing else. Same arm, same
    episodes, same headline -- otherwise a sweep answers a different question
    from the single run it generalises."""
    q = QuoteParams(e_p=0.02, shares=1.0)
    common = dict(investigation_dir=str(tmp_path), execn=ExecConfig(),
                  sample=Sample(), output=Output(), episodes=episodes)
    alone = backtest(quote=q, **common)
    swept = grid_search(grid=[("only", q)], **common)["only"]
    assert swept["summary"]["headline"] == alone.summary["headline"]


def test_a_per_arm_execn_overrides_the_grids(tmp_path, episodes):
    """"Different quoting rules" is not only QuoteParams -- requote cadence
    and latency live on ExecConfig, and an arm has to be able to move them."""
    res = grid_search(
        grid=[Arm(label="slow", quote=QuoteParams(shares=1.0),
                  execn=ExecConfig(requote_every=50)),
              Arm(label="fast", quote=QuoteParams(shares=1.0))],
        investigation_dir=str(tmp_path), execn=ExecConfig(requote_every=10),
        sample=Sample(), output=Output(), episodes=episodes)
    import json
    got = {}
    for label, r in res.items():
        with open(os.path.join(r["run_dir"], "manifest.json")) as fh:
            got[label] = json.load(fh)["config"]["requote_every"]
    assert got == {"slow": 50, "fast": 10}
