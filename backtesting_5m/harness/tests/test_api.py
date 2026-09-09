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


# -- the manifest identifies the data a public-API run read ----------------

def _manifest(result):
    import json
    return json.loads(
        open(os.path.join(result.run_dir, "manifest.json")).read())


def _register_file(tmp_path, name):
    from harness import Stream, register
    from harness.streams.spec import TimeKind

    path = tmp_path / f"{name}.parquet"
    path.write_bytes(b"not really a parquet, but it has an identity")
    register(name, str(path), adapter=Stream(
        name=name, time_kind=TimeKind.RECEIPT, causal=True))
    return str(path)


def _backtest(tmp_path, episodes, **kw):
    return backtest(investigation_dir=str(tmp_path), episodes=episodes,
                    quote=QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
                    execn=ExecConfig(), sample=Sample(), output=Output(), **kw)


def test_a_named_stream_is_fingerprinted_into_the_manifest(tmp_path, episodes):
    """`backtest()` passed no `inputs` at all, so every run made through the
    documented entry point had an empty `manifest["inputs"]` and could not be
    matched to the data it read. Naming a stream is enough: the registry
    already knows its path, and the caller who names streams is exactly the
    caller who will not list those paths a second time by hand.
    """
    path = _register_file(tmp_path, "probe_stream")
    manifest = _manifest(_backtest(tmp_path, episodes,
                                   streams=("probe_stream",)))

    assert [f["path"] for f in manifest["inputs"]] == [path]
    assert len(manifest["inputs"][0]["sha256"]) == 64


def test_an_explicit_input_is_kept_and_not_duplicated_by_its_stream(
        tmp_path, episodes):
    """One file, named twice, is one input -- in the caller's own spelling.

    The registry holds one path and the caller may write the same file
    another way, so the union is deduped on the resolved path and the
    explicit one wins.
    """
    path = _register_file(tmp_path, "probe_dedupe")
    spelt_differently = os.path.join(os.path.dirname(path), os.curdir,
                                     os.path.basename(path))
    assert spelt_differently != path

    manifest = _manifest(_backtest(tmp_path, episodes,
                                   inputs=(spelt_differently,),
                                   streams=("probe_dedupe",)))
    assert [f["path"] for f in manifest["inputs"]] == [spelt_differently]


def test_an_unregistered_stream_name_does_not_fail_the_run(tmp_path, episodes):
    """The streams themselves are resolved downstream, where a missing one is
    reported properly. A manifest detail must never be what fails a run."""
    result = _backtest(tmp_path, episodes, streams=("nothing_registers_this",))
    assert result.summary["headline"]["n_markets"] == 6
    assert _manifest(result)["inputs"] == []
