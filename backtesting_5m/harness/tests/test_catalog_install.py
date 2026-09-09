"""Pins that installing the standard catalog cannot revert a caller's shadow.

The spec tells an investigation to `register()` in its `run.py` BEFORE calling
`backtest()`, and says the registered name always wins. `backtest()` installed
the catalog as its first statement -- i.e. after that -- so the shadow was
clobbered: episodes were still built from the shadowed file while the manifest
fingerprinted the catalog's path, sha256 and all. An empty manifest leaves a
run unidentifiable; a confident and wrong one gets believed.
"""
import json
import os

import pytest

from harness import ExecConfig, Output, QuoteParams, Sample, backtest, paths
from harness.core.api import _fingerprinted_inputs
from harness.streams import (Stream, TimeKind, catalog, clear_registry,
                             register, registered, resolve)


@pytest.fixture(autouse=True)
def _clean():
    clear_registry()
    yield
    clear_registry()


def _shadow(tmp_path, name):
    path = tmp_path / f"{name}_shadow.parquet"
    path.write_bytes(b"a shadowed file, with an identity of its own")
    register(name, str(path), adapter=Stream(
        name=name, time_kind=TimeKind.RECEIPT, causal=True))
    return str(path)


def test_install_leaves_a_name_the_caller_already_registered(tmp_path):
    path = _shadow(tmp_path, "chainlink")
    catalog.install()
    assert resolve("chainlink").path == path


def test_install_still_registers_the_names_nobody_shadowed(tmp_path):
    _shadow(tmp_path, "chainlink")
    catalog.install()
    assert resolve("spot_usd").path == paths.SPOT_USD


def test_install_is_idempotent():
    catalog.install()
    before = registered()
    catalog.install()
    assert registered() == before


def test_an_explicit_register_still_overwrites(tmp_path):
    """The shadowing feature itself: only `install()` defers."""
    catalog.install()
    path = _shadow(tmp_path, "chainlink")
    assert resolve("chainlink").path == path


def test_the_manifest_names_the_shadowed_file_not_the_catalogs(
        tmp_path, flat_episode):
    """The production ordering, end to end: register a shadow, then call
    `backtest()`, which installs the catalog as its first statement."""
    path = _shadow(tmp_path, "chainlink")
    result = backtest(investigation_dir=str(tmp_path), episodes=[flat_episode],
                      quote=QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
                      execn=ExecConfig(), sample=Sample(), output=Output(),
                      streams=("chainlink",))

    assert resolve("chainlink").path == path, "install() reverted the shadow"
    manifest = json.loads(
        open(os.path.join(result.run_dir, "manifest.json")).read())
    assert [f["path"] for f in manifest["inputs"]] == [path]
    assert manifest["inputs"][0]["path"] != paths.RTDS_BTC


# -- strikes is documented as a standard stream, so it is one ---------------

def test_strikes_is_registered_at_the_documented_path():
    """The spec's own example passes `["book", "strikes", "spot",
    "chainlink"]`, and an unregistered name is skipped silently by
    `_fingerprinted_inputs` -- so a caller copying it got a manifest with no
    strikes_5m.parquet in it, while looking complete."""
    catalog.install()
    assert resolve("strikes").path == paths.STRIKES


def test_strikes_is_pre_gridded_so_it_is_never_routed_through_the_grid():
    """`open_ts` is the market open, not an observation receipt. Gridding it
    would compute a hugely negative offset and drop every row in silence."""
    catalog.install()
    assert resolve("strikes").pre_gridded is True


def test_the_specs_own_stream_list_fingerprints_the_strikes_file():
    catalog.install()
    got = _fingerprinted_inputs((), ("book", "strikes", "spot", "chainlink"))
    assert paths.STRIKES in got
    assert paths.PANEL in got
