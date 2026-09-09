"""Pins stream registration and resolution.

Streams resolve investigation-first, catalog-second, mirroring how blocks
already resolve. The registered name is the LOCAL ALIAS and always wins; a
conforming file's stream.name metadata is informational.
"""
import numpy as np
import pandas as pd
import pytest

from harness.streams import (Stream, StreamInvalid, StreamNotRegistered,
                             TimeKind, clear_registry, register, registered,
                             resolve, write_stream)


@pytest.fixture(autouse=True)
def _clean():
    clear_registry()
    yield
    clear_registry()


def _write(tmp_path, name="venue_l1"):
    base = 1_786_665_600_000_000_000
    df = pd.DataFrame({"recv_ns": base + np.arange(3, dtype="int64") * 10**8,
                       "mid": [1.0, 2.0, 3.0]})
    write_stream(df, str(tmp_path), name=name, asset="BTC", causal=True,
                 recorder="london_recorder")
    return str(tmp_path)


def test_a_conforming_stream_needs_no_declaration(tmp_path):
    register("spot", _write(tmp_path))
    got = resolve("spot")
    assert got.causal is True
    assert got.values == ("mid",)


def test_the_registered_alias_wins_over_the_files_own_name(tmp_path):
    register("my_alias", _write(tmp_path, name="venue_l1"))
    got = resolve("my_alias")
    assert got.name == "my_alias"
    assert got.meta["name"] == "venue_l1"


def test_an_unregistered_name_raises_and_lists_what_is_registered(tmp_path):
    register("spot", _write(tmp_path))
    with pytest.raises(StreamNotRegistered, match="spot"):
        resolve("chainlink")


def test_registering_the_same_name_twice_shadows_investigation_first(tmp_path):
    a = _write(tmp_path / "a", name="one")
    b = _write(tmp_path / "b", name="two")
    register("spot", a)
    register("spot", b)                       # later call wins: investigation
    assert resolve("spot").meta["name"] == "two"


def test_a_legacy_file_needs_an_adapter(tmp_path):
    import os
    d = tmp_path / "legacy"
    os.makedirs(str(d))
    pd.DataFrame({"oracle_ms": [1], "px": [2.0]}).to_parquet(
        str(d / "p.parquet"))
    with pytest.raises(StreamInvalid, match="adapter"):
        register("chainlink", str(d))

    register("chainlink", str(d), adapter=Stream(
        name="chainlink", time_col="oracle_ms", time_kind=TimeKind.RECEIPT,
        time_unit="ms", causal=True))
    assert resolve("chainlink").causal is True


def test_registered_lists_names(tmp_path):
    register("spot", _write(tmp_path))
    assert registered() == ("spot",)
