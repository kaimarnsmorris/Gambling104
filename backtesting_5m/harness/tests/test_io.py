"""harness.io.read_parquet: the normal path, and the polars fallback path.

The fallback exists for a reader/writer version mismatch we cannot
synthesise a fixture for (it needs a parquet-cpp-arrow 24 writer, which isn't
installed here). So instead of faking a bad file, these tests monkeypatch
the primary reader to fail and confirm the polars path still returns the
correct frame.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from harness import io  # noqa: E402


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "market_id": ["a", "a", "b"],
        "open_ts": [100, 100, 200],
        "value": [1.0, 2.0, 3.0],
    })


@pytest.fixture
def sample_parquet(tmp_path, sample_df):
    path = tmp_path / "sample.parquet"
    sample_df.to_parquet(path, index=False)
    return path


def test_normal_path_round_trips_a_pandas_written_file(sample_parquet, sample_df):
    out = io.read_parquet(sample_parquet)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), sample_df.reset_index(drop=True))


def test_columns_selects_a_subset(sample_parquet):
    out = io.read_parquet(sample_parquet, columns=["market_id", "value"])
    assert list(out.columns) == ["market_id", "value"]


def test_nonexistent_path_raises(tmp_path):
    with pytest.raises(Exception):
        io.read_parquet(tmp_path / "does_not_exist.parquet")


def test_fallback_to_polars_returns_the_same_frame(
        monkeypatch, sample_parquet, sample_df):
    def broken_read_parquet(*args, **kwargs):
        raise OSError("Repetition level histogram size mismatch")

    monkeypatch.setattr(io.pd, "read_parquet", broken_read_parquet)

    out = io.read_parquet(sample_parquet)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True).sort_values("open_ts").reset_index(drop=True),
        sample_df.sort_values("open_ts").reset_index(drop=True))


def test_fallback_applies_filters_in_pandas(monkeypatch, sample_parquet, sample_df):
    def broken_read_parquet(*args, **kwargs):
        raise OSError("Repetition level histogram size mismatch")

    monkeypatch.setattr(io.pd, "read_parquet", broken_read_parquet)

    out = io.read_parquet(
        sample_parquet, filters=[("market_id", "in", ["a"])])
    assert set(out["market_id"]) == {"a"}
    assert len(out) == 2


def test_fallback_raises_original_error_when_polars_also_unavailable(
        monkeypatch, sample_parquet):
    def broken_read_parquet(*args, **kwargs):
        raise OSError("Repetition level histogram size mismatch")

    monkeypatch.setattr(io.pd, "read_parquet", broken_read_parquet)

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "polars":
            raise ImportError("polars not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(OSError, match="Repetition level histogram"):
        io.read_parquet(sample_parquet)


def test_both_readers_fail_surfaces_both_errors(monkeypatch, sample_parquet):
    def broken_pandas_read(*args, **kwargs):
        raise OSError("PRIMARY-BOOM")

    def broken_polars_read(*args, **kwargs):
        raise ValueError("FALLBACK-BOOM")

    monkeypatch.setattr(io.pd, "read_parquet", broken_pandas_read)

    import polars as pl
    monkeypatch.setattr(pl, "read_parquet", broken_polars_read)

    with pytest.raises(OSError) as exc_info:
        io.read_parquet(sample_parquet)

    error_message = str(exc_info.value)
    assert "PRIMARY-BOOM" in error_message, f"Primary error not in message: {error_message}"
    assert "FALLBACK-BOOM" in error_message, f"Fallback error not in message: {error_message}"


def test_fallback_succeeds_when_only_pandas_fails(monkeypatch, sample_parquet, sample_df):
    def broken_pandas_read(*args, **kwargs):
        raise OSError("Repetition level histogram size mismatch")

    monkeypatch.setattr(io.pd, "read_parquet", broken_pandas_read)

    out = io.read_parquet(sample_parquet)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True).sort_values("open_ts").reset_index(drop=True),
        sample_df.sort_values("open_ts").reset_index(drop=True))
