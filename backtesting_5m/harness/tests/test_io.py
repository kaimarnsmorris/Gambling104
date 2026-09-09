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


# --- directories whose files disagree about their columns ---------------
#
# stream_venue_l1 does this on 2026-08-18: 12,961 of 13,438 files carry the
# okx and bybit blocks and 477 do not. polars takes the first file it globs
# as the schema for the scan and raises "extra column in file outside of
# expected schema: okx_bid" on the rest, which cost that whole day. The
# columns the spot build actually asks for are present in every file.


@pytest.fixture
def mixed_schema_dir(tmp_path):
    """Two files that agree on the wanted columns and disagree on the rest."""
    d = tmp_path / "date=2026-08-18"
    d.mkdir()
    pd.DataFrame({"ts": [1.0, 2.0], "bn_spot_mid": [10.0, 11.0]}).to_parquet(
        d / "a.parquet", index=False)
    pd.DataFrame({"ts": [3.0], "bn_spot_mid": [12.0],
                  "okx_bid": [99.0]}).to_parquet(d / "b.parquet", index=False)
    return d


def test_the_raw_polars_glob_really_does_reject_this(mixed_schema_dir):
    """The failure being worked around, pinned so the retry is not cargo."""
    import polars as pl
    with pytest.raises(Exception, match="okx_bid"):
        pl.read_parquet(str(mixed_schema_dir / "**" / "*.parquet"),
                        columns=["ts", "bn_spot_mid"])


def test_a_mixed_schema_directory_still_reads_the_wanted_columns(
        monkeypatch, mixed_schema_dir):
    def broken_pandas_read(*args, **kwargs):
        raise OSError("Repetition level histogram size mismatch")

    monkeypatch.setattr(io.pd, "read_parquet", broken_pandas_read)

    out = io.read_parquet(str(mixed_schema_dir),
                          columns=["ts", "bn_spot_mid"])
    assert list(out.columns) == ["ts", "bn_spot_mid"]
    assert sorted(out["ts"]) == [1.0, 2.0, 3.0]
    assert "okx_bid" not in out.columns


def test_the_schema_group_retry_reads_a_mixed_directory_on_its_own(
        mixed_schema_dir):
    """The last resort, exercised directly: it must not need the cheap path."""
    out = io._read_dir_by_schema_group(str(mixed_schema_dir),
                                       ["ts", "bn_spot_mid"])
    assert sorted(out["ts"]) == [1.0, 2.0, 3.0]


def test_the_schema_group_retry_keeps_every_column_when_none_are_named(
        mixed_schema_dir):
    """Diagonal, not intersecting: a column present in one group survives."""
    out = io._read_dir_by_schema_group(str(mixed_schema_dir), None)
    assert set(out.columns) == {"ts", "bn_spot_mid", "okx_bid"}
    assert out["okx_bid"].isna().sum() == 2


def test_a_column_missing_from_some_files_is_an_error_not_a_null(
        monkeypatch, tmp_path):
    """Silence would be the dangerous answer here.

    A requested column absent from part of the archive means the caller's
    request cannot be honoured. Filling it with nulls would hand the spot
    build a day of unpriced rows that look merely sparse.
    """
    d = tmp_path / "date=2026-08-18"
    d.mkdir()
    pd.DataFrame({"ts": [1.0], "usdt_basis": [40.0]}).to_parquet(
        d / "a.parquet", index=False)
    pd.DataFrame({"ts": [2.0]}).to_parquet(d / "b.parquet", index=False)

    def broken_pandas_read(*args, **kwargs):
        raise OSError("Repetition level histogram size mismatch")

    monkeypatch.setattr(io.pd, "read_parquet", broken_pandas_read)

    with pytest.raises(OSError):
        io.read_parquet(str(d), columns=["ts", "usdt_basis"])
