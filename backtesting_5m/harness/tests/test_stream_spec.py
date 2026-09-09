"""Pins the stream standard.

The rule that carries most of the value: alignment is ALWAYS `recv_ns`, never
`src_ns`. A source stamp can LEAD receipt -- rtds_btc's oracle_ms leads its own
px_first_recv_ns by ~1.5 s -- so aligning on it feeds a model prices before it
was told them. Making receipt the only alignable column means that bug cannot
be written.
"""
import numpy as np
import pandas as pd
import pytest

from harness.streams import (RESERVED, Stream, StreamInvalid, TimeKind,
                             validate_stream, write_stream)


def _frame(n=5, with_src=True):
    base = 1_786_665_600_000_000_000
    d = pd.DataFrame({
        "recv_ns": base + np.arange(n, dtype="int64") * 100_000_000,
        "mid": np.linspace(100.0, 101.0, n),
    })
    if with_src:
        d["src_ns"] = d["recv_ns"] - 1_500_000_000      # source LEADS receipt
    return d


def test_reserved_columns_are_the_documented_four():
    assert RESERVED == ("recv_ns", "src_ns", "venue", "seq")


def test_write_then_validate_round_trips_the_metadata(tmp_path):
    write_stream(_frame(), str(tmp_path), name="venue_l1", asset="BTC",
                 causal=True, recorder="london_recorder")
    meta = validate_stream(str(tmp_path))
    assert meta["name"] == "venue_l1"
    assert meta["asset"] == "BTC"
    assert meta["causal"] is True
    assert meta["recorder"] == "london_recorder"
    assert meta["schema"] == "1"


def test_value_columns_are_inferred_from_the_schema(tmp_path):
    """No metadata map -- the schema is authoritative and cannot drift."""
    write_stream(_frame(), str(tmp_path), name="s", asset="BTC",
                 causal=True, recorder="r")
    assert validate_stream(str(tmp_path))["values"] == ("mid",)


def test_a_frame_without_recv_ns_is_refused(tmp_path):
    bad = _frame().drop(columns=["recv_ns"])
    with pytest.raises(StreamInvalid, match="recv_ns"):
        write_stream(bad, str(tmp_path), name="s", asset="BTC",
                     causal=True, recorder="r")


def test_recv_ns_must_be_int64(tmp_path):
    bad = _frame()
    bad["recv_ns"] = bad["recv_ns"].astype("float64")
    with pytest.raises(StreamInvalid, match="int64"):
        write_stream(bad, str(tmp_path), name="s", asset="BTC",
                     causal=True, recorder="r")


def test_files_are_partitioned_by_date(tmp_path):
    import os
    write_stream(_frame(), str(tmp_path), name="s", asset="BTC",
                 causal=True, recorder="r")
    assert any(p.startswith("date=") for p in os.listdir(str(tmp_path)))


def test_validate_rejects_a_parquet_with_no_stream_metadata(tmp_path):
    import os
    os.makedirs(str(tmp_path / "date=2026-08-20"), exist_ok=True)
    _frame().to_parquet(str(tmp_path / "date=2026-08-20" / "part.parquet"))
    with pytest.raises(StreamInvalid, match="metadata"):
        validate_stream(str(tmp_path))


def test_causal_absent_from_metadata_is_refused_not_defaulted(tmp_path):
    """Whether a series may be carried forward is not safe to guess."""
    import os, pyarrow.parquet as pq, pyarrow as pa
    os.makedirs(str(tmp_path / "date=2026-08-20"), exist_ok=True)
    tbl = pa.Table.from_pandas(_frame(), preserve_index=False)
    tbl = tbl.replace_schema_metadata({b"stream.schema": b"1",
                                       b"stream.name": b"s"})
    pq.write_table(tbl, str(tmp_path / "date=2026-08-20" / "part.parquet"))
    with pytest.raises(StreamInvalid, match="causal"):
        validate_stream(str(tmp_path))


def test_source_stamp_kind_requires_a_transport_offset():
    with pytest.raises(StreamInvalid, match="transport_offset_ms"):
        Stream(name="x", time_col="oracle_ms", time_kind=TimeKind.SOURCE_STAMP)


def test_receipt_kind_needs_no_offset():
    s = Stream(name="x", time_col="px_first_recv_ns",
               time_kind=TimeKind.RECEIPT, causal=True)
    assert s.time_kind is TimeKind.RECEIPT
    assert s.transport_offset_ms is None


def test_time_kind_has_no_default():
    with pytest.raises(StreamInvalid, match="time_kind"):
        Stream(name="x", time_col="whatever", causal=True)
