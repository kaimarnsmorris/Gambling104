"""Emit conforming stream files, so a recorder never hand-rolls metadata."""
import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from harness.streams.spec import META_PREFIX, SCHEMA_VERSION, StreamInvalid


def write_stream(df, root, *, name, asset, causal, recorder, date_col=None):
    """Write `df` as a conforming stream under `root`, partitioned by date.

    Returns the list of files written. `causal` is mandatory and is written to
    the metadata: whether a series may be carried forward onto a decision grid
    is not something a reader should have to guess.
    """
    if "recv_ns" not in df.columns:
        raise StreamInvalid(f"{name}: a stream must carry recv_ns")
    if df["recv_ns"].dtype != "int64":
        raise StreamInvalid(
            f"{name}: recv_ns must be int64 epoch nanoseconds, got "
            f"{df['recv_ns'].dtype}")

    day = (df[date_col] if date_col else
           pd.to_datetime(df["recv_ns"], unit="ns").dt.strftime("%Y-%m-%d"))

    meta = {
        f"{META_PREFIX}schema": SCHEMA_VERSION,
        f"{META_PREFIX}name": name,
        f"{META_PREFIX}asset": asset,
        f"{META_PREFIX}causal": "true" if causal else "false",
        f"{META_PREFIX}recorder": recorder,
        f"{META_PREFIX}span_start_ns": str(int(df["recv_ns"].min())),
        f"{META_PREFIX}span_end_ns": str(int(df["recv_ns"].max())),
        f"{META_PREFIX}created_utc": pd.Timestamp.utcnow().isoformat(),
    }

    written = []
    for d, part in df.groupby(day, sort=True):
        out_dir = os.path.join(root, f"date={d}")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "part.parquet")
        tbl = pa.Table.from_pandas(part, preserve_index=False)
        tbl = tbl.replace_schema_metadata(
            {k.encode(): str(v).encode() for k, v in meta.items()})
        pq.write_table(tbl, path)
        written.append(path)
    return written
