"""A shared parquet reader with a soft polars fallback.

Why this exists: the real data files under `data/` (and the
`stream_venue_l1` archive on Z:) were written by `parquet-cpp-arrow 24.0.0`,
but the pyarrow installed alongside this harness is 19.0.0. Reading those
files through pandas/pyarrow 19 raises

    OSError: Repetition level histogram size mismatch

at metadata-parsing time, before any row data is touched. This is pyarrow 19's
footer validation rejecting a `SizeStatistics.repetition_level_histogram`
written by a much newer writer -- it is a reader/writer version mismatch, NOT
file corruption: `polars` (its own, independent Rust Arrow implementation)
reads the same files cleanly.

The fix authorized for this environment is neither "upgrade pyarrow" (shared
environment, other live work depends on the pinned version) nor "rewrite the
data files" (forbidden -- `data/` is read-only from the harness's point of
view). Instead, every parquet read in the harness goes through
`read_parquet()` below: try pandas/pyarrow first, and only if that raises,
retry with polars and convert back to a pandas DataFrame.

The fallback is deliberately soft. `polars` is imported lazily, inside the
`except` block, and only when the primary path has already failed. If polars
is not installed, the original pyarrow error is re-raised (chained, with a
note pointing back here) rather than swallowed. The harness gains no hard
dependency on polars -- it is a diagnostic escape hatch, not a design choice.

When both pandas/pyarrow and polars fail to read, both errors are surfaced
because either can be the real cause. A schema difference across files in a
directory (e.g., a column present in only some files that day) is a common
trigger for failures on directory reads.

`filters` (pyarrow's `[(col, op, value), ...]` predicate-pushdown format) is
only honored natively on the pyarrow path, where it is pushed into the read.
On the polars fallback there is no equivalent pushed-down read here -- the
same filters are applied in pandas *after* the full read instead. This is
simplest and correct for every current caller (each filters on a plain
equality/membership test against a column that's cheap to hold in memory for
one day/one panel at a time); it is not a low-memory streaming path.
"""
import os

import pandas as pd

_OPS = {
    "==": lambda s, v: s == v,
    "=": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "in": lambda s, v: s.isin(v),
    "not in": lambda s, v: ~s.isin(v),
}


def _apply_filters(df, filters):
    """Apply a flat, AND-combined list of (column, op, value) filters."""
    mask = pd.Series(True, index=df.index)
    for col, op, value in filters:
        mask &= _OPS[op](df[col], value)
    return df[mask]


def read_parquet(path, columns=None, filters=None):
    """Read a parquet file or directory-of-parquet-files into a DataFrame.

    Tries pandas' pyarrow-backed reader first (which honors `filters` as
    predicate pushdown). On failure -- any exception raised by the parquet
    layer, including `OSError` and `pyarrow.lib.ArrowInvalid` -- falls back to
    polars, reading the same `columns` and then applying `filters` in pandas
    afterward. See the module docstring for why this fallback exists.
    """
    try:
        return pd.read_parquet(path, columns=columns, filters=filters)
    except Exception as primary_exc:
        try:
            import polars as pl
        except ImportError:
            raise OSError(
                f"pandas/pyarrow failed to read {path!r}: {primary_exc!r}. "
                "This looks like a parquet reader/writer version mismatch "
                "(see harness/io.py module docstring) and polars is not "
                "installed to fall back to."
            ) from primary_exc

        source = path
        if os.path.isdir(path):
            # One glob, one call: a day's worth of tiny files is read in a
            # single scan rather than one file at a time.
            source = os.path.join(path, "**", "*.parquet")

        try:
            df = pl.read_parquet(source, columns=columns).to_pandas()
        except Exception as polars_exc:
            raise OSError(
                f"Failed to read {path!r}: both pandas/pyarrow and polars readers failed. "
                f"Primary error (pyarrow): {primary_exc!r}. "
                f"Fallback error (polars): {polars_exc!r}. "
                f"When reading a directory of parquet files, a schema difference across files "
                f"(e.g., a column present in only some files) is a common cause."
            ) from polars_exc

        if filters:
            df = _apply_filters(df, filters)
        return df
