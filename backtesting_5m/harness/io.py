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

THE THIRD ATTEMPT, for directories whose files disagree about their columns.
`polars` takes the first file it globs as the schema for the whole scan and
raises on any file that has a column that one did not -- "extra column in
file outside of expected schema: okx_bid". `stream_venue_l1` does exactly
this on 2026-08-18: 12,961 of its 13,438 files carry the okx and bybit
blocks and 477 do not, and 2026-08-17 has none of them at all. That is an
inconsistency in the source archive, not in the caller's request, and it
should not cost a whole day of data when every column the caller actually
asked for is present in every file.

So a directory read that polars refuses is retried, in two steps.

First, and only when the caller named its `columns`: `scan_parquet` with
`extra_columns="ignore"`, then project. Columns the caller did not ask for
are exactly what "extra" means here, so ignoring them is not a loosening of
anything -- and a file genuinely MISSING a requested column still raises
`ColumnNotFoundError` rather than coming back as silent nulls, which is the
behaviour worth keeping. This is one more globbed scan and no per-file work.

Second, if that fails too (or if `columns` was None, where "extra" has no
meaning), the files are grouped by their exact column signature, each group
is read on its own, and the groups are concatenated -- vertically when the
caller named its columns, diagonally otherwise, so a column absent from one
group comes back null on those rows rather than disappearing from the
result. This step costs one schema read per file -- on the 13,438 files of
2026-08-18, about 100 s against the 25 s of the read itself -- which is why
it is last.

`filters` (pyarrow's `[(col, op, value), ...]` predicate-pushdown format) is
only honored natively on the pyarrow path, where it is pushed into the read.
On the polars fallback there is no equivalent pushed-down read here -- the
same filters are applied in pandas *after* the full read instead. This is
simplest and correct for every current caller (each filters on a plain
equality/membership test against a column that's cheap to hold in memory for
one day/one panel at a time); it is not a low-memory streaming path.
"""
import glob
import os
from collections import OrderedDict

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


def _read_dir_ignoring_extra_columns(path, columns):
    """Project a heterogeneous directory onto the columns actually asked for.

    Cheap: one more globbed scan, no per-file schema read. Only meaningful
    with an explicit `columns` list, since without one there is nothing for
    "extra" to be extra to. A file missing a REQUESTED column still raises.
    """
    import polars as pl

    if not columns:
        raise ValueError("needs an explicit column list")
    source = os.path.join(path, "**", "*.parquet")
    return (pl.scan_parquet(source, extra_columns="ignore")
              .select(columns).collect().to_pandas())


def _read_dir_by_schema_group(path, columns):
    """Read a directory whose files disagree about their columns.

    Groups the files by their exact column signature, reads each group in one
    globbed call, and concatenates. Returns a pandas DataFrame. See the module
    docstring for why this exists and when it runs.
    """
    import polars as pl

    files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"),
                             recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files under {path!r}")

    groups = OrderedDict()
    for f in files:
        key = tuple(pl.read_parquet_schema(f).keys())
        groups.setdefault(key, []).append(f)

    if columns is not None:
        missing = {c for c in columns for k in groups if c not in k}
        if missing:
            raise OSError(
                f"{path!r}: {sorted(missing)} absent from at least one "
                f"file's schema across {len(groups)} distinct schemas.")

    frames = [pl.read_parquet(g, columns=columns) for g in groups.values()]
    if len(frames) == 1:
        return frames[0].to_pandas()
    how = "vertical" if columns is not None else "diagonal_relaxed"
    return pl.concat(frames, how=how).to_pandas()


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
            if os.path.isdir(path):
                try:
                    df = _read_dir_ignoring_extra_columns(path, columns)
                except Exception:
                    df = None
                if df is None:
                    try:
                        df = _read_dir_by_schema_group(path, columns)
                    except Exception:
                        raise OSError(
                            f"Failed to read {path!r}: pandas/pyarrow, polars, "
                            f"the extra-column retry and the per-schema-group "
                            f"retry all failed. "
                            f"Primary error (pyarrow): {primary_exc!r}. "
                            f"Fallback error (polars): {polars_exc!r}."
                        ) from polars_exc
                if filters:
                    df = _apply_filters(df, filters)
                return df
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
