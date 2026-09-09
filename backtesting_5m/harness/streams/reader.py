"""Put one stream's observations onto the decision grid.

Two rules carried from the panel build so every grid means the same thing:
buckets are [t, t+100) since the market open, and the FIRST observation in a
bucket wins.

The epsilon is in BUCKET units. An epoch second near 1.79e9 has a float64 ULP
of 2.4e-7 s, so (ts - open_ts) * 1000 lands up to ~2.4e-4 ms below a whole
millisecond: an observation exactly 0.1 s after the open computes as 99.9999 ms
and would floor into bucket 0 instead of 100. A 10 Hz sampler puts most
observations ON those exact multiples, so that is the common case. Do NOT
"fix" it by rounding to the nearest millisecond -- that pushes a true 99.6 ms
observation into bucket 100, an error 500x larger.

ONE BUCKETING PER RECEIPT, NOT PER STREAM. A recorder that stamps each field
with its own arrival -- `rtds_btc` does -- gets one pass of the bucketing
below per distinct receipt column, and the values it governs ride on that
pass. Bucketing every field at one field's receipt is the lookahead
`harness.streams.spec` exists to make unwritable; see its module docstring.
"""
import numpy as np

from harness.core.episode import shift_to_decision_grid
from harness.paths import BUCKET_MS, H, N_BUCKET

EPS = 1e-5

_TO_S = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def _bucket(df, open_ts, time_col, time_unit):
    """Bucket one frame on ONE time column.

    Returns (slots, rows, present): the bucket indices that were observed, the
    frame rows that won them in the same order, and the N_BUCKET presence
    mask. Every value column governed by `time_col` reads its own numbers
    through this one answer.
    """
    ts = df[time_col].to_numpy(dtype="float64") * _TO_S[time_unit]

    # Bounded in FLOAT space, then floored -- identical buckets, because
    # floor(x) < N is x < N for integer N and floor(x) >= 0 is x >= 0, but it
    # survives two things the int64 cast does not. A per-field receipt is null
    # wherever that field did not update in a row (`rtds_btc`: 137 of 568,261
    # rows on px, 102 on twap60), and NaN.astype("int64") is undefined; and a
    # value far outside the window can wrap the cast back INTO a valid bucket.
    # NaN fails both comparisons, so such rows are dropped, which is what a
    # missing receipt means.
    scaled = (ts - open_ts) * 1000.0 / BUCKET_MS + EPS
    keep = (scaled >= 0) & (scaled < N_BUCKET)
    k = np.floor(scaled[keep]).astype("int64")

    ts_kept = ts[keep]
    order = np.lexsort((ts_kept, k))     # primary k, secondary ts
    k_sorted = k[order]
    first = np.ones(len(k_sorted), dtype=bool)
    first[1:] = k_sorted[1:] != k_sorted[:-1]
    slots = k_sorted[first]
    rows = np.flatnonzero(keep)[order][first]

    present = np.zeros(N_BUCKET, dtype=bool)
    present[slots] = True
    return slots, rows, present


def _age_of(present, causal):
    """`age_ms` for one bucketing, from WHICH buckets were observed alone.

    No value column enters this: age is a property of arrival, not of what
    arrived, which is why it does not depend on column order and why a
    receipt group with no value columns of its own still has one.
    """
    if not causal:
        return np.where(present, 0.0, np.inf)
    _, age = shift_to_decision_grid(np.zeros(N_BUCKET), present)
    return age


def grid_stream(df, open_ts, value_cols, time_col="recv_ns",
                time_unit="ns", causal=True, value_time_cols=()):
    """Grid one stream's rows for one market window.

    Returns {col: array(N_BUCKET)} plus "age_ms" and "has". When `causal`,
    every series routes through `shift_to_decision_grid`, the same function the
    built-in columns use -- a registered stream inherits the lookahead
    guarantee rather than re-earning it.

    `value_time_cols` is a tuple of (value_col, receipt_col) pairs: a column
    named there is bucketed on ITS OWN receipt, everything else on `time_col`.
    The columns are grouped by receipt and the bucketing above runs once per
    group, so the per-column case costs one extra pass per distinct receipt
    and changes nothing about how a bucket is chosen.

    "age_ms" AND "has" DESCRIBE THE STREAM-LEVEL `time_col` ONLY -- the group
    the stream declares itself on. They do not cover a column bucketed on its
    own receipt, which by construction arrived at a different time; a reader
    will otherwise assume they do. A consumer that needs the staleness of one
    such column should register it as its own stream.
    """
    receipt_of = dict(value_time_cols)

    # Grouped by receipt column, insertion-ordered, so the result is
    # deterministic. The stream-level `time_col` group always exists even
    # when no value column rides on it: `age_ms`/`has` come from it.
    groups = {time_col: []}
    for col in value_cols:
        groups.setdefault(receipt_of.get(col, time_col), []).append(col)

    out = {}
    for group_time_col, cols in groups.items():
        slots, rows, present = _bucket(df, open_ts, group_time_col, time_unit)
        if group_time_col == time_col:
            out["age_ms"] = _age_of(present, causal)
        for col in cols:
            raw = np.full(N_BUCKET, np.nan)
            raw[slots] = df[col].to_numpy(dtype="float64")[rows]
            out[col] = (shift_to_decision_grid(raw, present)[0] if causal
                        else raw)

    out["has"] = np.isfinite(out["age_ms"])
    return out
