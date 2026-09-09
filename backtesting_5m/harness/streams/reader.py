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
"""
import numpy as np

from harness.core.episode import shift_to_decision_grid
from harness.paths import BUCKET_MS, H, N_BUCKET

EPS = 1e-5

_TO_S = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def grid_stream(df, open_ts, value_cols, time_col="recv_ns",
                time_unit="ns", causal=True):
    """Grid one stream's rows for one market window.

    Returns {col: array(N_BUCKET)} plus "age_ms" and "has". When `causal`,
    every series routes through `shift_to_decision_grid`, the same function the
    built-in columns use -- a registered stream inherits the lookahead
    guarantee rather than re-earning it.
    """
    ts = df[time_col].to_numpy(dtype="float64") * _TO_S[time_unit]
    k = np.floor((ts - open_ts) * 1000.0 / BUCKET_MS + EPS).astype("int64")
    keep = (k >= 0) & (k < N_BUCKET)
    k = k[keep]

    order = np.argsort(k, kind="stable")            # first-in-bucket wins
    k_sorted = k[order]
    first = np.ones(len(k_sorted), dtype=bool)
    first[1:] = k_sorted[1:] != k_sorted[:-1]
    slots = k_sorted[first]

    out = {}
    present = np.zeros(N_BUCKET, dtype=bool)
    present[slots] = True

    age = None
    for col in value_cols:
        raw = np.full(N_BUCKET, np.nan)
        vals = df[col].to_numpy(dtype="float64")[keep][order][first]
        raw[slots] = vals
        if causal:
            carried, a = shift_to_decision_grid(raw, present)
            out[col] = carried
            age = a if age is None else age
        else:
            out[col] = raw
            age = np.where(present, 0.0, np.inf) if age is None else age

    out["age_ms"] = age
    out["has"] = np.isfinite(out[value_cols[0]]) if value_cols else present
    return out
