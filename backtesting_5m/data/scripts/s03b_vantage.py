"""Put every capture's timestamps on ONE vantage: polydata's.

`ts` in the panel is when we RECEIVED the book, and the two captures did not
receive it at the same moment -- polydata sits closer to the exchange than the
laptop. Left uncorrected, the same book state carries two different times and
the 100 ms buckets do not line up.

The offset is measured, not assumed, and it is NOT a constant: the laptop runs
from roughly 0 ms behind polydata on 2026-08-14 out to 74 ms on 08-28 and back
to ~15 ms in September. So it is estimated per source per day.

How: a book STATE (bid, ask) that occurs exactly once inside a market is an
identifier, so the same state seen by both captures is the same event seen
twice, and the difference of first-observation times is the vantage gap. States
that recur within a market are discarded -- a repeat makes the match a
coincidence rather than an identification.

⚠ WHAT THIS CANNOT FIX. Neither capture records book EVENTS. Both are 10 Hz
state samplers -- the archive's `ts` is the quoter's republish cadence, dt
exactly 0.100 s, not the instant a tick landed. So a corrected `ts` is still
"the first 10 Hz sample that showed this book", and the true arrival lies in
the ~100 ms before it. No amount of offset correction recovers that; it would
need an event-level CLOB capture, which does not exist in this repo.
"""
import os
import sys

import numpy as np
import pandas as pd

import common

REF = "archive"
#: A match further apart than this is two different events colliding on one
#: book state, not one event seen twice.
MAX_GAP_S = 5.0
#: Below this many matched states a day's median is noise; fall back to the
#: era median rather than inventing a per-day number from a handful.
MIN_MATCH = 200


def changepoints(d):
    """First observation of each new book state, keeping only unique states."""
    d = d.sort_values(["market_id", "ts"], kind="stable")
    b, k, m = d.bid.values, d.ask.values, d.market_id.values
    new = np.r_[True, (b[1:] != b[:-1]) | (k[1:] != k[:-1]) | (m[1:] != m[:-1])]
    d = d[new].copy()
    d["n"] = d.groupby(["market_id", "bid", "ask"]).transform("size")
    return d[d.n == 1].drop(columns="n")


def offsets(ref, other, src):
    j = ref.merge(other, on=["market_id", "bid", "ask"], suffixes=("_r", "_o"))
    j["d_ms"] = (j.ts_o - j.ts_r) * 1000.0
    j = j[j.d_ms.abs() < MAX_GAP_S * 1000]
    if not len(j):
        # No offset can be measured at all. Return nothing rather than a
        # column of NaN: s04 refuses to run without an offset, which is the
        # right outcome, and a silent NaN would look like a measurement.
        print(f"  {src}: no matched states -- no offset measurable", flush=True)
        return pd.DataFrame(columns=["day", "n", "offset_ms", "iqr_ms", "src"])
    j["day"] = (j.ts_r // 86400).astype("int64")
    g = j.groupby("day").d_ms.agg(
        n="size", offset_ms="median",
        iqr_ms=lambda s: s.quantile(.75) - s.quantile(.25)).reset_index()
    thin = g.n < MIN_MATCH
    # Days with enough matches set the fallback; if NO day has enough, pool
    # every match rather than taking a median of an empty set, which would
    # make every offset NaN and only surface two scripts later.
    era = (float(g.offset_ms[~thin].median()) if (~thin).any()
           else float(j.d_ms.median()))
    if thin.any():
        print(f"  {src}: {int(thin.sum())} day(s) under {MIN_MATCH} matches, "
              f"using the pooled offset {era:.1f} ms", flush=True)
        g.loc[thin, "offset_ms"] = era
    g["src"] = src
    return g


def main():
    ref = changepoints(pd.read_parquet(common.QUOTES[REF]))
    print(f"reference ({REF}): {len(ref):,} uniquely identifying states",
          flush=True)

    out = []
    for src in common.PRIORITY:
        if src == REF:
            continue
        p = common.QUOTES[src]
        if not os.path.exists(p):
            continue
        g = offsets(ref, changepoints(pd.read_parquet(p)), src)
        out.append(g)
        print(f"  {src}: {len(g)} days, offset median {g.offset_ms.median():.1f} ms "
              f"(range {g.offset_ms.min():.1f}..{g.offset_ms.max():.1f})",
              flush=True)

    t = pd.concat(out, ignore_index=True)
    t["date"] = pd.to_datetime(t.day * 86400, unit="s").dt.strftime("%Y-%m-%d")
    os.makedirs(common.RESULTS, exist_ok=True)
    t.to_parquet(os.path.join(common.DATA, "vantage_offsets.parquet"),
                 index=False)
    t[["src", "date", "n", "offset_ms", "iqr_ms"]].to_csv(
        os.path.join(common.RESULTS, "vantage_offsets.tsv"), sep="\t",
        index=False)
    print(f"\nwrote offsets for {len(t)} (source, day) pairs", flush=True)


if __name__ == "__main__":
    sys.exit(main())
