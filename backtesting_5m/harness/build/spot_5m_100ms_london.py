"""The London recorder's 100 ms panel onto the book grid, on a **USDT** basis.

READ THIS FIRST. The `spot` column this module writes is BTC/**USDT**,
UNCORRECTED. It is NOT the same quantity as `harness.paths.SPOT` /
`SPOT_ORACLE_WINDOW`, whose `spot` is BTC/**USD** -- the venue mid with the
capture's `usdt_basis` already subtracted. The two columns have the same name,
the same dtype and the same grid, and they differ by roughly **+$43** (sd ~16,
and time-varying by day). Anything that reads this panel as USD is wrong by
about 0.17 of a 300 s sigma, which near the money is ~7 c of probability bias
toward UP -- some four times the taker fee, i.e. large enough to manufacture an
edge out of nothing.

The source, `<Gambling102>/research/btc/backtesting/2026-08-23_backtesting_5m/
data/london/panel_100ms.parquet`, carries no `usdt_basis` column at all, so the
correction cannot be applied here even in principle. That is why this module
emits NO `usdt_basis` column: a NaN column would let `spot_usdt - usdt_basis`
evaluate to NaN silently, and an absent column raises `KeyError` at the first
consumer that assumes the correction exists. The omission is the warning.

A BASIS-LEARNING FAIR BLOCK IS THEREFORE REQUIRED, not optional. Only a model
that estimates the venue-to-oracle gap itself can price off this panel. The
one in `models/normal_qq/fair.py` does exactly that --

    S_t = M_t - B_t,   B_t = ewm(M - C),   C = the Chainlink BTC/USD oracle

-- and reads `ep.spot_usdt` and `ep.chainlink`, never `ep.spot`. On this panel
`spot == spot_usdt` by construction (see below), so that block behaves the same
here as on the USD panels; a block that reads `ep.spot` and treats it as USD
does not, and there is no gate anywhere downstream that would catch it.

WHY THIS PANEL EXISTS: it is the only spot source that spans the whole recorder
window. The three feeds this harness needs cover:

    book panel                2026-08-14 00:05 -> 2026-09-08 13:45
    stream_venue_l1 (Z:)      2026-08-17 04:19 -> 2026-09-09
    RTDS Chainlink oracle     2026-08-14 02:54 -> 2026-08-21 01:59
    london panel_100ms        2026-08-14 02:54:56.9 -> 2026-08-21 01:59:59.9

`paths.SPOT_ORACLE_WINDOW` is built from `stream_venue_l1`, so it starts on
08-17 and the priceable sample is 1,102 markets. The london panel starts where
the oracle starts, so the same three-way overlap becomes 2026-08-14 02:55 ->
2026-08-21 02:00 -- roughly 1,970 markets, an extra 2.9 days. The price of
those days is the currency: this recorder kept the raw venue book and not the
USDT/USD conversion.

`spot` AND `spot_usdt` ARE THE SAME NUMBER HERE, deliberately. The schema is
kept identical to the USD panels (minus `usdt_basis`) so `load_episodes`
consumes it unchanged, and `spot_usdt` means what it says everywhere: the raw
BTC/USDT mid. That `spot` happens to equal it is precisely the fact this
docstring exists to shout about.

THE GRID, MEASURED AND NOT ASSUMED. Every one of the source's 6,015,031 rows
has `ts_ns % 100_000_000 == 0` -- exact 100 ms phase, zero offset -- and every
consecutive gap is exactly 100,000,000 ns, so the source is a dense, gapless
100 ms grid with no jitter at all. Every market open in `data/book_5m_100ms.
parquet` satisfies `open_ts % 300 == 0`, and 300 s is a whole multiple of
100 ms, so the source's bucket boundaries coincide EXACTLY with the market-open
boundaries: `(ts_ns - open_ts*1e9) % 100_000_000 == 0` for every row of every
window. There is no phase to reconcile. `build` asserts this rather than
trusting it.

The float64 bucketing rule is nevertheless carried over verbatim from
`spot_5m_100ms.py` -- buckets are [t, t+100) since the open, the FIRST
observation in a bucket wins, and the bucket index is
`floor((ts - open_ts) * 1000 / 100 + 1e-5)` with EPS in BUCKET units. On this
source the epsilon guard is not decorative: `ts_ns / 1e9` near 1.787e9 has a
float64 ULP of 2.4e-7 s, so a row exactly on a boundary computes as 99.9999 ms
and would floor a bucket low without it -- and here EVERY row is exactly on a
boundary. `build` cross-checks the float path against exact int64 arithmetic on
`ts_ns` and raises if they ever disagree, so the rule is reused AND verified.
Do NOT "fix" this by rounding to the nearest millisecond first; see the note in
`spot_5m_100ms.py`.

FRESHNESS, and the one semantic that is NOT identical to the venue build.

`stream_venue_l1` is a 10 Hz SNAPSHOT sampler: it emits the current L1 state
every ~100 ms whether or not the book changed (measured dt: 0.099/0.100/0.101 s,
mean 0.0999991 s). Its build therefore marks ~99.4 % of buckets present, and
`spot_age_ms` there means "time since the last SNAPSHOT".

The london panel is forward-filled instead: it has a row for every 100 ms
bucket unconditionally, and `spot_n` counts how many real L1 updates fell in
that bucket. 39.4 % of buckets have `spot_n == 0`, i.e. carry the previous
row's numbers unchanged. Emitting those as observations would make
`spot_age_ms` identically zero and destroy the harness's only staleness signal,
so a bucket is emitted here only when `spot_n > 0`. `spot_age_ms` on this panel
therefore means "time since the last L1 CHANGE", which is >= the venue panel's
"time since the last snapshot" and never claims a price is fresher than it is.

This costs nothing where the feed is healthy: a carried bucket's forward-filled
value is bit-identical to the value `shift_to_decision_grid` carries into it, so
`ep.spot` is unchanged and only `ep.spot_age_ms` differs. Zero-runs are short --
median 2 buckets (0.2 s), p99 11, p99.99 42 -- with two exceptions worth naming:
55 runs of >= 5 s, mostly recorder reconnects on the hour, and one genuine
Binance-spot outage of 683 s starting 2026-08-15 05:06:45 UTC (perp and OKX kept
updating through it, so the recorder was alive and the spot feed was not). On
the forward-filled source that outage is 683 s of silently stale price; here it
is 683 s of honest `spot_age_ms`, and `fair.py`'s MAX_MID_AGE_MS gate declines
to sample a basis across it.

WHICH OBSERVATION IN A BUCKET. The source's row for bucket k is the L1 state at
the END of [k, k+100), not at its start. Measured against `stream_venue_l1` over
a dense 600 s window on 2026-08-20 by matching exact (bid, ask, bid_sz, ask_sz)
tuples: the venue sampler sits at phase +25 ms, and the venue first shows a
given london row's state either +24 ms later (25 % of 2,112 matched changes) or
+124 ms later (75 %). That 25/75 split is exactly the fraction of a 100 ms
bucket lying before and after the sampler's +25 ms phase, which pins the source
row at "last update in [k, k+100)". That is the panel convention the harness
already assumes -- a row labelled by its bucket's START, drawn from anywhere
inside it, and made visible only at decision index k+1 -- so it is causal, and
it is on average 75 ms FRESHER within the bucket than the venue build's
"first observation wins". No lookahead: the source row never reflects anything
after its own bucket closed.

THE CLOCK GATE is unchanged and still a DECLARED ASSUMPTION. The london
recorder is a different host from the book panel's vantage, the two streams
carry receipts of different event types (venue L1 vs Polymarket book updates),
and no shared event exists to align them on -- so `offset_s` is configured, not
measured, defaults to 0.0, and is validated for plausibility only by
`require_offset`, which is imported from `spot_5m_100ms.py` so the two builds
cannot drift apart. See that module's THE CLOCK GATE section.
"""
import os

import numpy as np
import pandas as pd

from harness import paths
from harness.build.spot_5m_100ms import require_offset  # one gate, one copy
from harness.io import read_parquet

#: Bid/ask of the Binance BTC/USDT book, and the count of L1 updates that fell
#: in the bucket. `spot_n` is what separates an observation from a carry.
LONDON_COLUMNS = ["ts_ns", "spot_bid", "spot_bid_sz",
                  "spot_ask", "spot_ask_sz", "spot_n"]

NS_PER_BUCKET = paths.BUCKET_MS * 1_000_000      # 100 ms in nanoseconds
NS_PER_S = 1_000_000_000


class GridPhaseError(RuntimeError):
    """The source's 100 ms grid does not line up with the market opens."""


def observations(df):
    """Boolean mask: rows that are a real L1 observation, not a forward fill.

    The source has a row for every 100 ms bucket whether or not the venue said
    anything, so `spot_n == 0` marks a carry. A carry is not an observation:
    emitting it would peg `spot_age_ms` at zero forever and hide the one
    683 s Binance-spot outage this window contains. See the module docstring.

    A row with a non-finite bid or ask is dropped for the same reason the venue
    build drops a row with no basis -- a missing bucket beats a wrong one.
    """
    bid = df["spot_bid"].to_numpy(dtype="float64")
    ask = df["spot_ask"].to_numpy(dtype="float64")
    fresh = df["spot_n"].to_numpy(dtype="int64") > 0
    return (fresh & np.isfinite(bid) & np.isfinite(ask)
            & (bid > 0.0) & (ask > 0.0))


def bucket_london(df, open_ts, offset_s=0.0):
    """One row per 100 ms bucket of the window opening at `open_ts`.

    Same rules as `spot_5m_100ms.bucket_venue_l1`: buckets are [t, t+100)
    since the open, the FIRST row in a bucket wins, and the bucket index is
    floored with EPS in bucket units. `spot` is the mid of `spot_bid`/
    `spot_ask` and is BTC/**USDT**, uncorrected -- there is no basis in this
    source to subtract. `spot_usdt` carries the identical number under the
    name that says what currency it is in.

    Forward-filled rows (`spot_n == 0`) are removed BEFORE bucketing, so a
    bucket is represented by its first real observation and vanishes only when
    it contains none.
    """
    df = df[observations(df)]
    ts = df["ts_ns"].to_numpy(dtype="float64") / NS_PER_S - offset_s
    # EPS is in BUCKET units. See spot_5m_100ms.bucket_venue_l1 -- ts/1e9 near
    # 1.787e9 s has a float64 ULP of 2.4e-7 s, and on THIS source every single
    # row sits exactly on a bucket boundary, so the guard fires constantly.
    EPS = 1e-5
    t_ms = np.floor((ts - open_ts) * 1000.0 / paths.BUCKET_MS + EPS).astype(
        "int64")
    t_ms *= paths.BUCKET_MS

    keep = (t_ms >= 0) & (t_ms < paths.H * 1000)
    bid = df["spot_bid"].to_numpy(dtype="float64")[keep]
    ask = df["spot_ask"].to_numpy(dtype="float64")[keep]
    mid = 0.5 * (bid + ask)
    out = pd.DataFrame({
        "open_ts": open_ts,
        "t_ms": t_ms[keep],
        "spot": mid,            # BTC/USDT -- UNCORRECTED. Not USD.
        "spot_usdt": mid,       # the same number, named for its currency
        "bid_sz": df["spot_bid_sz"].to_numpy(dtype="float64")[keep],
        "ask_sz": df["spot_ask_sz"].to_numpy(dtype="float64")[keep],
        "_ts": ts[keep],
    })
    out = (out.sort_values(["t_ms", "_ts"], kind="stable")
              .drop_duplicates("t_ms", keep="first")
              .drop(columns="_ts")
              .reset_index(drop=True))
    return out


def check_grid_phase(ts_ns, open_ts_values, offset_s=0.0):
    """Verify -- not assume -- that the source grid meets the market opens.

    Returns a dict of what was measured. Raises `GridPhaseError` if the source
    is not on an exact 100 ms phase, or if the market opens are not a whole
    number of buckets apart from it. Only enforced for a zero offset: a
    configured sub-bucket offset deliberately moves the phase, so the check is
    reported and not enforced in that case.
    """
    ts_ns = np.asarray(ts_ns, dtype="int64")
    phase = np.unique(ts_ns % NS_PER_BUCKET)
    opens = np.asarray(sorted(open_ts_values), dtype="int64")
    open_phase = np.unique((opens * NS_PER_S) % NS_PER_BUCKET)
    info = {
        "n_rows": int(ts_ns.shape[0]),
        "ts_phase_ns": [int(p) for p in phase[:5]],
        "n_distinct_ts_phases": int(phase.shape[0]),
        "open_phase_ns": [int(p) for p in open_phase[:5]],
        "n_distinct_open_phases": int(open_phase.shape[0]),
        "offset_s": float(offset_s),
    }
    aligned = (phase.shape[0] == 1 and phase[0] == 0
               and open_phase.shape[0] == 1 and open_phase[0] == 0)
    info["aligned"] = bool(aligned)
    if offset_s == 0.0 and not aligned:
        raise GridPhaseError(
            f"source grid does not align with the market opens: "
            f"ts phases {info['ts_phase_ns']} "
            f"(n={info['n_distinct_ts_phases']}), "
            f"open phases {info['open_phase_ns']} "
            f"(n={info['n_distinct_open_phases']})")
    return info


def verify_float_bucketing(ts_ns, open_ts, offset_s=0.0):
    """The float64 rule must agree with exact int64 arithmetic, every row.

    The bucketing rule is inherited verbatim from the venue build so the two
    panels mean the same thing, and `ts_ns` gives this source an exact integer
    ground truth the venue capture never had. Using it to CHECK the float path
    keeps the shared rule and removes the need to trust it. Returns the number
    of disagreements, which is always 0 because it raises otherwise, or None
    when a non-zero offset leaves no exact integer form to check against.
    """
    if offset_s != 0.0:
        return None
    ts_ns = np.asarray(ts_ns, dtype="int64")
    exact = (ts_ns - int(open_ts) * NS_PER_S) // NS_PER_BUCKET
    ts = ts_ns.astype("float64") / NS_PER_S - offset_s
    approx = np.floor((ts - open_ts) * 1000.0 / paths.BUCKET_MS + 1e-5).astype(
        "int64")
    bad = int(np.count_nonzero(exact != approx))
    if bad:
        raise GridPhaseError(
            f"float64 bucketing disagrees with exact int64 arithmetic on "
            f"{bad} rows of the window opening at {open_ts}")
    return bad


def build(days=None, out_path=None, panel_path=None, source_path=None,
          offset_s=0.0, verify=True):
    """Join the london 100 ms panel onto the book grid, on a USDT basis.

    Emits `spot` (BTC/**USDT**, UNCORRECTED), `spot_usdt` (the identical
    number) and the L1 sizes. There is NO `usdt_basis` column and no USD
    column -- see the module docstring. Only a fair block that learns the
    venue-to-oracle basis for itself may price off this panel.
    """
    out_path = out_path or paths.SPOT_LONDON
    panel_path = panel_path or paths.PANEL
    source_path = source_path or paths.LONDON_PANEL_100MS

    panel = read_parquet(panel_path, columns=["open_ts"]).drop_duplicates()
    panel["day"] = pd.to_datetime(panel["open_ts"], unit="s").dt.strftime(
        "%Y-%m-%d")

    src = read_parquet(source_path, columns=LONDON_COLUMNS)
    src = src.sort_values("ts_ns", kind="stable").reset_index(drop=True)
    ts_ns = src["ts_ns"].to_numpy(dtype="int64")

    if days:
        panel = panel[panel["day"].isin(set(days))]
    # Only windows the source can actually cover. A market whose window runs
    # off either end of the recorder is not a partial market here, it is a
    # market this panel has no rows for at all.
    lo = (ts_ns[0] / NS_PER_S) - offset_s
    hi = (ts_ns[-1] / NS_PER_S) - offset_s
    opens = sorted(int(o) for o in panel["open_ts"].unique()
                   if o + paths.H > lo and o < hi)

    phase = check_grid_phase(ts_ns, opens, offset_s)
    offset = require_offset("london", offset_s)

    frames, per_day = [], {}
    for open_ts in opens:
        a = np.searchsorted(ts_ns, int((open_ts + offset - 1.0) * NS_PER_S))
        b = np.searchsorted(ts_ns,
                            int((open_ts + offset + paths.H + 1.0) * NS_PER_S))
        if b <= a:
            continue
        window = src.iloc[a:b]
        if verify:
            verify_float_bucketing(ts_ns[a:b], open_ts, offset)
        out = bucket_london(window, open_ts, offset)
        if not len(out):
            continue
        frames.append(out)
        day = pd.to_datetime(open_ts, unit="s").strftime("%Y-%m-%d")
        d = per_day.setdefault(day, {"day": day, "markets": 0, "buckets": 0})
        d["markets"] += 1
        d["buckets"] += len(out)

    report = pd.DataFrame(sorted(per_day.values(), key=lambda r: r["day"]))
    if len(report):
        report["coverage"] = (report["buckets"]
                              / (report["markets"] * paths.N_BUCKET)).round(4)
    stem = os.path.splitext(os.path.basename(out_path))[0]
    report.to_csv(os.path.join(paths.RESULTS,
                               f"london_spot_coverage_{stem}.tsv"),
                  sep="\t", index=False)

    spot = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    spot.to_parquet(out_path, index=False)
    print(f"london spot: {len(spot):,} rows, {len(frames)} markets, "
          f"offset {offset:.3f}s, ts phase {phase['ts_phase_ns']} ns, "
          f"aligned={phase['aligned']} -> {out_path}", flush=True)
    return spot


if __name__ == "__main__":
    build()
