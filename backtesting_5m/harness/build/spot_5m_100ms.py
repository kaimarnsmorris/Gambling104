"""Venue L1 onto the panel's 100 ms grid, on a **USD** basis.

Two rules carried over from data/scripts/s04_panel.py so that the two grids
mean the same thing: buckets are [t, t+100) since the market open, and the
FIRST observation in a bucket wins.

THE CURRENCY GATE. These markets settle on Chainlink's BTC/**USD** 60 s TWAP,
so the panel's spot line must be BTC/USD too. `bn_spot_mid` is not that: it is
Binance's BTC/**USDT** mid, and USDT is not USD. The venue capture carries the
conversion in `usdt_basis` -- the USDT premium in DOLLARS at that instant
(mean 38.03, sd 16.92, range 0.16..74.6 on 2026-08-20) -- for exactly this
purpose, and this build ignored it until now.

Measured over 81,278 overlapping seconds on 2026-08-20 against the Chainlink
oracle price (`harness.paths.RTDS_BTC`, `px`):

    raw   bn_spot_mid                - px : mean +43.70  median +45.66  sd 16.63
    corr  bn_spot_mid - usdt_basis   - px : mean  +5.67  median  +5.68  sd  8.02
    correlation(raw gap, usdt_basis) = 0.886     sd reduction 51.8 %

Subtracting the basis removes 87 % of the level bias and halves the residual
noise. The uncorrected $43.70 is ~0.17 of a 250 USD 300-second sigma, which
near the money is roughly 7 cents of probability bias toward UP -- about four
times the taker fee, i.e. large enough to manufacture an edge out of nothing.
So `spot` is emitted as `bn_spot_mid - usdt_basis`. The raw mid is kept
alongside it as `spot_usdt`, and the basis actually applied as `usdt_basis`,
so the correction is auditable and reversible row by row.

MISSING OR STALE BASIS: the bucket is dropped, never filled with the
uncorrected mid. A silent $43 error is worse than a missing bucket, and this
panel's whole design principle is that a missing observation beats a wrong
one. Concretely, `basis_ok` rejects an observation whose `usdt_basis` is
absent or non-finite; where the capture also carries `usdt_basis_stale` or
`usdt_basis_ts` (this archive carries NEITHER -- see below), a truthy stale
flag or an age above MAX_BASIS_AGE_S rejects it too. Rejection happens BEFORE
bucketing, so a bucket is represented by its first *usable* observation and
disappears only when none of its observations has a usable basis. It never
falls back to an uncorrected value.

What the archive actually has: `stream_venue_l1` at Z:/parquet has no
`usdt_basis_ts` and no `usdt_basis_stale` column on any of 2026-08-19..24
(schemas were sampled across all six day-directories and are uniform within
each day). The staleness half of the rule is therefore forward-compatible
code that this data never exercises. What it does exercise is the
missing-value half: `usdt_basis` is null on 56 of 861,224 rows on 2026-08-19
and on zero rows of the other five days. Twenty-nine of those 56 already had
no `bn_spot_mid` either and were dropped before the currency gate ever saw
them, so the gate removes 27 rows across the whole 2026-08-19..24 sample and
costs exactly 27 of 4,644,759 buckets (0.0006 %) and zero markets. The basis
is dense in time as well: the implied USDT/USD rate steps roughly every
100 ms, p99 age 0.8 s, max 3.1 s.

Rebuilt and re-measured on the six-day panel against the same oracle
(1,654,313 buckets over 2026-08-19..21, the extent of RTDS coverage):

    raw  spot_usdt - px : mean +43.17  median +47.05  sd 16.36
    corr spot      - px : mean  +4.50  median  +4.07  sd  7.44

2026-08-20 alone lands at +5.66, reproducing the bench number above.

THE CLOCK GATE. stream_venue_l1 is recorded on a different host from the
panel's reference vantage, with an independent and drifting offset. That
offset CANNOT be measured from these two streams: the panel's recv_ms records
receipt of Polymarket order-book updates, while the venue feed's ts records
receipt of Binance/Coinbase/OKX/Bybit updates on a different host -- there is
no shared event between the two series to align on. Cross-correlating the
series would not help either, because the lag between spot moving and the
Polymarket book responding to it is a real market phenomenon (the very thing
this harness exists to study), so it cannot be told apart from a clock
offset by any property of the data.

So the offset used here is a DECLARED ASSUMPTION, passed in by the caller,
not something this module measures. The default of 0.0 rests on this repo's
own data/results/vantage_offsets.tsv, which measured same-book inter-host
drift on this infrastructure at roughly 0-74 ms -- so tens of milliseconds
bounds the plausible range for a vantage difference, and MAX_PLAUSIBLE_OFFSET_S
is set well above that band but far below a broken-clock magnitude. A user
who has independently learned the true venue-to-panel offset should pass it
explicitly via `offset_s`. Short of that, offset sensitivity is meant to be
swept the same way latency and fill optimism already are elsewhere in this
harness, not pinned to a single guessed number.
"""
import os

import numpy as np
import pandas as pd

from harness import paths
from harness.io import read_parquet

MAX_PLAUSIBLE_OFFSET_S = 1.0     # anything larger is a broken clock
MAX_BASIS_AGE_S = 5.0            # only used if the capture dates the basis

#: Binance BTC/USDT mid -- the RAW, uncorrected venue price.
SPOT_COL = "bn_spot_mid"
#: USDT premium in dollars. `SPOT_COL - BASIS_COL` is BTC/USD.
BASIS_COL = "usdt_basis"
#: Optional, and absent from this archive. Honored when present.
BASIS_TS_COL = "usdt_basis_ts"
BASIS_STALE_COL = "usdt_basis_stale"

#: Columns `build` needs from the venue capture. The optional staleness
#: columns are NOT listed: asking polars for a column the files do not have
#: is an error, and `basis_ok` degrades cleanly when they are missing.
VENUE_COLUMNS = ["ts", SPOT_COL, BASIS_COL, "bn_spot_bid_sz", "bn_spot_ask_sz"]


class ClockGateError(RuntimeError):
    """The configured venue-to-panel clock offset is not plausible."""


def basis_ok(df):
    """Boolean mask: rows whose USDT/USD basis is usable.

    An observation with no usable basis has no USD price, so it is not an
    observation this panel can carry. It is dropped -- NOT emitted with the
    uncorrected BTC/USDT mid, which would be wrong by ~$43 and silent about
    it. See the module docstring.

    `usdt_basis` must be present and finite. The two optional columns are
    honored only where the capture writes them (this archive writes neither):
    `usdt_basis_stale` truthy rejects the row, and `usdt_basis_ts` older than
    MAX_BASIS_AGE_S behind `ts` rejects it. A null in either optional column
    is treated as unusable rather than as "fine", so a partially populated
    staleness column can never wave a bad row through.
    """
    ok = np.isfinite(df[BASIS_COL].to_numpy(dtype="float64"))
    if BASIS_STALE_COL in df.columns:
        stale = df[BASIS_STALE_COL].to_numpy(dtype="float64")
        ok &= np.isfinite(stale) & (stale == 0)
    if BASIS_TS_COL in df.columns:
        age = (df["ts"].to_numpy(dtype="float64")
               - df[BASIS_TS_COL].to_numpy(dtype="float64"))
        ok &= np.isfinite(age) & (age <= MAX_BASIS_AGE_S)
    return ok


def bucket_venue_l1(df, open_ts, offset_s=0.0):
    """One row per 100 ms bucket of the window opening at `open_ts`.

    `spot` is BTC/**USD**: the venue's BTC/USDT mid less the USDT basis. The
    raw mid is preserved as `spot_usdt` and the basis applied as
    `usdt_basis`, so `spot_usdt - usdt_basis == spot` holds row by row and
    the correction can be undone by anyone who wants the old series back.

    Observations rejected by `basis_ok` are removed BEFORE bucketing, so a
    bucket is represented by its first *usable* observation and vanishes only
    when none of its observations has a usable basis.
    """
    df = df[basis_ok(df)]
    ts = df["ts"].to_numpy(dtype="float64") - offset_s
    # An epoch second near 1.79e9 has a float64 ULP of 2.4e-7 s, so
    # (ts - open_ts) * 1000 lands up to ~2.4e-4 ms below a whole millisecond:
    # a genuine observation exactly 0.1 s after the open computes as
    # 99.9999 ms and would floor into bucket 0 instead of 100. Since a 10 Hz
    # sampler puts most observations ON those exact multiples, that is the
    # common case, not an edge case.
    #
    # EPS is in BUCKET units and absorbs that error (2.4e-6 buckets) with room
    # to spare, while only mis-bucketing a real observation falling within a
    # microsecond of a boundary. Do NOT "fix" this by rounding to the nearest
    # millisecond first -- that pushes a true 99.6 ms observation into bucket
    # 100, an error 500x larger than the one being corrected.
    EPS = 1e-5
    t_ms = np.floor((ts - open_ts) * 1000.0 / paths.BUCKET_MS + EPS).astype("int64")
    t_ms *= paths.BUCKET_MS

    keep = (t_ms >= 0) & (t_ms < paths.H * 1000)
    spot_usdt = df[SPOT_COL].to_numpy(dtype="float64")[keep]
    basis = df[BASIS_COL].to_numpy(dtype="float64")[keep]
    out = pd.DataFrame({
        "open_ts": open_ts,
        "t_ms": t_ms[keep],
        "spot": spot_usdt - basis,          # BTC/USD -- the settlement basis
        "spot_usdt": spot_usdt,             # raw BTC/USDT, for audit
        "usdt_basis": basis,                # what was actually subtracted
        "bid_sz": df["bn_spot_bid_sz"].to_numpy()[keep],
        "ask_sz": df["bn_spot_ask_sz"].to_numpy()[keep],
        "_ts": ts[keep],
    })
    out = (out.sort_values(["t_ms", "_ts"], kind="stable")
              .drop_duplicates("t_ms", keep="first")
              .drop(columns="_ts")
              .reset_index(drop=True))
    return out


def require_offset(day, offset_s):
    """The gate. Validates a CONFIGURED offset; raises rather than guessing.

    stream_venue_l1 and the panel carry receipts of different event streams
    (venue spot ticks vs. Polymarket book updates), so there is no offset to
    measure here -- only one to declare. This checks that the declared value
    is present, finite, and within a plausible vantage-difference magnitude;
    it does not and cannot verify that the value is correct.
    """
    if offset_s is None or not np.isfinite(offset_s):
        raise ClockGateError(
            f"{day}: no venue-to-panel clock offset was configured. "
            f"Refusing to write a misaligned spot panel.")
    if abs(offset_s) > MAX_PLAUSIBLE_OFFSET_S:
        raise ClockGateError(
            f"{day}: implausible clock offset {offset_s:.3f} s "
            f"(limit {MAX_PLAUSIBLE_OFFSET_S} s). This is a broken clock, "
            f"not a vantage difference.")
    return float(offset_s)


def build(days=None, out_path=None, panel_path=None, offset_s=0.0):
    """Join venue L1 onto the panel grid for every day, on a USD basis.

    Uses a configured (not measured) clock offset, gated per day by
    `require_offset`, and a per-row USDT/USD conversion gated by `basis_ok`.
    Emits `spot` (BTC/USD), `spot_usdt` (the raw BTC/USDT mid) and
    `usdt_basis` (what was subtracted). Rows the currency gate rejected are
    counted per day as `n_no_basis` in the offsets report.
    """
    out_path = out_path or paths.SPOT
    panel_path = panel_path or paths.PANEL

    panel = read_parquet(panel_path, columns=["open_ts", "t_ms", "recv_ms"])
    panel["day"] = pd.to_datetime(panel["open_ts"], unit="s").dt.strftime(
        "%Y-%m-%d")

    available = sorted(
        d.split("=")[1] for d in os.listdir(paths.VENUE_L1)
        if d.startswith("date="))
    days = sorted(set(days or available) & set(available))

    frames, offsets = [], []
    for day in days:
        venue = read_parquet(
            os.path.join(paths.VENUE_L1, f"date={day}"),
            columns=VENUE_COLUMNS)
        venue = venue.dropna(subset=["ts", SPOT_COL]).sort_values("ts")
        # Applied once per day rather than once per 5-minute window: the mask
        # is row-local, so hoisting it out of the loop is a pure speed-up and
        # gives an honest per-day count of what the currency gate cost.
        n_priced = len(venue)
        venue = venue[basis_ok(venue)]

        day_panel = panel[panel["day"] == day]
        offset = require_offset(day, offset_s)
        offsets.append({
            "day": day, "offset_s": offset, "n": len(venue),
            "n_no_basis": n_priced - len(venue),
            "source": "configured",
        })

        for open_ts in sorted(day_panel["open_ts"].unique()):
            window = venue[(venue["ts"] >= open_ts + offset - 1.0)
                           & (venue["ts"] < open_ts + offset + paths.H + 1.0)]
            if len(window):
                frames.append(bucket_venue_l1(window, int(open_ts), offset))

    # The offsets report is named after the panel it describes, so building
    # an alternate panel (e.g. the USD one) cannot overwrite the report for
    # the default panel that other readers are still using.
    stem = os.path.splitext(os.path.basename(out_path))[0]
    suffix = "" if stem == "spot_5m_100ms" else f"_{stem}"
    pd.DataFrame(offsets).to_csv(
        os.path.join(paths.RESULTS, f"venue_vantage_offsets{suffix}.tsv"),
        sep="\t", index=False)

    spot = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    spot.to_parquet(out_path, index=False)
    print(f"spot: {len(spot):,} rows, {len(days)} days -> {out_path}",
          flush=True)
    return spot
