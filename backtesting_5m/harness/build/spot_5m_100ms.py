"""Venue L1 onto the panel's 100 ms grid.

Two rules carried over from data/scripts/s04_panel.py so that the two grids
mean the same thing: buckets are [t, t+100) since the market open, and the
FIRST observation in a bucket wins.

THE CLOCK GATE. stream_venue_l1 is recorded on a different host from the
panel's reference vantage, with an independent and drifting offset. This build
measures that offset per day and REFUSES TO WRITE if any day lacks a plausible
one. A silent misalignment would be invisible in the output and fatal in the
result, because beta(tau, h) is zero inside tau = 13 s.
"""
import os

import numpy as np
import pandas as pd

from harness import paths

MAX_PLAUSIBLE_OFFSET_S = 1.0     # anything larger is a broken clock
SPOT_COL = "bn_spot_mid"


class ClockGateError(RuntimeError):
    """The venue-to-panel clock offset could not be established."""


def bucket_venue_l1(df, open_ts, offset_s=0.0):
    """One row per 100 ms bucket of the window opening at `open_ts`."""
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
    out = pd.DataFrame({
        "open_ts": open_ts,
        "t_ms": t_ms[keep],
        "spot": df[SPOT_COL].to_numpy()[keep],
        "bid_sz": df["bn_spot_bid_sz"].to_numpy()[keep],
        "ask_sz": df["bn_spot_ask_sz"].to_numpy()[keep],
        "_ts": ts[keep],
    })
    out = (out.sort_values(["t_ms", "_ts"], kind="stable")
              .drop_duplicates("t_ms", keep="first")
              .drop(columns="_ts")
              .reset_index(drop=True))
    return out


def measure_offset(venue_ts, panel_ts):
    """Median venue-minus-panel timestamp difference, in seconds.

    Both captures are ~10 Hz samplers of the same world, so the median
    difference of their aligned observation times is the vantage offset.
    """
    venue_ts = np.asarray(venue_ts, dtype="float64")
    panel_ts = np.asarray(panel_ts, dtype="float64")
    n = min(len(venue_ts), len(panel_ts))
    if n == 0:
        return None
    return float(np.median(venue_ts[:n] - panel_ts[:n]))


def require_offset(day, offset_s):
    """The gate. Raises rather than guessing."""
    if offset_s is None or not np.isfinite(offset_s):
        raise ClockGateError(
            f"{day}: no venue-to-panel clock offset could be measured. "
            f"Refusing to write a misaligned spot panel.")
    if abs(offset_s) > MAX_PLAUSIBLE_OFFSET_S:
        raise ClockGateError(
            f"{day}: implausible clock offset {offset_s:.3f} s "
            f"(limit {MAX_PLAUSIBLE_OFFSET_S} s). This is a broken clock, "
            f"not a vantage difference.")
    return float(offset_s)


def build(days=None, out_path=None, panel_path=None):
    """Join venue L1 onto the panel grid for every day, gated per day."""
    out_path = out_path or paths.SPOT
    panel_path = panel_path or paths.PANEL

    panel = pd.read_parquet(panel_path, columns=["open_ts", "t_ms", "recv_ms"])
    panel["day"] = pd.to_datetime(panel["open_ts"], unit="s").dt.strftime(
        "%Y-%m-%d")

    available = sorted(
        d.split("=")[1] for d in os.listdir(paths.VENUE_L1)
        if d.startswith("date="))
    days = sorted(set(days or available) & set(available))

    frames, offsets = [], []
    for day in days:
        venue = pd.read_parquet(
            os.path.join(paths.VENUE_L1, f"date={day}"),
            columns=["ts", SPOT_COL, "bn_spot_bid_sz", "bn_spot_ask_sz"])
        venue = venue.dropna(subset=["ts", SPOT_COL]).sort_values("ts")

        day_panel = panel[panel["day"] == day]
        offset = require_offset(day, measure_offset(
            venue["ts"].to_numpy(),
            day_panel["recv_ms"].to_numpy(dtype="float64") / 1000.0))
        offsets.append({"day": day, "offset_s": offset, "n": len(venue)})

        for open_ts in sorted(day_panel["open_ts"].unique()):
            window = venue[(venue["ts"] >= open_ts + offset - 1.0)
                           & (venue["ts"] < open_ts + offset + paths.H + 1.0)]
            if len(window):
                frames.append(bucket_venue_l1(window, int(open_ts), offset))

    pd.DataFrame(offsets).to_csv(
        os.path.join(paths.RESULTS, "venue_vantage_offsets.tsv"),
        sep="\t", index=False)

    spot = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    spot.to_parquet(out_path, index=False)
    print(f"spot: {len(spot):,} rows, {len(days)} days -> {out_path}",
          flush=True)
    return spot
