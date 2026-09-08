"""Paths, constants, the exclusion mask and the shared loaders.

Everything downstream of here reads its data through `load_grid`, which is the only
place that knows about the 2026-07-07 capture defect. The mask is applied by *nulling*
`cl` inside the window rather than by returning a flag for the caller to remember: a
forgotten flag produces a number, a null produces nothing, and the second failure mode
is the one that shows up in a test.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

from . import _vendor  # noqa: F401  (puts the vendored packages on sys.path)

ROOT = Path(__file__).resolve().parents[1]          # the investigation folder
TABLES_DIR = ROOT / "tables"
CACHE = ROOT / "cache"
RUNS = ROOT / "runs"
for _p in (CACHE, RUNS):
    _p.mkdir(parents=True, exist_ok=True)

PARAMS_JSON = TABLES_DIR / "params_v2_1_0.json"

# The 1 s inputs are large and are not vendored. `fvmodel.config` resolves these
# and fails loudly when one is missing; these are only the defaults.
_SRC = Path(os.environ.get(
    "FV_SOURCE_ROOT",
    r"C:\Users\kaima\OneDrive\Documents\GitHub\autoresearch"
    r"\btc_volatility_clock_chainlink"))
DATA_ROOT = _SRC
PERP_PARQUET = _SRC.parent / "btc_volatility_clock" / "output" / "data" / "btcusdt_1s.parquet"

BP = 1e4
SEC_PER_DAY = 86400.0

# ------------------------------------------------------------------ the calendar
CL_T0 = 1776041818          # 2026-04-13 00:56:58 UTC, first Chainlink second
CL_T1 = 1788220799          # 2026-08-31 23:59:59 UTC, last perp bar
PERP_T0 = 1756684800        # 2025-09-01 00:00:00 UTC, first perp bar
HOLDOUT_T0 = 1784505600     # 2026-07-20 00:00:00 UTC, v2.1's six-week holdout
STAMPED_T0 = 1787011200     # 2026-08-18 00:00:00 UTC, first observed Chainlink stamp

# --------------------------------------------------------------- exclusion windows
#
# The Chainlink preliminary note flagged 2026-07-07 "21:27-21:41 UTC" as a capture
# defect. Re-measured at 5-minute resolution on the deseasonalised Chainlink-minus-perp
# residual, the defect is **much wider than that**: the residual sd rises from its
# 0.6 bp baseline to 2.7 bp at 17:20 and does not come back until 22:40, peaking at
# 21 bp. 21:27-21:41 is only the worst quarter of an hour. Inside the window Chainlink
# sits ~55 bp above the perp and jitters +/-30 bp second to second while the perp is
# flat, on single-source seconds - which is a capture, not a feed.
#
# The window below is the conservative envelope 17:00-23:00 UTC: 21,600 seconds, 0.18%
# of the Chainlink overlap.
EXCLUSIONS = [
    (1783443600, 1783465200, "2026-07-07 17:00-23:00 UTC Chainlink capture defect"),
]


def exclusion_mask(ts: np.ndarray) -> np.ndarray:
    """True where the second falls inside a known-bad capture window."""
    ts = np.asarray(ts)
    bad = np.zeros(ts.shape, dtype=bool)
    for a, b, _ in EXCLUSIONS:
        bad |= (ts >= a) & (ts < b)
    return bad


def load_params(path: Path = None) -> dict:
    with open(path or PARAMS_JSON) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------ loaders
def load_grid(t0: int = None, t1: int = None, apply_exclusions: bool = True) -> pl.DataFrame:
    """The aligned 1-second Chainlink/perp/spot frame, with the mask applied.

    Columns: ts, cl, perp, spot, n_trades, quote_volume, gap_flag, lag_ms,
    stamp_observed, cl_src, n_src, excluded.

    `cl` is nulled and `excluded` is set inside every window in EXCLUSIONS. The grid
    itself stays contiguous - dropping rows would break every index-arithmetic loader
    downstream - and `perp`/`spot` are left alone, because the defect is in the
    Chainlink capture and the perp over the same window is fine.
    """
    from clkit.common import load_aligned

    df = load_aligned(t0, t1)
    ts = df["ts"].to_numpy()
    bad = exclusion_mask(ts) if apply_exclusions else np.zeros(ts.size, dtype=bool)
    if bad.any():
        cl = df["cl"].to_numpy().copy()
        cl[bad] = np.nan
        df = df.with_columns(pl.Series("cl", cl))
    return df.with_columns(pl.Series("excluded", bad))


def load_perp(t0: int, t1: int) -> pl.DataFrame:
    """1-second perp bars on a contiguous grid over [t0, t1], prices at the instant.

    Same convention as the aligned frame: the bar stamped `ts` covers [ts, ts+1), so
    its close is the price at instant ts+1 and the row for instant `ts` comes from bar
    `ts-1`.
    """
    n = t1 - t0 + 1
    raw = pl.scan_parquet(PERP_PARQUET).filter(
        (pl.col("ts_ms") >= (t0 - 1) * 1000) & (pl.col("ts_ms") <= (t1 - 1) * 1000)
    ).select(["ts_ms", "close", "n_trades", "quote_volume", "gap_flag"]).collect()
    idx = (raw["ts_ms"].to_numpy() // 1000 + 1 - t0).astype(np.int64)
    keep = (idx >= 0) & (idx < n)
    idx = idx[keep]
    close = np.full(n, np.nan)
    close[idx] = raw["close"].to_numpy()[keep]
    ntr = np.zeros(n, dtype=np.int32)
    ntr[idx] = raw["n_trades"].to_numpy()[keep]
    qv = np.zeros(n, dtype=np.float32)
    qv[idx] = raw["quote_volume"].to_numpy()[keep].astype(np.float32)
    gap = np.full(n, 4, dtype=np.int8)                       # GAP_MISSING
    gap[idx] = raw["gap_flag"].to_numpy()[keep]
    return pl.DataFrame({"ts": np.arange(t0, t1 + 1, dtype=np.int64), "perp": close,
                         "n_trades": ntr, "quote_volume": qv, "gap_flag": gap})


# ------------------------------------------------------------------- small numerics
def ffill(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if not ok.any():
        return x
    idx = np.where(ok, np.arange(x.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[: int(np.argmax(ok))] = np.nan
    return out


def rmse_bp(x: np.ndarray) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2)) * BP)


def utc(ts) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
