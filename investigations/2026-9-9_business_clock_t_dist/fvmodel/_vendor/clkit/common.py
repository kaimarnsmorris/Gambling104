"""Shared paths, loaders and small numerics for the Chainlink settlement study.

Three series live on the same 1-second UTC grid once `build_aligned()` has run:

  * `cl`    Chainlink BTC/USD, on Chainlink's own payload second (null where no capture
            saw a new mark that second),
  * `perp`  Binance BTCUSDT USDT-M perpetual last-trade 1s close, forward-filled,
  * `spot`  Binance BTCUSDT spot last-trade 1s close, forward-filled.

The perp bar stamped `ts` covers [ts, ts+1), so its close is the price at instant
`ts + 1`. Everything here keeps that convention explicit: the perp price *sample time*
is `ts + 1`, and the fitted lag `delta` is measured against that axis. A model that
ignored it would silently absorb a whole second into delta.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from fvmodel.base import DATA_ROOT, PARAMS_JSON, PERP_PARQUET, load_params  # noqa: F401

# `ROOT` used to be this file's own repo root (SRC), which is where the large 1 s
# inputs and the pre-built `aligned_1s.parquet` cache actually live. Vendoring moved
# this file under the investigation folder, so the on-disk source root now comes from
# `fvmodel.base.DATA_ROOT` (the read-only SRC checkout) instead of `__file__`.
ROOT = DATA_ROOT
CACHE = ROOT / "cache"
OUT = ROOT / "output"
IMG = OUT / "img"
TABLES = OUT / "tables"

CL_PARQUET = ROOT / "data" / "chainlink" / "chainlink_1s.parquet"
ALIGNED = CACHE / "aligned_1s.parquet"

GAP_NO_TRADE, GAP_STALE, GAP_MISSING = 1, 2, 4

# the overlap: Chainlink starts 2026-04-13, the perp bar file ends 2026-08-31 23:59:59
T0 = 1776041818          # 2026-04-13 00:56:58 UTC, first Chainlink second
T1 = 1788220799          # 2026-08-31 23:59:59 UTC, the last perp bar


# ------------------------------------------------------------------ alignment build
def build_aligned(force: bool = False) -> Path:
    if ALIGNED.exists() and not force:
        return ALIGNED
    n = T1 - T0 + 1
    ts = np.arange(T0, T1 + 1, dtype=np.int64)

    cl = pl.scan_parquet(CL_PARQUET).filter(
        (pl.col("ts") >= T0) & (pl.col("ts") <= T1)
    ).select(["ts", "cl", "lag_ms", "src", "stamp_kind", "n_src", "chg_age_s"]).collect()
    idx = (cl["ts"].to_numpy() - T0).astype(np.int64)
    cl_v = np.full(n, np.nan)
    cl_v[idx] = cl["cl"].to_numpy()
    lag = np.full(n, np.nan, dtype=np.float32)
    lag[idx] = cl["lag_ms"].to_numpy().astype(np.float32)
    obs = np.zeros(n, dtype=bool)
    obs[idx] = (cl["stamp_kind"].to_numpy() == "observed")
    src_codes = {"archive": 1, "recorder": 2, "tdb_old": 3, "tdb_live": 4}
    src = np.zeros(n, dtype=np.int8)
    src[idx] = np.array([src_codes.get(s, 0) for s in cl["src"].to_list()],
                        dtype=np.int8)
    nsrc = np.zeros(n, dtype=np.int8)
    _ns = cl["n_src"].to_numpy()
    nsrc[idx] = np.nan_to_num(_ns.astype(np.float64), nan=0).astype(np.int8)
    del cl

    # perp: bar ts covers [ts, ts+1); its close is the price at ts+1. We want the price
    # *at* each grid instant, so the bar carrying the price at instant `ts` is `ts-1`.
    perp = pl.scan_parquet(PERP_PARQUET).filter(
        (pl.col("ts_ms") >= (T0 - 1) * 1000) & (pl.col("ts_ms") <= (T1 - 1) * 1000)
    ).select(["ts_ms", "close", "n_trades", "quote_volume", "gap_flag"]).collect()
    p_idx = (perp["ts_ms"].to_numpy() // 1000 + 1 - T0).astype(np.int64)
    keep = (p_idx >= 0) & (p_idx < n)
    p_idx = p_idx[keep]
    perp_v = np.full(n, np.nan)
    perp_v[p_idx] = perp["close"].to_numpy()[keep]
    ntr = np.zeros(n, dtype=np.int32)
    ntr[p_idx] = perp["n_trades"].to_numpy()[keep]
    qv = np.zeros(n, dtype=np.float32)
    qv[p_idx] = perp["quote_volume"].to_numpy()[keep].astype(np.float32)
    gap = np.full(n, GAP_MISSING, dtype=np.int8)
    gap[p_idx] = perp["gap_flag"].to_numpy()[keep]
    del perp

    spot_v = np.full(n, np.nan)
    files = sorted(CACHE.glob("spot_1s_*.parquet"))
    for f in files:
        sp = pl.read_parquet(f)
        s_idx = (sp["ts"].to_numpy() + 1 - T0).astype(np.int64)
        k = (s_idx >= 0) & (s_idx < n)
        spot_v[s_idx[k]] = sp["close"].to_numpy()[k]
    spot_v = ffill(spot_v)

    df = pl.DataFrame({
        "ts": ts, "cl": cl_v, "perp": perp_v, "spot": spot_v,
        "n_trades": ntr, "quote_volume": qv, "gap_flag": gap,
        "lag_ms": lag, "stamp_observed": obs, "cl_src": src, "n_src": nsrc,
    })
    df.write_parquet(ALIGNED, compression="zstd")
    return ALIGNED


def ffill(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if not ok.any():
        return x
    idx = np.where(ok, np.arange(x.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[: int(np.argmax(ok))] = np.nan
    return out


def load_aligned(t0: int = None, t1: int = None) -> pl.DataFrame:
    build_aligned()
    lf = pl.scan_parquet(ALIGNED)
    if t0 is not None:
        lf = lf.filter(pl.col("ts") >= t0)
    if t1 is not None:
        lf = lf.filter(pl.col("ts") <= t1)
    return lf.collect()


# ----------------------------------------------------------------------- numerics
def ewma(x: np.ndarray, tau: float) -> np.ndarray:
    """Clock-time EWMA with time constant `tau` seconds on a complete 1s grid."""
    a = float(np.exp(-1.0 / tau))
    out = np.empty_like(x)
    _ewma_kernel(x, a, out)
    return out


def ewma_trade_time(x: np.ndarray, w: np.ndarray, n_tau: float) -> np.ndarray:
    """EWMA whose decay per second is exp(-w_t / n_tau): a window in trades, not seconds."""
    out = np.empty_like(x)
    _ewma_var_kernel(x, np.exp(-w / n_tau), out)
    return out


try:
    from numba import njit

    @njit(cache=True, fastmath=False)
    def _ewma_kernel(x, a, out):
        acc = x[0]
        for i in range(x.size):
            acc = a * acc + (1.0 - a) * x[i]
            out[i] = acc

    @njit(cache=True, fastmath=False)
    def _ewma_var_kernel(x, a, out):
        acc = x[0]
        for i in range(x.size):
            ai = a[i]
            acc = ai * acc + (1.0 - ai) * x[i]
            out[i] = acc
except ImportError:                                          # pragma: no cover
    def _ewma_kernel(x, a, out):
        acc = x[0]
        for i in range(x.size):
            acc = a * acc + (1.0 - a) * x[i]
            out[i] = acc

    def _ewma_var_kernel(x, a, out):
        acc = x[0]
        for i in range(x.size):
            acc = a[i] * acc + (1.0 - a[i]) * x[i]
            out[i] = acc


def rolling_mean(x: np.ndarray, half: int) -> np.ndarray:
    """Centred rolling mean over [-half, +half] seconds, edges shortened."""
    n = x.size
    c = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)


BP = 1e4


def rmse_bp(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)) * BP)
