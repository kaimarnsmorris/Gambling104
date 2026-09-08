"""Calibration on the panel's real markets.

Interim, until the execution harness can rank variants by PnL. It scores the export
against real strikes and the realised 60 s Chainlink TWAP, which we compute ourselves
rather than take on trust - and which therefore also cross-checks the venue's own
resolution (see tests/test_scorecard.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from fvmodel import config
from harness_paths import STRIKES

MARKET_LEN = 300
TWAP_LEN = 60
GRID = (-1.0, -0.5, 0.5, 1.0)
EDGES = (0.02, 0.05, 0.10)
BASE_FEE_RATE = 0.07
EPS = 1e-9


def log_loss(p, up) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    up = np.asarray(up, dtype=np.float64)
    return float(-np.mean(up * np.log(p) + (1 - up) * np.log(1 - p)))


def brier(p, up) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(up)) ** 2))


def settlements(t0: int, t1: int) -> pl.DataFrame:
    """market_id, strike, settle, up - the realised 60 s Chainlink TWAP."""
    src = config.source_root()
    cl = pl.read_parquet(src / "data" / "chainlink" / "chainlink_1s.parquet")
    ts = cl["ts"].to_numpy().astype(np.int64)
    px = cl["cl"].to_numpy().astype(np.float64)
    grid = np.full(int(ts.max() - ts.min() + 1), np.nan)
    grid[ts - ts.min()] = px
    # a null second means no NEW mark, not a silent feed: a settlement averages the
    # feed's value each second, which is the step function
    ok = np.isfinite(grid)
    idx = np.where(ok, np.arange(grid.size), 0)
    np.maximum.accumulate(idx, out=idx)
    step = grid[idx]
    step[: int(np.argmax(ok))] = np.nan
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(step))])
    cnt = np.concatenate([[0], np.cumsum(ok[idx].astype(np.int64))])

    k = pl.read_parquet(STRIKES).filter(
        (pl.col("open_ts") >= t0) & (pl.col("open_ts") + MARKET_LEN <= t1)).sort("open_ts")
    end = (k["open_ts"].to_numpy() + MARKET_LEN - ts.min()).astype(np.int64)
    good = (end >= TWAP_LEN) & (end < step.size)
    settle = np.full(end.size, np.nan)
    full = (cnt[end[good] + 1] - cnt[end[good] + 1 - TWAP_LEN]) == TWAP_LEN
    v = (csum[end[good] + 1] - csum[end[good] + 1 - TWAP_LEN]) / TWAP_LEN
    settle[np.where(good)[0][full]] = v[full]
    return k.with_columns(pl.Series("settle", settle)).with_columns(
        (pl.col("settle") > pl.col("strike")).alias("up")).drop_nulls("settle")


def _p_at(z, nu, mu, sg):
    return stats.t.cdf((z + mu) / np.maximum(sg, 1e-12), df=nu)


def proxy_pnl(p_model, p_market, up, edge: float, fee_rate: float = BASE_FEE_RATE,
              taker_rebate: float = 0.0833) -> dict:
    """Buy the side the model likes when the gap clears `edge`, sized by the gap.

    Every trade is a taker at the market price. This is a proxy, not the harness:
    no queue, no latency, no inventory. It exists so a variant can be sanity-checked
    against a real price series before the execution harness lands.
    """
    gap = np.asarray(p_model, dtype=np.float64) - np.asarray(p_market, dtype=np.float64)
    take = np.abs(gap) > edge
    if not take.any():
        return {"n_trades": 0, "total": 0.0, "mean": 0.0, "sd": 0.0, "sharpe": 0.0}
    g = gap[take]
    pm = np.asarray(p_market, dtype=np.float64)[take]
    u = np.asarray(up, dtype=np.float64)[take]
    shares = np.abs(g)                                   # size proportional to the gap
    side = np.sign(g)                                    # +1 buy UP, -1 sell UP
    payoff = side * shares * (u - pm)
    fee = fee_rate * pm * (1.0 - pm) * shares * (1.0 - taker_rebate)
    p = payoff - fee
    sd = float(np.std(p))
    return {"n_trades": int(take.sum()), "total": float(p.sum()),
            "mean": float(np.mean(p)), "sd": sd,
            "sharpe": float(np.mean(p) / sd * np.sqrt(p.size)) if sd > 0 else 0.0,
            "pnl": p, "take": take}


def score_variant(name: str, export_path: Path, ref_path: Path = None) -> dict:
    df = pl.read_parquet(export_path).filter(pl.col("ok"))
    t0 = int(df["open_ts"].min())
    t1 = int(df["open_ts"].max()) + MARKET_LEN
    st = settlements(t0, t1).select(["market_id", "strike", "settle", "up"])
    d = df.join(st, on="market_id")
    if ref_path is not None and Path(ref_path) != Path(export_path):
        # the strike grid goes on the BASELINE's sd, so every variant is asked the
        # same question; a grid at each model's own sd asks the wider model an
        # easier one and the wider model then appears to win
        r = (pl.read_parquet(ref_path).filter(pl.col("ok"))
             .select(["market_id", "t_s", pl.col("sigma").alias("sigma_ref")]))
        d = d.join(r, on=["market_id", "t_s"], how="inner")
    else:
        d = d.with_columns(pl.col("sigma").alias("sigma_ref"))
    excluded = 1.0 - len(d) / max(len(df), 1)

    s = d["s"].to_numpy()
    sg = d["sigma"].to_numpy()
    K = d["strike"].to_numpy()
    nu, mu, sgt = (d[c].to_numpy() for c in ("nu", "mu", "sigma_t"))
    up = d["up"].to_numpy().astype(np.float64)
    tte = MARKET_LEN - d["t_s"].to_numpy()
    z = (s - K) / np.maximum(sg, 1e-300)
    p = _p_at(z, nu, mu, sgt)

    out = {"variant": name, "n_rows": len(d),
           "n_markets": int(d["market_id"].n_unique()),
           "excluded_fraction": float(excluded),
           "log_loss": log_loss(p, up), "brier": brier(p, up)}

    sd_ref = d["sigma_ref"].to_numpy()
    ll, br = [], []
    for c in GRID:
        Kc = s - c * sd_ref
        pc = _p_at((s - Kc) / np.maximum(sg, 1e-300), nu, mu, sgt)
        uc = (d["settle"].to_numpy() > Kc).astype(np.float64)
        ll.append(log_loss(pc, uc))
        br.append(brier(pc, uc))
    out["log_loss_grid"] = float(np.mean(ll))
    out["brier_grid"] = float(np.mean(br))

    # the cell the brief singles out: near the strike, in the last thirty seconds
    near = (np.abs(z) < 1.0) & (tte <= 30)
    out["near30_n"] = int(near.sum())
    out["near30_log_loss"] = log_loss(p[near], up[near]) if near.sum() > 30 else float("nan")
    out["near30_brier"] = brier(p[near], up[near]) if near.sum() > 30 else float("nan")

    # variance calibration by remaining time
    resid = (d["settle"].to_numpy() - s) / np.maximum(d["p_ref"].to_numpy()
                                                      * d["omega"].to_numpy(), 1e-300)
    var_y = (sg / np.maximum(d["p_ref"].to_numpy() * d["omega"].to_numpy(),
                             1e-300)) ** 2
    bins = [(1, 5), (6, 15), (16, 30), (31, 60), (61, 120), (121, 300)]
    lvl, qk = {}, {}
    for lo, hi in bins:
        m = (tte >= lo) & (tte <= hi)
        if m.sum() < 200:
            continue
        r2 = resid[m] ** 2
        v = np.maximum(var_y[m], 1e-30)
        lvl["%d-%d" % (lo, hi)] = float(np.mean(r2) / np.mean(v))
        nz = r2 > 0
        ratio = r2[nz] / v[nz]
        qk["%d-%d" % (lo, hi)] = float(np.mean(ratio - np.log(ratio) - 1.0)
                                       - 1.2704034809047095)
    out["level_by_tte"], out["qlike_excess_by_tte"] = lvl, qk

    # reliability by (tte bucket, moneyness bucket)
    rel = []
    for lo, hi in bins:
        for zlo, zhi in ((-99, -1), (-1, -0.25), (-0.25, 0.25), (0.25, 1), (1, 99)):
            m = (tte >= lo) & (tte <= hi) & (z >= zlo) & (z < zhi)
            if m.sum() < 100:
                continue
            rel.append({"tte": "%d-%d" % (lo, hi), "z": "%g..%g" % (zlo, zhi),
                        "n": int(m.sum()), "p_mean": float(p[m].mean()),
                        "freq_up": float(up[m].mean())})
    out["reliability"] = rel

    # the trading proxy, against the panel's real book mid
    mid = book_mid(d)
    have = np.isfinite(mid)
    out["pnl"] = {}
    for e in EDGES:
        r = proxy_pnl(p[have], mid[have], up[have], edge=e)
        frac = 0.0
        if r["n_trades"] and r["total"]:
            frac = float(r["pnl"][tte[have][r["take"]] <= 30].sum() / r["total"])
        out["pnl"]["edge_%.2f" % e] = dict(
            {k: r[k] for k in ("n_trades", "total", "mean", "sd", "sharpe")},
            frac_last30=frac)
    return out


def book_mid(d: pl.DataFrame) -> np.ndarray:
    """The UP token's book mid at each row's own bucket, from the 5 m panel.

    The panel is 100 ms and carries no row for a bucket nobody observed, so this
    joins on (market_id, the first bucket the row is usable in) and leaves NaN where
    the book was not seen. Nothing is forward-filled: a stale book is worse than a
    missing one, and the proxy simply does not trade where it cannot see a price.

    `t_ms` is computed with the vectorised Polars expression
    `(t_s * 1000 + 100).clip(0, None)` rather than a per-row Python callback over
    `export.build_export.usable_t_ms` - the two must agree everywhere (see
    tests/test_scorecard.py::test_book_mid_t_ms_matches_usable_t_ms), and the
    vectorised form is what keeps this join from dominating the scorecard's runtime
    over ~1.5 M rows.
    """
    from harness_paths import PANEL

    key = d.select(["market_id", "t_s"]).with_columns(
        (pl.col("t_s") * 1000 + 100).clip(0, None).cast(pl.Int64).alias("t_ms"))
    panel = (pl.scan_parquet(PANEL).select(["market_id", "t_ms", "mid"])
             .collect())
    j = key.join(panel, on=["market_id", "t_ms"], how="left")
    return j["mid"].to_numpy().astype(np.float64)
