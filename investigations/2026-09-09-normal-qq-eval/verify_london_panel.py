"""Verification for `paths.SPOT_LONDON`: coverage, currency, and the overlap.

Three questions, none of them assumed:

  1. What does the panel cover -- rows, markets and bucket coverage per day?
  2. How far is its `spot` from the settlement oracle? It is BTC/USDT and
     uncorrected, so the answer must be ~+$43 with sd ~16. A number near zero
     would mean the panel is not what its docstring says it is.
  3. On the three days that overlap `paths.SPOT_ORACLE_WINDOW`, do the two
     panels agree once the USD correction is undone -- i.e. is
     `london.spot ~= oracle_window.spot + oracle_window.usdt_basis`?

Prints plain ASCII only; the console here is cp1252.
"""
import os

import numpy as np
import pandas as pd

from harness import paths
from harness.build.episodes import load_chainlink
from harness.io import read_parquet

OVERLAP_DAYS = ("2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20",
                "2026-08-21")


def day_of(open_ts):
    return pd.to_datetime(open_ts, unit="s").dt.strftime("%Y-%m-%d")


def main():
    lon = read_parquet(paths.SPOT_LONDON)
    lon["day"] = day_of(lon["open_ts"])
    print(f"columns: {list(lon.columns)}")
    print(f"rows: {len(lon):,}   markets: {lon['open_ts'].nunique():,}")
    assert "usdt_basis" not in lon.columns, "this panel must carry no basis"
    same = np.allclose(lon["spot"], lon["spot_usdt"])
    print(f"spot == spot_usdt on every row: {same}")

    # --- 1. coverage per day ------------------------------------------------
    cov = (lon.groupby("day")
              .agg(rows=("t_ms", "size"), markets=("open_ts", "nunique")))
    cov["buckets_per_market"] = (cov["rows"] / cov["markets"]).round(1)
    cov["coverage"] = (cov["rows"] / (cov["markets"] * paths.N_BUCKET)).round(4)
    print("\n-- per-day coverage --")
    print(cov.to_string())
    print(f"TOTAL  rows={len(lon):,}  markets={lon['open_ts'].nunique()}  "
          f"coverage="
          f"{len(lon) / (lon['open_ts'].nunique() * paths.N_BUCKET):.4f}")

    # --- 2. the gap to the settlement oracle --------------------------------
    # Bucket the oracle by RECEIPT onto the same 100 ms grid the panel uses,
    # then take the last oracle print at or before each spot observation. The
    # spot line is BTC/USDT with no correction, so this gap IS the USDT basis
    # plus whatever staleness the 1 Hz oracle contributes.
    recv_ms, px = load_chainlink(paths.RTDS_BTC)
    ts_ms = lon["open_ts"].to_numpy() * 1000.0 + lon["t_ms"].to_numpy()
    idx = np.searchsorted(recv_ms, ts_ms, side="right") - 1
    ok = idx >= 0
    gap = np.full(len(lon), np.nan)
    gap[ok] = lon["spot"].to_numpy()[ok] - px[idx[ok]]
    lon["gap"] = gap
    g = lon.dropna(subset=["gap"])
    print(f"\n-- spot (BTC/USDT, uncorrected) minus Chainlink px --")
    print(f"n={len(g):,}  mean={g['gap'].mean():+.2f}  "
          f"median={g['gap'].median():+.2f}  sd={g['gap'].std():.2f}")
    byday = g.groupby("day")["gap"].agg(["size", "mean", "median", "std"])
    print(byday.round(2).to_string())

    # --- 3. the three-day cross-check against the USD panel -----------------
    ow = read_parquet(paths.SPOT_ORACLE_WINDOW,
                      columns=["open_ts", "t_ms", "spot", "spot_usdt",
                               "usdt_basis"])
    ow["day"] = day_of(ow["open_ts"])
    shared = sorted(set(lon["day"]) & set(ow["day"]))
    print(f"\n-- overlap with SPOT_ORACLE_WINDOW: days {shared} --")
    j = lon[["open_ts", "t_ms", "spot", "day"]].merge(
        ow[["open_ts", "t_ms", "spot", "spot_usdt", "usdt_basis"]],
        on=["open_ts", "t_ms"], suffixes=("_lon", "_ow"))
    print(f"buckets present in BOTH panels: {len(j):,}")
    # The USD panel's own raw mid is the like-for-like comparator; adding the
    # basis back to its corrected `spot` must reproduce it exactly, and does.
    j["d_usdt"] = j["spot_lon"] - j["spot_usdt"]
    j["d_uncorrected"] = j["spot_lon"] - (j["spot_ow"] + j["usdt_basis"])
    for col in ("d_usdt", "d_uncorrected"):
        s = j[col]
        print(f"{col:16s} mean={s.mean():+.4f} median={s.median():+.4f} "
              f"sd={s.std():.4f} p99|.|={np.percentile(np.abs(s), 99):.4f} "
              f"max|.|={np.abs(s).max():.4f}")
    print("\nby day, london.spot - (oracle_window.spot + usdt_basis):")
    print(j.assign(day=day_of(j["open_ts"]))
           .groupby("day")["d_uncorrected"]
           .agg(["size", "mean", "median", "std"]).round(4).to_string())

    # --- 4. what the freshness rule costs -----------------------------------
    # This panel marks a bucket present only where the venue actually changed
    # (`spot_n > 0`), so `spot_age_ms` measures time since the last CHANGE.
    # The gate that cares is fair.py's MAX_MID_AGE_MS = 1000 ms.
    for name, df in (("london", lon), ("oracle_window", ow)):
        n_mkt = df["open_ts"].nunique()
        have = np.zeros((n_mkt, paths.N_BUCKET), dtype=bool)
        codes = pd.factorize(df["open_ts"], sort=True)[0]
        have[codes, (df["t_ms"].to_numpy() // paths.BUCKET_MS)] = True
        # age at decision index i, in ms, from the last present bucket <= i-1
        src = np.where(have, np.arange(paths.N_BUCKET)[None, :], -1)
        src = np.maximum.accumulate(src, axis=1)
        shifted = np.full_like(src, -1)
        shifted[:, 1:] = src[:, :-1]
        seen = shifted >= 0
        age = np.full(src.shape, np.inf)
        age[seen] = ((np.arange(paths.N_BUCKET)[None, :] - shifted - 1)
                     * paths.BUCKET_MS)[seen]
        print(f"\n{name}: bucket coverage {have.mean():.4f}, "
              f"spot_age_ms==0 {np.mean(age == 0):.4f}, "
              f">1000ms {np.mean(age > 1000):.4f}, "
              f"never seen {np.mean(~seen):.4f}")


if __name__ == "__main__":
    main()
