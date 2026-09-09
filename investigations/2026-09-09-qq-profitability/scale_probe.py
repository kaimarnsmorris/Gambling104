"""Is the distribution too narrow, and by how much?

The reliability table says the model is overconfident everywhere: it predicts
0.024 where the truth is 0.138, and 0.989 where the truth is 0.937, while the
book is within ~0.02 throughout. That is the signature of a distribution that
is too NARROW -- too little sigma, or a link with tails too thin for 5 minutes
of BTC.

This does not need a backtest to test. `fair_p = link(z)` and `z` is on every
tick, so widening sigma by a factor k is exactly `link(z / k)`. Sweeping k
offline over the recorded ticks costs seconds instead of an arm each, and
separates the two candidate causes:

  * if some k makes the model match the book, the shape is right and only the
    SCALE is wrong -- a one-parameter fix;
  * if no k does, and the residual error is still in the extreme buckets, the
    normal link's tails are the problem and scaling cannot rescue it.

Reported against the book, not against zero: the book is the benchmark that
matters, since we only trade when we disagree with it.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, "backtesting_5m"))
from harness.io import read_parquet   # noqa: E402

MODEL = os.path.join(HERE, os.pardir, os.pardir, "models", "normal_qq")


def _link():
    spec = importlib.util.spec_from_file_location(
        "qq_link", os.path.join(MODEL, "link.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return np.vectorize(m.link)


def main():
    with open(os.path.join(HERE, "last_run.txt")) as fh:
        run_dir = fh.read().strip()
    t = read_parquet(os.path.join(run_dir, "ticks.parquet"),
                     columns=["market_id", "t_ms", "z", "fair_p", "mid"])
    m = read_parquet(os.path.join(run_dir, "markets.parquet"),
                     columns=["market_id", "winner_up"]).drop_duplicates()
    t = t.merge(m, on="market_id", how="left")
    t = t[np.isfinite(t["z"]) & np.isfinite(t["mid"]) & t["winner_up"].notna()]
    # One row per (market, t_ms): seeds replay the same signal.
    t = t.drop_duplicates(subset=["market_id", "t_ms"])
    z = t["z"].to_numpy(dtype="float64")
    y = t["winner_up"].astype(float).to_numpy()
    mid = t["mid"].to_numpy(dtype="float64")
    print(f"{len(t):,} distinct quoting ticks over "
          f"{t['market_id'].nunique():,} markets")

    link = _link()
    b_book = np.mean((mid - y) ** 2)
    print(f"\nBrier(book)  = {b_book:.5f}")
    print(f"Brier(0.5)   = {np.mean((0.5 - y) ** 2):.5f}\n")

    rows = []
    for k in (1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 3.0, 4.0):
        p = link(z / k)
        rows.append({
            "k": k,
            "brier": np.mean((p - y) ** 2),
            "vs_book": np.mean((p - y) ** 2) - b_book,
            # how far the extreme buckets still miss, the error the
            # reliability table showed
            "err_lo": _bucket_err(p, y, 0.0, 0.1),
            "err_hi": _bucket_err(p, y, 0.9, 1.0),
            "frac_extreme": float(((p < 0.1) | (p > 0.9)).mean()),
            "mean_abs_edge_c": 100 * float(np.mean(np.abs(p - mid))),
        })
    df = pd.DataFrame(rows).set_index("k")
    print("--- widening sigma by k: link(z / k) ---")
    print(df.to_string(float_format=lambda v: f"{v:9.4f}"))

    best = df["brier"].idxmin()
    print(f"\nbest k = {best}  (Brier {df.loc[best, 'brier']:.5f} against "
          f"book {b_book:.5f}, gap {df.loc[best, 'vs_book']:+.5f})")

    p = link(z / best)
    print(f"\n--- reliability at k={best} ---")
    out = pd.DataFrame({"p": p, "y": y, "mid": mid})
    out["bucket"] = pd.cut(out["p"], np.linspace(0, 1, 11))
    g = out.groupby("bucket", observed=True).agg(
        n=("p", "size"), predicted=("p", "mean"), realised=("y", "mean"),
        book=("mid", "mean"))
    g["model_err"] = g["predicted"] - g["realised"]
    g["book_err"] = g["book"] - g["realised"]
    print(g.to_string(float_format=lambda v: f"{v:8.4f}"))


def _bucket_err(p, y, lo, hi):
    m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo)
    return float(p[m].mean() - y[m].mean()) if m.any() else np.nan


if __name__ == "__main__":
    main()
