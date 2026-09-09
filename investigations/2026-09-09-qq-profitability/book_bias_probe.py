"""Is the BOOK biased in the tails, and is the bias tradeable?

The sweep says widening sigma makes PnL worse, monotonically, even though it
makes the forecast better. That is only paradoxical if you assume the money
comes from forecasting the level -- and `fair_probe.py` says it cannot:
rmse(s) is 62.77 against 62.68 for the raw venue mid and 63.40 for carrying
the oracle. The model has NO location edge, so recalibrating its width only
changes which coin flips it takes.

What survives that is a SHAPE edge, and the reliability tables show one in
the book itself:

    book says 0.088 where the truth is 0.073   (longshots overpriced)
    book says 0.948 where the truth is 0.969   (favourites underpriced)

the classic favourite-longshot bias. It needs no forecast of the level: it
says sell cheap, buy dear, and wait.

This estimates it honestly, which the reliability table does not. Those rows
average 350k-720k QUOTING TICKS drawn from ~1,100 markets, and ticks inside
one market are the same bet observed 3,000 times -- so their apparent
precision is a fiction. Here the unit is the MARKET, and the confidence
interval is a day-blocked bootstrap, because markets within a day share the
same regime.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, os.pardir, "backtesting_5m"))
from harness.io import read_parquet      # noqa: E402

RNG = np.random.default_rng(0)
N_BOOT = 2000


def day_blocked_ci(df, value, day="day", n=N_BOOT, lo=2.5, hi=97.5):
    """Percentile CI resampling whole DAYS with replacement."""
    days = df[day].unique()
    by_day = {d: df.loc[df[day] == d, value].to_numpy() for d in days}
    means = np.empty(n)
    for b in range(n):
        pick = RNG.choice(days, size=len(days), replace=True)
        means[b] = np.concatenate([by_day[d] for d in pick]).mean()
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def gates(t, lo, hi, label):
    """The two gates a CI cannot supply: concentration, and stability.

    A band at p<0.05 pays out on rare wins, so its mean is a handful of
    markets. `delete-top-10` asks whether the edge is the distribution or
    the ten luckiest bets; the period split asks whether it is a regime.
    """
    sel = t[(t["mid"] >= lo) & (t["mid"] < hi)]
    per = sel.groupby(["day", "market_id"]).agg(
        book=("mid", "mean"), y=("y", "first"), ticks=("mid", "size"))
    per = per[per["ticks"] >= 100].reset_index()
    # buying the band: pay `book`, receive `y`
    per["edge_c"] = 100.0 * (per["y"] - per["book"])
    full = per["edge_c"].mean()
    trimmed = per.sort_values("edge_c").iloc[:-10]["edge_c"].mean()
    print(f"\n--- gates: BUYING {label} ({len(per)} markets) ---")
    print(f"  full sample        : {full:+7.3f} c/share")
    print(f"  delete best 10     : {trimmed:+7.3f} c/share")
    print(f"  winners            : {int((per['y'] > 0.5).sum())} of {len(per)}"
          f"  ({(per['y'] > 0.5).mean():.1%}, book implies "
          f"{per['book'].mean():.1%})")
    print("  by day:")
    for d, g in per.groupby("day"):
        print(f"    {d}  n={len(g):4d}  {g['edge_c'].mean():+7.3f} c  "
              f"winners={int((g['y'] > 0.5).sum())}")
    signs = per.groupby("day")["edge_c"].mean()
    print(f"  sign survives every day: "
          f"{'YES' if (signs > 0).all() or (signs < 0).all() else 'NO'}")


def main():
    with open(os.path.join(HERE, "last_run.txt")) as fh:
        run_dir = fh.read().strip()
    t = read_parquet(os.path.join(run_dir, "ticks.parquet"),
                     columns=["market_id", "t_ms", "mid", "fair_p"])
    t = t.drop_duplicates(subset=["market_id", "t_ms"])
    m = read_parquet(os.path.join(run_dir, "markets.parquet"),
                     columns=["market_id", "day", "winner_up"]).drop_duplicates()
    t = t.merge(m, on="market_id", how="inner")
    t = t[t["winner_up"].notna() & np.isfinite(t["mid"])]
    t["y"] = t["winner_up"].astype(float)
    t["tte_s"] = (300_000 - t["t_ms"]) / 1000.0

    print(f"{len(t):,} ticks, {t['market_id'].nunique():,} markets, "
          f"{t['day'].nunique()} days\n")

    # One observation per market per band: the book's average price in that
    # band and the outcome. A market contributes to a band only if it spent
    # real time there.
    bands = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40),
             (0.40, 0.60), (0.60, 0.80), (0.80, 0.90), (0.90, 0.95),
             (0.95, 1.00)]
    rows = []
    for lo, hi in bands:
        sel = t[(t["mid"] >= lo) & (t["mid"] < hi)]
        if not len(sel):
            continue
        per = sel.groupby(["day", "market_id"]).agg(
            book=("mid", "mean"), y=("y", "first"), ticks=("mid", "size"))
        per = per[per["ticks"] >= 100].reset_index()      # >= 10 s in band
        if len(per) < 30:
            continue
        # Selling the band at the book: you receive `book`, you pay `y`.
        per["edge_c"] = 100.0 * (per["book"] - per["y"])
        clo, chi = day_blocked_ci(per, "edge_c")
        rows.append({
            "band": f"[{lo:.2f},{hi:.2f})",
            "markets": len(per),
            "book": per["book"].mean(),
            "realised": per["y"].mean(),
            "sell_edge_c": per["edge_c"].mean(),
            "ci_lo": clo, "ci_hi": chi,
            "signif": "yes" if (clo > 0) == (chi > 0) else "",
        })

    out = pd.DataFrame(rows).set_index("band")
    print("--- book price vs realised outcome, per market, by band ---")
    print("(sell_edge_c > 0 means SELLING at the book wins; < 0 means BUYING")
    print(" wins. CI is a 2.5/97.5 percentile day-blocked bootstrap.)\n")
    print(out.to_string(float_format=lambda v: f"{v:9.3f}"))

    print("\nA band is only worth anything if its interval excludes zero AND")
    print("the edge clears the fee. Taker fee is 0.07*p*(1-p)*0.9167, in")
    print("c/share:")
    for lo, hi in bands:
        p = (lo + hi) / 2
        print(f"  p={p:4.2f}: {100 * 0.07 * p * (1 - p) * 0.9167:5.2f} c")

    gates(t, 0.00, 0.05, "the book's sub-5c longshots")
    gates(t, 0.80, 0.90, "the 0.80-0.90 band")


if __name__ == "__main__":
    main()
