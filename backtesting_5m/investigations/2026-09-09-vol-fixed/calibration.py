"""Unconditional calibration of a fair-value model against the book's own mid.

The method is `investigations/2026-09-09-normal-qq-eval/calibration_uncond.py`,
unchanged in substance: five decision indices per market (tte 270, 210, 150, 90
and 30 s), the model's p and the book's mid on exactly the same rows, scored
against the realised outcome. Two things are added.

  * A DAY FILTER, because the whole point of this investigation is that sigma
    is fitted on 08-19..21 and scored on 08-22..24. A calibration number that
    mixes them is an in-sample number wearing a disguise.
  * SATURATION, the fraction of rows with p >= 0.999. That is the statistic the
    diagnosis rests on: 40 % of the baseline's observations sit at p = 1.000
    and settle in the money only 77 % of the time, which is what a sigma too
    small looks like from the outside.

The book is scored on IDENTICAL ROWS, never on its own larger sample. It is the
benchmark that says how much of the residual Brier is irreducible.
"""
import numpy as np
import pandas as pd

EVAL_INDICES = (300, 900, 1500, 2100, 2700)     # tte 270, 210, 150, 90, 30 s
SATURATED = 0.999


def observations(episodes, modules, days=None):
    """One row per (market, eval index): model p, book mid, outcome."""
    rows = []
    for ep in episodes:
        if days is not None and ep.day not in days:
            continue
        if ep.winner_up is None or not ep.has_spot.any():
            continue
        s = modules["fair"].precompute(ep)
        sig = modules["vol"].precompute(ep)
        for i in EVAL_INDICES:
            if not (np.isfinite(s[i]) and np.isfinite(sig[i])):
                continue
            z = modules["f"].standardise(float(s[i]), ep.strike, float(sig[i]))
            rows.append((ep.market_id, ep.day, ep.tte_s(i),
                         modules["link"].link(z), float(ep.mid[i]),
                         1.0 if ep.winner_up else 0.0))
    return pd.DataFrame(
        rows, columns=["market_id", "day", "tte", "p", "book_mid", "y"]).dropna()


def _decile(df, col):
    q = pd.qcut(df[col], 10, duplicates="drop")
    g = df.groupby(q, observed=True).agg(pred=(col, "mean"),
                                         real=("y", "mean"), n=("y", "size"))
    g["gap"] = g["real"] - g["pred"]
    return g.reset_index(drop=True)


def score(df, col):
    """Brier, saturation, worst decile gap and the decile table for `col`."""
    if not len(df):
        return {"n": 0}
    brier = float(((df[col] - df["y"]) ** 2).mean())
    hi = df[col] >= SATURATED
    lo = df[col] <= 1.0 - SATURATED
    dec = _decile(df, col)
    worst = dec.loc[dec["gap"].abs().idxmax()]
    return {
        "n": int(len(df)),
        "brier": brier,
        "sat_hi_frac": float(hi.mean()),
        "sat_hi_realised": float(df.loc[hi, "y"].mean()) if hi.any() else float("nan"),
        "sat_lo_frac": float(lo.mean()),
        "sat_lo_realised": float(df.loc[lo, "y"].mean()) if lo.any() else float("nan"),
        "sat_frac": float((hi | lo).mean()),
        "worst_decile_gap": float(worst["gap"]),
        "worst_decile_pred": float(worst["pred"]),
        "worst_decile_real": float(worst["real"]),
        "mean_p": float(df[col].mean()),
        "mean_y": float(df["y"].mean()),
        "deciles": dec.to_dict("records"),
    }


def calibration_curve(df, col, bins=10):
    """(predicted, realised, n) per equal-count bin, for plotting."""
    dec = _decile(df, col)
    return (dec["pred"].to_numpy(), dec["real"].to_numpy(),
            dec["n"].to_numpy())
