"""Scoring, and the four gates that decide whether to believe it.

Day-blocked, not market-blocked. Markets inside a day share a regime, a book
and a competitor set, so resampling markets independently understates the
uncertainty -- often badly. Resampling whole days is the only version that
answers 'would another fortnight have shown this?'.

The gates come from the architecture doc's section 4, which observes that they
exist in the programme and are applied inconsistently. Here they run on every
result whether anyone remembers to ask or not.
"""
import numpy as np
import pandas as pd


def _settled(markets):
    if len(markets) == 0:
        return markets
    return markets[np.isfinite(markets["pnl_net"])]


def headline(markets):
    m = _settled(markets)
    total_shares = float(m["shares"].sum()) if len(m) else 0.0
    total_pnl = float(m["pnl_net"].sum()) if len(m) else 0.0
    return {
        "n_markets": int(len(m)),
        "n_fills": int(m["n_fills"].sum()) if len(m) else 0,
        "total_shares": total_shares,
        "pnl_total": total_pnl,
        "pnl_per_market": total_pnl / len(m) if len(m) else float("nan"),
        "c_per_share": 100.0 * total_pnl / total_shares
        if total_shares else float("nan"),
    }


def day_blocked_ci(markets, column="pnl_net", n_boot=10000, seed=0,
                   alpha=0.05):
    """Percentile CI for mean per-market PnL, resampling whole days."""
    m = _settled(markets)
    if len(m) == 0:
        return float("nan"), float("nan")

    by_day = [g[column].to_numpy() for _, g in m.groupby("day", sort=True)]
    if len(by_day) < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n_days = len(by_day)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_days, n_days)
        means[b] = np.concatenate([by_day[j] for j in pick]).mean()
    return (float(np.quantile(means, alpha / 2.0)),
            float(np.quantile(means, 1.0 - alpha / 2.0)))


def gate_sign_survives_periods(markets, n_periods=3):
    """Split the calendar into equal periods; the sign must hold in each."""
    m = _settled(markets)
    if len(m) == 0:
        return {"passed": False, "detail": "no settled markets"}

    days = sorted(m["day"].unique())
    if len(days) < n_periods:
        n_periods = max(1, len(days))
    chunks = np.array_split(np.array(days), n_periods)

    overall = np.sign(m["pnl_net"].mean())
    per_period = []
    for chunk in chunks:
        sub = m[m["day"].isin(set(chunk))]
        per_period.append(float(sub["pnl_net"].mean()) if len(sub) else 0.0)

    passed = bool(overall != 0 and all(
        np.sign(v) == overall for v in per_period))
    return {"passed": passed, "per_period": per_period,
            "overall_mean": float(m["pnl_net"].mean())}


def gate_ci_excludes_zero(markets, **kw):
    lo, hi = day_blocked_ci(markets, **kw)
    passed = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0))
    return {"passed": passed, "ci": [lo, hi]}


def gate_delete_top_n(markets, n=10):
    """An edge that lives in ten markets is not an edge."""
    m = _settled(markets)
    if len(m) <= n:
        return {"passed": False, "detail": f"only {len(m)} markets"}

    full = float(m["pnl_net"].mean())
    trimmed = m.sort_values("pnl_net", ascending=False).iloc[n:]
    without = float(trimmed["pnl_net"].mean())
    passed = bool(full != 0 and np.sign(without) == np.sign(full))
    return {"passed": passed, "mean_full": full, "mean_without_top": without}


def run_gates(markets, **kw):
    gates = {
        "sign_survives_periods": gate_sign_survives_periods(markets),
        "ci_excludes_zero": gate_ci_excludes_zero(markets, **kw),
        "delete_top_10": gate_delete_top_n(markets, n=10),
    }
    return {"passed": all(g["passed"] for g in gates.values()),
            "gates": gates}
