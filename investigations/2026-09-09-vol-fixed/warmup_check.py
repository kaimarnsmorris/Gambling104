"""Is the warm-up actually being USED? Measure it; do not assert it.

Turning `warmup=True` on in a runner proves nothing by itself: the flag only
fills some extra arrays on the Episode, and a block that never reads them
produces byte-identical output while paying the gridding cost. So this file
runs the SAME blocks over the SAME markets twice -- once with warm-up on and
once with it off -- and measures the three things that must move if the
history is really being consumed.

  1. B_t AT THE OPEN. Cold, `fair.py` seeds the basis from the single first
     (M - C) print of the market, whose sampling error is the raw gap's own
     sd. Warm, it arrives at the open with 900 s / 180 s = five halflives
     behind it. If the two agree the warm-up is inert.

  2. THE STEP ACROSS A MARKET BOUNDARY. Markets are back to back on a 300 s
     grid, so B at the last bucket of market k and B at the first bucket of
     market k+1 are two estimates of the same quantity 100 ms apart. A cold
     restart puts a sawtooth there: the second estimate throws away
     everything the first one learned. Warm, the recursion crosses the seam
     on an absolute clock and the step should collapse to the size of one
     EWM update.

  3. THE VOL EWMA'S LEVEL. `vol_baseline.py`'s realised-variance EWMA is
     seeded at zero every open and has a 100 s halflife against a 300 s
     window, so cold it reports a variance biased low for most of the episode
     it is scored on. The ratio warm/cold is the size of that bias.

Also writes `warmup_basis.png`: B_t drawn across a run of consecutive
markets, cold and warm on the same axes, with the 300 s boundaries marked.

Usage: python warmup_check.py [--days 2026-08-19,2026-08-20]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

from harness.build.episodes import load_episodes
from harness.core import provenance
from harness import paths

HERE = os.path.dirname(os.path.abspath(__file__))
#: `fair.py` was byte-identical to the canonical model and has been deleted
#: from this folder; loaded from MODEL instead of HERE below.
MODEL = os.path.join(HERE, os.pardir, os.pardir, "models", "normal_qq")
WARMUP_S = 900.0
DEFAULT_DAYS = ("2026-08-19", "2026-08-20")
PLOT_MARKETS = 8                # consecutive markets drawn in the figure
OUT_PNG = "warmup_basis.png"
OUT_JSON = "warmup_check.json"


def _consecutive(eps):
    """Index pairs (k, k+1) whose opens are exactly one market apart.

    `eps` must already be in open order -- `load_episodes` groups by
    (market_id, open_ts) and so returns them sorted by the ID string, which
    for these market IDs is NOT chronological. Sorting is done once, at load,
    so every array in this file shares one ordering.
    """
    return [(i, i + 1) for i in range(len(eps) - 1)
            if eps[i + 1].open_ts - eps[i].open_ts == paths.H]


def main():
    days = DEFAULT_DAYS
    for arg in sys.argv[1:]:
        if arg.startswith("--days"):
            days = tuple(arg.split("=", 1)[1].split(","))

    fair = provenance.load_slot(os.path.join(MODEL, "fair.py"), "fair")
    vol = provenance.load_slot(os.path.join(HERE, "vol_baseline.py"), "vol")
    print(f"fair.BASIS_HALFLIFE_S = {fair.BASIS_HALFLIFE_S} s, "
          f"warm-up = {WARMUP_S} s "
          f"({WARMUP_S / fair.BASIS_HALFLIFE_S:.1f} halflives)")

    cold = load_episodes(spot_path=paths.SPOT_ORACLE_WINDOW, days=days,
                         rtds_path=paths.RTDS_BTC)
    warm = load_episodes(spot_path=paths.SPOT_ORACLE_WINDOW, days=days,
                         rtds_path=paths.RTDS_BTC, warmup=True,
                         warmup_s=WARMUP_S)
    cold.sort(key=lambda e: e.open_ts)
    warm.sort(key=lambda e: e.open_ts)
    assert [e.market_id for e in cold] == [e.market_id for e in warm]
    print(f"markets: {len(cold)}  "
          f"(complete warm-up region on {sum(e.has_warmup for e in warm)})")

    b_cold = [fair.basis_path(e)[1] for e in cold]
    b_warm = [fair.basis_path(e)[1] for e in warm]

    # --- 1. the basis at the open -------------------------------------
    def first_finite(a):
        f = np.flatnonzero(np.isfinite(a))
        return float(a[f[0]]) if len(f) else np.nan

    open_cold = np.array([first_finite(b) for b in b_cold])
    open_warm = np.array([first_finite(b) for b in b_warm])
    ok = np.isfinite(open_cold) & np.isfinite(open_warm)
    d_open = open_warm[ok] - open_cold[ok]
    identical = float(np.mean(d_open == 0.0))

    # --- 2. the step across a market boundary -------------------------
    def last_finite(a):
        f = np.flatnonzero(np.isfinite(a))
        return float(a[f[-1]]) if len(f) else np.nan

    steps_cold, steps_warm = [], []
    for i, j in _consecutive(cold):
        prev_c, prev_w = last_finite(b_cold[i]), last_finite(b_warm[i])
        nxt_c, nxt_w = first_finite(b_cold[j]), first_finite(b_warm[j])
        if np.isfinite(prev_c) and np.isfinite(nxt_c):
            steps_cold.append(abs(nxt_c - prev_c))
        if np.isfinite(prev_w) and np.isfinite(nxt_w):
            steps_warm.append(abs(nxt_w - prev_w))
    steps_cold = np.asarray(steps_cold)
    steps_warm = np.asarray(steps_warm)

    # --- 3. the vol EWMA ----------------------------------------------
    ratios = []
    for ec, ew in zip(cold, warm):
        sc, sw = vol.precompute(ec), vol.precompute(ew)
        f = np.isfinite(sc) & np.isfinite(sw) & (sc > 0)
        if f.any():
            ratios.append(float(np.median(sw[f] / sc[f])))
    ratios = np.asarray(ratios)

    out = {
        "days": list(days),
        "n_markets": len(cold),
        "n_complete_warmup": int(sum(e.has_warmup for e in warm)),
        "basis_halflife_s": float(fair.BASIS_HALFLIFE_S),
        "warmup_s": WARMUP_S,
        "basis_at_open": {
            "n": int(ok.sum()),
            "frac_identical_to_cold": identical,
            "mean_abs_shift_usd": float(np.abs(d_open).mean()),
            "median_abs_shift_usd": float(np.median(np.abs(d_open))),
            "p90_abs_shift_usd": float(np.percentile(np.abs(d_open), 90)),
            "sd_cold_usd": float(np.std(open_cold[ok], ddof=1)),
            "sd_warm_usd": float(np.std(open_warm[ok], ddof=1)),
        },
        "boundary_step_abs_usd": {
            "n_cold": int(len(steps_cold)), "n_warm": int(len(steps_warm)),
            "mean_cold": float(steps_cold.mean()),
            "mean_warm": float(steps_warm.mean()),
            "median_cold": float(np.median(steps_cold)),
            "median_warm": float(np.median(steps_warm)),
            "p90_cold": float(np.percentile(steps_cold, 90)),
            "p90_warm": float(np.percentile(steps_warm, 90)),
            "reduction_x": float(steps_cold.mean() / steps_warm.mean()),
        },
        "vol_ewma_sigma_warm_over_cold": {
            "n_markets": int(len(ratios)),
            "median": float(np.median(ratios)),
            "p10": float(np.percentile(ratios, 10)),
            "p90": float(np.percentile(ratios, 90)),
        },
    }

    print("\n-- 1. B_t at the open, warm vs cold --")
    b = out["basis_at_open"]
    print(f"   identical on {b['frac_identical_to_cold']*100:.2f} % of "
          f"{b['n']} markets")
    print(f"   |shift|  mean ${b['mean_abs_shift_usd']:.3f}  "
          f"median ${b['median_abs_shift_usd']:.3f}  "
          f"p90 ${b['p90_abs_shift_usd']:.3f}")
    print(f"   cross-market sd of B at the open: cold ${b['sd_cold_usd']:.3f} "
          f"-> warm ${b['sd_warm_usd']:.3f}")

    print("\n-- 2. |step in B| across a 300 s market boundary --")
    s = out["boundary_step_abs_usd"]
    print(f"   cold  mean ${s['mean_cold']:.3f}  median ${s['median_cold']:.3f}"
          f"  p90 ${s['p90_cold']:.3f}   (n={s['n_cold']})")
    print(f"   warm  mean ${s['mean_warm']:.3f}  median ${s['median_warm']:.3f}"
          f"  p90 ${s['p90_warm']:.3f}   (n={s['n_warm']})")
    print(f"   reduction: {s['reduction_x']:.2f}x")

    print("\n-- 3. vol_baseline sigma, warm / cold --")
    v = out["vol_ewma_sigma_warm_over_cold"]
    print(f"   median {v['median']:.4f}  p10 {v['p10']:.4f}  "
          f"p90 {v['p90']:.4f} over {v['n_markets']} markets")

    with open(os.path.join(HERE, OUT_JSON), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT_JSON}")

    plot(cold, b_cold, b_warm, os.path.join(HERE, OUT_PNG))
    print(f"wrote {OUT_PNG}")


def plot(eps, b_cold, b_warm, path):
    """B_t across a run of consecutive markets, cold and warm."""
    # The LONGEST unbroken run, not the first one: 50 market slots are absent
    # from the spot panel, so the first run is often only two or three markets
    # wide and would understate what the figure is there to show.
    pairs = set(_consecutive(eps))
    best, cur = [], []
    for i in range(len(eps)):
        cur = cur + [i] if (cur and (cur[-1], i) in pairs) else [i]
        if len(cur) > len(best):
            best = cur
    idx = best[:PLOT_MARKETS]
    if len(idx) < 2:
        print("no consecutive markets to plot")
        return

    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    t0 = eps[idx[0]].open_ts
    for k in idx:
        t = (eps[k].open_ts - t0) + np.arange(len(eps[k])) * (
            paths.BUCKET_MS / 1000.0)
        ax.plot(t, b_cold[k], "-", color="#c0392b", lw=1.0,
                label="cold start (no warm-up)" if k == idx[0] else None)
        ax.plot(t, b_warm[k], "-", color="#27853f", lw=1.0,
                label="900 s warm-up" if k == idx[0] else None)
        ax.axvline(eps[k].open_ts - t0, color="k", lw=0.7, ls=":", alpha=0.6)

    ax.set_xlabel("seconds from the first market's open "
                  "(dotted = 300 s market boundaries)")
    ax.set_ylabel("B_t  =  ewm(venue mid - Chainlink), USD")
    ax.set_title("The learned basis across consecutive markets.\n"
                 "Cold, B restarts from one print at every boundary; warm, "
                 "the recursion crosses the seam.", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
