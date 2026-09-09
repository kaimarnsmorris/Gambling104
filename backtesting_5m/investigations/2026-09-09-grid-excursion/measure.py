"""Measure how much intra-bucket venue price movement the harness's 100 ms
sampling grid hides.

BACKGROUND. `harness/build/spot_5m_100ms.py` builds the panel's spot column
by taking the FIRST observation in each [t, t+100) ms bucket. Any excursion
that happens inside a bucket and reverts before the next bucket boundary is
invisible to the backtest. The owner's concern is directional: a venue
overreacting for tens of milliseconds could trigger a real trade that loses
money, while a backtest that never samples the spike never makes that trade.
The grid is biased toward calm, and calm is optimistic.

DATA. This script reads
    l1_mid_10ms.parquet  (recv_ns, venue, mid) -- 5 venues, ~23.8M rows
a 10 ms-cadence (event-driven, not a fixed clock) L1 mid series that is
finer than the harness's 100 ms grid but coarser than "everything that
happened" -- a spike that both forms and reverts inside a single 10 ms tick
gap is invisible here too. That caveat is unavoidable with this data and is
stated plainly in REPORT.md; it does not change the direction of the
argument (if anything it means the true hidden-excursion tail is understated
here, not overstated).

METHOD. For each 100 ms bucket that has data, the harness's own bucket
boundaries are reproduced exactly: market opens (`open_ts`, from the
harness's own `data/spot_5m_100ms.parquet`) are whole seconds and always a
multiple of 300 s (5 minutes), so flooring `recv_ns` to the nearest 100 ms on
the GLOBAL epoch grid lands every observation in the same bucket the harness
would have used (see `_prove_bucket_equivalence` below and REPORT.md for the
algebra). No per-market join is needed for bucket assignment; a per-market
join is used only to restrict "per market" statistics to buckets the harness
actually could have used, i.e. markets whose full [open_ts, open_ts+300)
window sits inside this file's covered time range.

Within a bucket: mid_first = the harness's sampled value (first tick by
recv_ns). Because |x - mid_first| is maximized, over any finite set that
contains mid_first, at one of the set's own max or min, the excursion is

    excursion = max(bucket_max - mid_first, mid_first - bucket_min, 0)

which is obtainable from a single groupby().agg(['first','max','min','last'])
-- no per-row transform needed.

Deterministic: no randomness anywhere in this script.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from harness.io import read_parquet
from harness import paths

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = (r"C:/Users/kaima/OneDrive/Documents/GitHub/Gambling102/research"
             r"/btc/backtesting/2026-08-23_backtesting_5m/data/london"
             r"/l1_mid_10ms.parquet")

BUCKET_NS = 100_000_000          # 100 ms, matches harness.paths.BUCKET_MS
MARKET_BUCKETS = paths.N_BUCKET  # 3000
REVERT_TOL_BPS = 1.0             # "back to mid_first" tolerance for stat 2
MATERIAL_BPS_FLOOR = 0.0         # no floor; materiality is relative (1x/2x/5x median)

# vol.py / f.py / link.py defaults, replicated directly (see those modules
# under harness/blocks/defaults/) rather than instantiating an Episode:
SIGMA_AT_300S = 250.0            # USD, 1 sigma over 300 s (vol.py default)


def sigma_at(tte_s):
    return SIGMA_AT_300S * np.sqrt(max(tte_s, 1e-6) / 300.0)


def standardise(level, strike, sigma):
    if sigma <= 0.0:
        return 0.0
    return (level - strike) / sigma


def link(z):
    if z > 40.0:
        return 1.0
    if z < -40.0:
        return 0.0
    return 1.0 / (1.0 + np.exp(-z))


def load_l1(columns=("recv_ns", "venue", "mid")):
    df = read_parquet(DATA_PATH, columns=list(columns))
    return df


def load_harness_opens():
    """Every 5-minute market open_ts the harness actually replays."""
    spot = read_parquet(paths.SPOT, columns=["open_ts"])
    return np.sort(spot["open_ts"].unique().astype("int64"))


def _prove_bucket_equivalence():
    """Sanity check cited in the module docstring: global-epoch 100 ms
    flooring reproduces harness.build.spot_5m_100ms.bucket_venue_l1's
    per-market bucketing, given open_ts is an integer multiple of 100 (it is
    always a multiple of 300, one 5-minute market). Returns True/asserts.
    """
    rng = np.random.default_rng(0)
    open_ts = 1_787_098_200  # a real 5-min-aligned open, arbitrary here
    ts = open_ts + rng.uniform(0, 300, size=2000)
    # harness formula (offset_s=0, EPS negligible at this scale)
    t_ms_local = np.floor((ts - open_ts) * 1000.0 / 100.0 + 1e-5).astype("int64") * 100
    harness_bucket = open_ts * 10 + (t_ms_local // 100)
    # this script's formula
    recv_ns = (ts * 1e9).astype("int64")
    global_bucket = recv_ns // BUCKET_NS
    assert np.array_equal(harness_bucket, global_bucket), \
        "global epoch flooring diverges from the harness bucket grid"
    return True


def per_venue_bucket_stats(df):
    """One row per (venue, bucket) with first/last/max/min/count of mid."""
    df = df.copy()
    df["bucket"] = df["recv_ns"] // BUCKET_NS
    df = df.sort_values(["venue", "recv_ns"], kind="mergesort")
    g = (df.groupby(["venue", "bucket"], sort=False)["mid"]
           .agg(["first", "last", "max", "min", "count"])
           .reset_index())
    g["excursion_usd"] = np.maximum(g["max"] - g["first"], g["first"] - g["min"])
    g["excursion_usd"] = g["excursion_usd"].clip(lower=0.0)
    g["excursion_bps"] = 10_000.0 * g["excursion_usd"] / g["first"]
    g["reverted"] = (
        (g["last"] - g["first"]).abs() <= (REVERT_TOL_BPS / 10_000.0) * g["first"]
    )
    return g


def restrict_to_covered_markets(bucket_stats, harness_opens, data_min_ns, data_max_ns):
    """Keep only buckets belonging to a market the harness replays AND whose
    full 300 s window sits inside this file's time coverage."""
    bucket_stats = bucket_stats.copy()
    bucket_stats["market_open_ts"] = (bucket_stats["bucket"] // MARKET_BUCKETS) * 300
    opens_set = set(harness_opens.tolist())
    covered_opens = np.array(sorted(
        o for o in opens_set
        if o * 1_000_000_000 >= data_min_ns
        and (o + 300) * 1_000_000_000 <= data_max_ns
    ))
    mask = bucket_stats["market_open_ts"].isin(set(covered_opens.tolist()))
    return bucket_stats[mask].copy(), covered_opens


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def summarize_distribution(g, label):
    exc_usd = g["excursion_usd"].to_numpy()
    exc_bps = g["excursion_bps"].to_numpy()
    row = {
        "label": label,
        "n_buckets": len(g),
        "median_usd": pct(exc_usd, 50), "p90_usd": pct(exc_usd, 90),
        "p99_usd": pct(exc_usd, 99), "p999_usd": pct(exc_usd, 99.9),
        "max_usd": float(exc_usd.max()) if len(exc_usd) else float("nan"),
        "median_bps": pct(exc_bps, 50), "p90_bps": pct(exc_bps, 90),
        "p99_bps": pct(exc_bps, 99), "p999_bps": pct(exc_bps, 99.9),
        "max_bps": float(exc_bps.max()) if len(exc_bps) else float("nan"),
        "frac_single_obs": float((g["count"] == 1).mean()) if len(g) else float("nan"),
    }
    return row


def reversion_stats(g, materiality_usd):
    """Of buckets whose excursion exceeds `materiality_usd`, what fraction
    reverted to within REVERT_TOL_BPS of mid_first by bucket end?"""
    material = g[g["excursion_usd"] > materiality_usd]
    if len(material) == 0:
        return {"n_material": 0, "frac_reverted": float("nan")}
    return {
        "n_material": int(len(material)),
        "frac_reverted": float(material["reverted"].mean()),
    }


def frequency_per_market(g, covered_opens, median_bucket_usd):
    """Events per 5-minute market exceeding 1x/2x/5x the median bucket
    excursion, per venue."""
    n_markets = len(covered_opens)
    out = {}
    for mult in (1, 2, 5):
        thresh = mult * median_bucket_usd
        n = int((g["excursion_usd"] > thresh).sum())
        out[f"events_per_market_{mult}x"] = (n / n_markets) if n_markets else float("nan")
        out[f"threshold_usd_{mult}x"] = thresh
    return out


def probability_translation(excursion_usd_values, ttes=(240, 120, 30)):
    """z/p translation of representative excursions at the shipped default
    vol/f/link chain, assuming the market sits at-the-money (baseline
    z=0, p=0.5) when the spike hits -- the most probability-sensitive case."""
    rows = []
    for label, exc in excursion_usd_values:
        for tte in ttes:
            sigma = sigma_at(tte)
            z = standardise(exc, 0.0, sigma)
            p_after = link(z)
            rows.append({
                "excursion_label": label, "excursion_usd": exc, "tte_s": tte,
                "sigma_usd": sigma, "z": z, "p_after": p_after,
                "delta_p_cents": (p_after - 0.5) * 100.0,
            })
    return pd.DataFrame(rows)


def make_plots(g, out_dir):
    venues = sorted(g["venue"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    all_exc = g["excursion_bps"].to_numpy()
    all_exc = all_exc[all_exc > 0]
    bins = np.logspace(np.log10(max(all_exc.min(), 1e-4)), np.log10(all_exc.max()), 60)
    ax.hist(all_exc, bins=bins, color="#3b6ea5")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("intra-bucket excursion (bps, log scale)")
    ax.set_ylabel("bucket count (log scale)")
    ax.set_title("Excursion distribution, all venues pooled\n(tail hidden by the 100 ms grid)")

    ax = axes[1]
    # Raw per-venue median is 0 (most buckets never move at all -- see
    # REPORT.md); "median | moving" is the median size of a real
    # intra-bucket move, conditional on one happening, which is the
    # meaningful central comparison here.
    med_moving = [
        g.loc[(g["venue"] == v) & (g["excursion_bps"] > 0), "excursion_bps"].median()
        for v in venues
    ]
    p999s = [pct(g.loc[g["venue"] == v, "excursion_bps"].to_numpy(), 99.9) for v in venues]
    x = np.arange(len(venues))
    width = 0.35
    ax.bar(x - width / 2, med_moving, width, label="median | moving", color="#3b6ea5")
    ax.bar(x + width / 2, p999s, width, label="p99.9 (all buckets)", color="#c0524a")
    ax.set_xticks(x)
    ax.set_xticklabels(venues)
    ax.set_ylabel("excursion (bps)")
    ax.set_title("Per-venue excursion: typical move vs. tail")
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(out_dir, "excursion_distribution.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main():
    out_dir = HERE
    _prove_bucket_equivalence()

    df = load_l1()
    data_min_ns = int(df["recv_ns"].min())
    data_max_ns = int(df["recv_ns"].max())
    harness_opens = load_harness_opens()

    g_all = per_venue_bucket_stats(df)
    g_cov, covered_opens = restrict_to_covered_markets(
        g_all, harness_opens, data_min_ns, data_max_ns)

    # --- coverage summary -------------------------------------------------
    n_harness_markets = len(harness_opens)
    n_covered_markets = len(covered_opens)
    coverage = {
        "data_min_utc": pd.Timestamp(data_min_ns, unit="ns", tz="UTC").isoformat(),
        "data_max_utc": pd.Timestamp(data_max_ns, unit="ns", tz="UTC").isoformat(),
        "n_harness_markets_total": int(n_harness_markets),
        "n_markets_fully_covered": int(n_covered_markets),
        "coverage_frac_of_harness_markets": n_covered_markets / n_harness_markets,
    }

    # --- stat 1: excursion distribution, pooled and per venue -------------
    dist_rows = [summarize_distribution(g_cov, "ALL_VENUES_POOLED")]
    for v in sorted(g_cov["venue"].unique()):
        dist_rows.append(summarize_distribution(g_cov[g_cov["venue"] == v], v))
    dist_df = pd.DataFrame(dist_rows)

    # "Median bucket range" is degenerate (0) at two levels: a single-tick
    # bucket has no measurable intra-bucket movement by construction
    # (27% of buckets, "frac_single_obs" above); and 86% of MULTI-tick
    # buckets (count >= 2) still show zero excursion -- the mid genuinely
    # did not move between L1 updates inside that 100 ms window. Both are
    # real, reported findings, not noise to filter past. But a "1x/2x/5x
    # the median bucket range" materiality threshold is meaningless at 0,
    # so the baseline used for stats 2 and 3 is the median excursion among
    # buckets where the mid actually moved at all (excursion_usd > 0) --
    # i.e. "the median size of a real intra-bucket move, when one happens".
    g_moving = g_cov[g_cov["excursion_usd"] > 0]

    # --- stat 2: reversion, materiality = pooled median moving-bucket
    # excursion --
    pooled_median_usd = g_moving["excursion_usd"].median()
    rev_rows = []
    for v in sorted(g_cov["venue"].unique()):
        gv = g_cov[g_cov["venue"] == v]
        r = reversion_stats(gv, pooled_median_usd)
        r["venue"] = v
        rev_rows.append(r)
    rev_df = pd.DataFrame(rev_rows)

    # --- stat 3: frequency of material excursions per market --------------
    freq_rows = []
    for v in sorted(g_cov["venue"].unique()):
        gv = g_cov[g_cov["venue"] == v]
        median_v = g_moving.loc[g_moving["venue"] == v, "excursion_usd"].median()
        r = frequency_per_market(gv, covered_opens, median_v)
        r["venue"] = v
        r["median_moving_bucket_excursion_usd"] = median_v
        freq_rows.append(r)
    freq_df = pd.DataFrame(freq_rows)

    # --- stat 4: probability translation -----------------------------------
    p99 = pct(g_cov["excursion_usd"].to_numpy(), 99)
    p999 = pct(g_cov["excursion_usd"].to_numpy(), 99.9)
    mx = float(g_cov["excursion_usd"].max())
    prob_df = probability_translation([
        ("p99_pooled", p99), ("p99.9_pooled", p999), ("max_pooled", mx),
    ])

    # --- stat 5: per-venue share of extreme excursions ---------------------
    thresh_extreme = pct(g_cov["excursion_usd"].to_numpy(), 99.9)
    extreme = g_cov[g_cov["excursion_usd"] > thresh_extreme]
    venue_share = (extreme["venue"].value_counts(normalize=True)
                   .rename("share_of_p999_extremes").reset_index()
                   .rename(columns={"index": "venue"}))
    venue_counts = (extreme["venue"].value_counts()
                    .rename("n_extremes").reset_index()
                    .rename(columns={"index": "venue"}))
    # Raw share of extremes is exposure-weighted (a venue with more buckets
    # of data contributes more extremes even at an equal underlying rate).
    # The rate below -- extremes per bucket OF THAT VENUE -- is the fair
    # per-venue comparison; it also flags that a coarser tick cadence (cb
    # ticks ~3-6x less often than the others, see REPORT.md) can suppress
    # observed extremes independent of true venue behaviour.
    n_buckets_by_venue = g_cov["venue"].value_counts().rename("n_buckets_venue")
    venue_counts = venue_counts.merge(
        n_buckets_by_venue.reset_index().rename(columns={"index": "venue"}),
        on="venue")
    venue_counts["extreme_rate_per_bucket"] = (
        venue_counts["n_extremes"] / venue_counts["n_buckets_venue"])

    plot_path = make_plots(g_cov, out_dir)

    # --- write machine-readable output for REPORT.md to cite ---------------
    dist_df.to_csv(os.path.join(out_dir, "excursion_distribution.csv"), index=False)
    rev_df.to_csv(os.path.join(out_dir, "reversion_stats.csv"), index=False)
    freq_df.to_csv(os.path.join(out_dir, "frequency_per_market.csv"), index=False)
    prob_df.to_csv(os.path.join(out_dir, "probability_translation.csv"), index=False)
    venue_share.to_csv(os.path.join(out_dir, "venue_extreme_share.csv"), index=False)
    venue_counts.to_csv(os.path.join(out_dir, "venue_extreme_counts.csv"), index=False)
    pd.DataFrame([coverage]).to_csv(os.path.join(out_dir, "coverage.csv"), index=False)

    print("=== COVERAGE ===")
    for k, v in coverage.items():
        print(f"  {k}: {v}")
    print("=== EXCURSION DISTRIBUTION (usd/bps) ===")
    print(dist_df.to_string(index=False))
    print("=== REVERSION (materiality = pooled median MOVING-bucket excursion, "
          f"${pooled_median_usd:.2f}) ===")
    print(rev_df.to_string(index=False))
    print("=== FREQUENCY PER MARKET (300-bucket market) ===")
    print(freq_df.to_string(index=False))
    print("=== PROBABILITY TRANSLATION (vol.py/f.py/link.py defaults) ===")
    print(prob_df.to_string(index=False))
    print(f"=== PER-VENUE SHARE OF >p99.9 EXTREME EXCURSIONS "
          f"(threshold ${thresh_extreme:.2f}) ===")
    print(venue_counts.merge(venue_share, on="venue").to_string(index=False))
    print(f"=== PLOT: {plot_path} ===")


if __name__ == "__main__":
    main()
