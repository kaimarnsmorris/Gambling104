"""Generic figures over one or many runs.

Never called by the engine. Bespoke figures belong in the investigation that
wants them, composed from the same loader -- see
`investigations/2026-09-09-normal-qq-eval/plot_report.py` for the full,
per-run, per-market detail set these are generalised from.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                      # noqa: E402

#: Liquidity.MAKER's int value, duplicated rather than imported so this
#: module never needs a resolved fee block to draw a maker/taker split.
LIQ_MAKER = 1


def cumulative_pnl(markets, path, *, gross=True, title="Cumulative PnL"):
    """One net line per run; the gross line greyed behind it.

    Gross versus net is the split between forecast error and fee drag, which
    on this venue is the difference between "the model is wrong" and "the
    threshold is wrong". It is worth seeing on every chart.
    """
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for label, g in markets.groupby("run", sort=True):
        m = g[g["seed"] == g["seed"].iloc[0]].sort_values("open_ts")
        net = m["pnl_net"].fillna(0.0).cumsum()
        if gross and "pnl_gross" in m:
            gr = m["pnl_gross"].fillna(0.0).cumsum()
            ax.plot(range(len(m)), gr, lw=1.0, color="0.75", zorder=1)
            drag = float(gr.iloc[-1] - net.iloc[-1])
            shares = float(m["shares"].sum())
            label = (f"{label}  (fee drag ${drag:,.0f}"
                     f" = {100 * drag / shares:.3f} c/share)") if shares else label
        ax.plot(range(len(m)), net, lw=1.3, zorder=2, label=label)

    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("market (chronological)")
    ax.set_ylabel("cumulative PnL, USD")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markout_distribution(ledgers, path, clip_c=30.0):
    """Markout (`delta_quality_c`) distribution, maker vs taker, per run.

    Generalised from `plot_markout` in the normal-QQ investigation's
    `plot_report.py`: that function pooled one run's maker and taker fills
    into two histograms; this iterates `groupby("run")` so a sweep's arms
    are overlaid rather than plotted one at a time.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-clip_c, clip_c, 61)
    for label, g in ledgers.groupby("run", sort=True):
        is_maker = g["liquidity"] == LIQ_MAKER
        for liq_label, sub, ls in (("maker", g[is_maker], "-"),
                                   ("taker", g[~is_maker], "--")):
            dq = sub["delta_quality_c"].dropna()
            if not len(dq):
                continue
            ax.hist(dq.clip(-clip_c, clip_c), bins=bins, alpha=0.35,
                   density=True, histtype="step", linestyle=ls, lw=1.5,
                   label=f"{label} {liq_label} (n={len(dq)})")

    ax.axvline(0.0, color="0.3", lw=0.8)
    ax.set_xlabel("markout, delta_quality_c (cents/share, +10s or settlement)")
    ax.set_ylabel("density")
    ax.set_title("Markout distribution by liquidity, per run")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def pnl_by_tte(ledgers, path, bucket_s=30.0):
    """PnL-proxy and fill count by time-to-expiry bucket, one line per run.

    Generalised from `plot_by_tte_bucket`: that function drew bars for one
    run's ledger; this draws a line per run so the shape across the sweep is
    comparable at a glance.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for label, g in ledgers.groupby("run", sort=True):
        if not len(g):
            continue
        tte = 300.0 - g["t_ms"] / 1000.0
        bucket = (tte // bucket_s * bucket_s).astype(int)
        proxy = g["delta_quality_c"] / 100.0 * g["shares"]
        grp = g.assign(tte_bucket=bucket, pnl_proxy_usd=proxy).groupby(
            "tte_bucket").agg(pnl_proxy=("pnl_proxy_usd", "sum"),
                              n_fills=("shares", "size")).sort_index(
            ascending=False)
        axes[0].plot(grp.index, grp["pnl_proxy"], marker="o", ms=3, lw=1.2,
                    label=label)
        axes[1].plot(grp.index, grp["n_fills"], marker="o", ms=3, lw=1.2,
                    label=label)

    axes[0].axhline(0.0, color="0.5", lw=0.8)
    axes[0].set_ylabel("markout PnL proxy, USD")
    axes[0].set_title(
        f"Markout-based PnL proxy and fills, by time-to-expiry bucket "
        f"({bucket_s:.0f}s)")
    axes[0].legend(fontsize=7)

    axes[1].set_ylabel("fill count")
    axes[1].set_xlabel("time to expiry at fill, s (bucket start)")
    axes[1].invert_xaxis()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _default_link(z):
    """The harness's default logistic z -> p, for a generic calibration read.

    A run whose investigation overrides `link.py` should compose its own
    calibration figure against that resolved block, the way
    `plot_report.py::plot_calibration` does -- this generic version exists so
    a sweep can be eyeballed without wiring provenance through the report
    module.
    """
    z = np.asarray(z, dtype="float64")
    out = np.empty_like(z)
    out[z > 40.0] = 1.0
    out[z < -40.0] = 0.0
    mid = (z >= -40.0) & (z <= 40.0)
    out[mid] = 1.0 / (1.0 + np.exp(-z[mid]))
    return out


def calibration(ledgers, path, n_buckets=10):
    """Calibration: predicted fair_p (default logistic link) vs realised
    settlement frequency, one series per run.

    Generalised from `plot_calibration`, which took an external
    `winner_up_by_market` mapping built from episode data. Here the outcome
    is read straight off the ledger's own `mid_t10` / `markout_settled`
    columns: a fill markout out to settlement (`markout_settled`) records
    the realised UP/DOWN outcome (1.0/0.0) in `mid_t10` directly, so no
    external episode lookup is needed.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="0.5", lw=1.0, ls="--", label="perfect")

    for label, g in ledgers.groupby("run", sort=True):
        l = g[g.get("markout_settled", False) == True].dropna(  # noqa: E712
            subset=["z", "mid_t10"])
        if not len(l):
            continue
        predicted = _default_link(l["z"].to_numpy())
        realised = l["mid_t10"].to_numpy()
        try:
            import pandas as pd
            decile = pd.qcut(predicted, n_buckets, duplicates="drop")
            grp = pd.DataFrame({"predicted": predicted, "realised": realised,
                               "decile": decile}).groupby(
                "decile", observed=True).agg(
                predicted=("predicted", "mean"),
                realised=("realised", "mean"), n=("realised", "size"))
        except (ValueError, ImportError):
            continue
        ax.scatter(grp["predicted"], grp["realised"],
                  s=grp["n"] / grp["n"].max() * 200 + 20, zorder=5,
                  label=f"{label} (n={int(grp['n'].sum())})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted fair_p at fill time (decile mean)")
    ax.set_ylabel("realised settlement frequency")
    ax.set_title("Calibration: fills, maker+taker pooled, per run")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
