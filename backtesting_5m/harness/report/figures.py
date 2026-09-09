"""Generic figures over one or many runs.

Never called by the engine. Bespoke figures belong in the investigation that
wants them, composed from the same loader. These four are the generic core
of a per-run, per-market detail set that lived in
`investigations/2026-09-09-normal-qq-eval/plot_report.py`; that folder has
since been deleted, so the fuller version is in git history rather than on
disk.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

from harness.blocks.defaults.fees import Liquidity
from harness.core.types import Side

#: Imported from the enum, never hardcoded -- a hardcoded `1` here once
#: matched `Liquidity.TAKER`, silently swapping every maker/taker split.
LIQ_MAKER = int(Liquidity.MAKER)

#: Same rule, same reason. `Side` is BUY=1, SELL=-1, so the obvious guess of
#: 0-and-1 labels every buy a sell and draws no buys at all -- which is
#: exactly what a first draft of `market_detail` did.
SIDE_BUY, SIDE_SELL = int(Side.BUY), int(Side.SELL)


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


def calibration(ledgers, markets, path, n_buckets=10):
    """Calibration: predicted `fair_p` at fill time vs realised settlement
    frequency, one series per run, over ALL fills (not just the last ~10 s).

    Generalised from `plot_calibration`, which took an external
    `winner_up_by_market` mapping built from episode data and re-derived
    `fair_p` from `z` by resolving the investigation's own `link.py`. Neither
    is available to a generic, multi-run report function, so both are read
    straight off the artefacts instead:

    - `fair_p` is recorded on the fill row directly (`harness.core.loop`
      already computes it as `link(z_i)` beside `z_i`; it is now passed into
      `Ledger.record_fill` rather than discarded), so no link needs
      re-resolving here -- and, unlike re-deriving it from `z` with a
      hardcoded default link, this is exact for whatever link an
      investigation actually used.
    - the outcome is `markets["winner_up"]` (now recorded by `run_episode`),
      joined onto the ledger by `("run", "market_id")`. Earlier this reused
      the ledger's own settlement-markout column, which only covers fills in
      roughly the final 10 s of a 300 s market -- an unrepresentative slice.
      Joining to `markets` gives every fill the market's actual outcome.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="0.5", lw=1.0, ls="--", label="perfect")

    outcomes = markets[["run", "market_id", "winner_up"]].drop_duplicates()
    joined = ledgers.merge(outcomes, on=["run", "market_id"], how="inner")

    for label, g in joined.groupby("run", sort=True):
        l = g.dropna(subset=["fair_p", "winner_up"])
        if not len(l):
            continue
        predicted = l["fair_p"].to_numpy()
        realised = l["winner_up"].astype("float64").to_numpy()
        try:
            decile = pd.qcut(predicted, n_buckets, duplicates="drop")
            grp = pd.DataFrame({"predicted": predicted, "realised": realised,
                               "decile": decile}).groupby(
                "decile", observed=True).agg(
                predicted=("predicted", "mean"),
                realised=("realised", "mean"), n=("realised", "size"))
        except ValueError:
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


def market_detail(ticks, ledger, market_id, path, *, title=None):
    """One market, tick by tick: quotes against the book, BTC space, inventory.

    The generic half of the per-run detail set. Three stacked panels sharing
    a time axis, because the question "why did this trade" is never answered
    in one space:

    * PROBABILITY -- the book's bid and ask, our retreated `eff_bid`/`eff_ask`
      against them, `fair_p`, and every fill marked at its own price. Where a
      quote sits outside the book we were never going to trade; where it sits
      inside, the fill is the interesting one.
    * BTC -- the venue mid we quote off, the Chainlink oracle that settles the
      market, and `s`, the model's estimate of where that oracle lands. The
      gap between `spot` and `chainlink` is the USDT basis the fair block is
      learning, and it is worth seeing that it learned it.
    * INVENTORY -- position and mark-to-market PnL. A retreat parameter is a
      claim about inventory, so the position path is how you check it.

    `ticks` and `ledger` are frames as `load_ticks`/`load_ledgers` return
    them, filtered here to one market and one run.
    """
    t = ticks[ticks["market_id"] == market_id].sort_values("t_ms")
    if not len(t):
        raise ValueError(
            f"no ticks for market {market_id!r}. Ticks are written only for "
            "runs made with Output(emit_ticks=True) and only for the markets "
            "named in tick_markets=.")

    # Every seed replays the SAME market, so a multi-seed run holds one row
    # per (market, t_ms, seed). Overlaying them would draw three inventory
    # paths on one axis and look like noise in the model rather than a
    # choice about latency draws. Take the lowest seed and say which.
    f = (ledger[ledger["market_id"] == market_id]
         if len(ledger) else ledger.iloc[:0])
    seed = None
    if "seed" in t and t["seed"].nunique() > 1:
        seed = int(t["seed"].min())
        t = t[t["seed"] == seed]
        if len(f) and "seed" in f:
            f = f[f["seed"] == seed]
    secs = t["t_ms"].to_numpy() / 1000.0

    fig, (ax, axb, axq) = plt.subplots(
        3, 1, figsize=(11, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1.6]})

    ax.plot(secs, t["book_bid"], lw=0.9, color="0.55", label="book bid")
    ax.plot(secs, t["book_ask"], lw=0.9, color="0.75", label="book ask")
    ax.plot(secs, t["eff_bid"], lw=1.1, color="tab:blue", label="eff_bid")
    ax.plot(secs, t["eff_ask"], lw=1.1, color="tab:red", label="eff_ask")
    ax.plot(secs, t["fair_p"], lw=1.2, color="tab:green", label="fair_p")
    for side, marker, colour, name in ((SIDE_BUY, "^", "tab:blue", "buy"),
                                       (SIDE_SELL, "v", "tab:red", "sell")):
        s = f[f["side"] == side] if len(f) else f
        if len(s):
            ax.scatter(s["t_ms"] / 1000.0, s["price"], s=26, marker=marker,
                       color=colour, edgecolor="k", linewidth=0.4, zorder=5,
                       label=f"{name} ({len(s)})")
    ax.set_ylabel("probability")
    ax.legend(fontsize=7, ncol=4, loc="best")
    head = title or f"market {market_id}"
    ax.set_title(f"{head}  (seed {seed})" if seed is not None else head)

    axb.plot(secs, t["spot"], lw=1.0, color="tab:orange", label="venue spot")
    if "chainlink" in t and t["chainlink"].notna().any():
        axb.plot(secs, t["chainlink"], lw=1.0, color="tab:purple",
                 label="chainlink")
    axb.plot(secs, t["s"], lw=1.2, color="tab:green", label="s (fair BTC)")
    axb.set_ylabel("BTC, USD")
    axb.legend(fontsize=7, ncol=3)

    axq.plot(secs, t["q"], lw=1.1, color="k", label="position")
    axq.axhline(0.0, color="0.6", lw=0.8)
    axq.set_ylabel("position")
    axq.set_xlabel("seconds since open")
    pnl = axq.twinx()
    pnl.plot(secs, t["cum_pnl"], lw=1.0, color="tab:brown", label="MTM PnL")
    pnl.set_ylabel("MTM PnL, USD")
    lines = axq.get_lines()[:1] + pnl.get_lines()
    axq.legend(lines, [ln.get_label() for ln in lines], fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
