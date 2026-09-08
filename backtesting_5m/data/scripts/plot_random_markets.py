"""Plot the UP token's book path for a random selection of markets.

Small multiples, one panel per market: the mid as a line, the bid/ask spread as
a band behind it. Nothing is carried forward in the panel, so a bucket no
capture observed is a *gap* -- runs separated by more than `GAP_MS` are drawn
broken rather than joined, because a straight line across a hole is a claim the
data does not make.

    python plot_random_markets.py                 # 3 random markets, shown
    python plot_random_markets.py --seed 7        # reproducible pick
    python plot_random_markets.py --out books.png # written instead of shown
    python plot_random_markets.py --market <id>   # a specific one, repeatable
"""
import argparse
import datetime as dt
import os
import sys

import matplotlib
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

#: Categorical slots 1-3, validated all-pairs on the light surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"

#: Draw a break when consecutive observations are further apart than this. The
#: grid is 100 ms, so this is ten missed buckets -- long enough that the run is
#: a real hole in the capture, not the ordinary one-bucket miss.
GAP_MS = 1000


def pick_markets(n, seed, wanted=None):
    """Return (market_id, open_ts, strike, settle) for the markets to draw.

    The universe is the panel's own markets, not the strike table's: every
    panel market has a strike, but 229 markets with a strike did not survive
    the completeness filter and have no book to plot.
    """
    strikes = pl.read_parquet(common.STRIKES)
    universe = (
        pl.scan_parquet(common.PANEL).select("market_id").unique().collect()
    )
    have = strikes.join(universe, on="market_id", how="semi")

    if wanted:
        missing = set(wanted) - set(have["market_id"])
        if missing:
            raise SystemExit(
                "not in the panel: " + ", ".join(sorted(missing))
            )
        picked = have.filter(pl.col("market_id").is_in(wanted))
    else:
        picked = have.sample(n, seed=seed)

    # settle(N) is the strike of the market opening at open_ts + 300; it is
    # not stored anywhere else. The last markets in the sample have no
    # successor yet, so this is a left join and settle may be null.
    nxt = strikes.select(
        (pl.col("open_ts") - common.H).alias("open_ts"),
        pl.col("strike").alias("settle"),
    )
    return picked.join(nxt, on="open_ts", how="left").sort("open_ts")


def load_books(market_ids):
    """The book for these markets, as {market_id: DataFrame}."""
    df = (
        pl.scan_parquet(common.PANEL)
        .select("market_id", "t_ms", "bid", "ask", "mid")
        .filter(pl.col("market_id").is_in(market_ids))
        .sort("t_ms")
        .collect()
    )
    return {mid: g for (mid,), g in df.group_by("market_id", maintain_order=True)}


def broken(t_ms, *series):
    """Split the series at coverage gaps by inserting a NaN at each break."""
    cut = np.flatnonzero(np.diff(t_ms) > GAP_MS) + 1
    t = np.insert(t_ms.astype(float), cut, np.nan)
    return (t,) + tuple(np.insert(s.astype(float), cut, np.nan) for s in series)


def draw(markets, books, out):
    import matplotlib.pyplot as plt

    # Dollar figures in the titles are text, not TeX.
    matplotlib.rcParams["text.parse_math"] = False

    n = len(markets)
    height = 2.5 * n + 1.1
    fig, axes = plt.subplots(n, 1, figsize=(9.5, height), sharex=True, dpi=150)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)

    for ax, row, color in zip(axes, markets.iter_rows(named=True), SERIES):
        b = books[row["market_id"]]
        t_ms = b["t_ms"].to_numpy()
        t, mid, bid, ask = broken(
            t_ms, b["mid"].to_numpy(), b["bid"].to_numpy(), b["ask"].to_numpy()
        )
        secs = t / 1000.0

        ax.set_facecolor(SURFACE)
        ax.axhline(0.5, color=GRID, lw=1.0, zorder=0)
        ax.fill_between(
            secs, bid, ask, color=color, alpha=0.22, lw=0, zorder=1
        )
        ax.plot(secs, mid, color=color, lw=1.2, solid_capstyle="round", zorder=2)

        opened = dt.datetime.fromtimestamp(row["open_ts"], dt.timezone.utc)
        head = opened.strftime("%Y-%m-%d %H:%M UTC")
        strike, settle = row["strike"], row["settle"]
        if settle is None:
            tail = f"strike ${strike:,.2f} · not yet settled"
        else:
            up = settle >= strike           # ties are Up
            tail = (
                f"strike ${strike:,.2f} → settle ${settle:,.2f} "
                f"({settle - strike:+,.2f}) · resolved "
                + ("UP" if up else "DOWN")
            )
        ax.set_title(f"{head}   {tail}", loc="left", x=0.022, fontsize=9.5,
                     color=INK, pad=8)
        # A chip before the title, sat on the title's own baseline plus a third
        # of its cap height. Identity is carried by the title text; the chip
        # only ties the panel to the colour of its line.
        cap = (9.5 * 0.7 / 72) / ax.get_window_extent().height * fig.dpi
        ax.plot([0.0, 0.013], [ax.title.get_position()[1] + cap * 0.85] * 2,
                transform=ax.transAxes, color=color, lw=2.6, clip_on=False,
                solid_capstyle="round")

        gaps = int((np.diff(t_ms) > GAP_MS).sum())
        note = f"{len(b):,} of {common.N_BUCKET:,} buckets"
        if gaps:
            note += f" · {gaps} gap{'s' if gaps > 1 else ''}"
        ax.text(1.0, 1.028, note, transform=ax.transAxes, ha="right",
                fontsize=8, color=INK_MUTED)

        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
        ax.set_ylabel("P(up)", fontsize=9, color=INK_2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8.5, length=0)

    axes[-1].set_xlim(0, common.H)
    axes[-1].set_xticks(range(0, common.H + 1, 60))
    axes[-1].set_xlabel("seconds since the window opened", fontsize=9,
                        color=INK_2)

    # Placed in inches off the top edge, so the header keeps its spacing
    # whatever `n` does to the figure height.
    fig.text(0.012, 1 - 0.30 / height,
             f"UP token book — {n} random 5 m market{'s' if n > 1 else ''}",
             ha="left", va="top", fontsize=12.5, color=INK, weight="medium")
    fig.text(0.012, 1 - 0.56 / height,
             "line = mid, band = bid/ask spread; breaks are buckets no capture saw",
             ha="left", va="top", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.80 / height))

    if out:
        fig.savefig(out, facecolor=SURFACE)
        print(f"wrote {out}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=3, help="how many markets (default 3)")
    p.add_argument("--seed", type=int, default=None, help="seed for the pick")
    p.add_argument("--market", action="append", metavar="ID",
                   help="plot this market id instead of a random one; repeatable")
    p.add_argument("--out", metavar="PNG", help="write here instead of showing")
    a = p.parse_args()

    if a.out:
        matplotlib.use("Agg")

    markets = pick_markets(a.n, a.seed, a.market)
    books = load_books(markets["market_id"].to_list())
    for row in markets.iter_rows(named=True):
        print(row["market_id"], dt.datetime.fromtimestamp(
            row["open_ts"], dt.timezone.utc).isoformat())
    draw(markets, books, a.out)


if __name__ == "__main__":
    main()
