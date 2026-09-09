"""Sample markets, drawn tick by tick.

    python samples.py [run_dir]

Picks a spread across the PnL distribution of a previous run -- worst, lower
quartile, median, upper quartile, best -- then replays exactly those markets with
per-tick output on and draws each one.

Two reasons this is a separate script from `figures.py`. Ticks are ~3,000 rows a
market, so they are off by default and only written for markets named in
`tick_markets=`. And the replay loads `spot` and `rtds` as well, which `run.py`
does not: without them the detail figure's middle panel -- the venue mid, the
Chainlink oracle that actually settles the market, and `s` chasing it -- is empty,
and that panel is most of why you would look at one market.

Picking by outcome, not at random, is deliberate. The worst and best markets are
where a quoting policy's assumptions show; the median is what it does all day.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))

#: Quoting parameters, set 2026-09-09. e_p is the half-spread in probability;
#: rpl_p is the retreat per lot, which the first run showed was far too weak --
#: position sat pinned at max_pos for most of every market.
E_P, RPL_P, MAX_POS = 0.04, 0.0035, 100.0
MODEL = os.path.join(HERE, os.pardir, os.pardir, "models", "chainlink_fv")
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "figures", "markets")

#: Where in the PnL distribution to sample, and what to call each one.
PICKS = ((0.0, "worst"), (0.25, "q1"), (0.5, "median"), (0.75, "q3"), (1.0, "best"))


def latest_run() -> str:
    dirs = [os.path.join(RUNS, d) for d in os.listdir(RUNS)
            if os.path.exists(os.path.join(RUNS, d, "markets.parquet"))]
    if not dirs:
        raise SystemExit("no runs under %s -- run `python run.py` first" % RUNS)
    return max(dirs, key=os.path.getmtime)


def choose(run_dir):
    """`[(label, market_id, day, pnl_net), ...]` spanning the PnL distribution."""
    m = (pl.read_parquet(os.path.join(run_dir, "markets.parquet"))
         .filter(pl.col("n_fills") > 0).sort("pnl_net"))
    # The detail figure's middle panel needs spot, and venue L1 only begins
    # 2026-08-17 -- markets before it carry has_spot=False and draw an empty
    # panel. Prefer covered markets when the run has any; say so when it does not,
    # because an empty panel looks like a broken figure rather than a data bound.
    if "has_spot" in m.columns and m["has_spot"].any():
        m = m.filter(pl.col("has_spot"))
    elif "has_spot" in m.columns:
        print("note: no market in this run has venue spot (it begins 2026-08-17), "
              "so the BTC panel will show the oracle and `s` but no venue mid")
    n = len(m)
    if not n:
        raise SystemExit("the run filled nothing, so there is nothing to look at")
    out, seen = [], set()
    for q, label in PICKS:
        i = min(int(round(q * (n - 1))), n - 1)
        row = m.row(i, named=True)
        if row["market_id"] in seen:          # a tiny run can collide on quantiles
            continue
        seen.add(row["market_id"])
        out.append((label, row["market_id"], row["day"], row["pnl_net"]))
    return out


def main(run_dir):
    from harness import ExecConfig, Output, QuoteParams, Sample, backtest, paths
    from harness.build.episodes import load_episodes
    from harness.report import load_ledgers, load_ticks, market_detail
    from harness.streams import catalog

    picks = choose(run_dir)
    print("sampling %d markets from %s" % (len(picks), os.path.basename(run_dir)))
    for label, mid, day, pnl in picks:
        print("  %-7s %s  %s  pnl_net %+8.2f" % (label, day, mid[:18], pnl))

    catalog.install()
    days = tuple(sorted({d for _, _, d, _ in picks}))
    ids = tuple(mid for _, mid, _, _ in picks)
    # spot and rtds are what fill the detail figure's BTC panel; `run.py` omits
    # them because nothing it draws reads them
    episodes = [ep for ep in load_episodes(
        days=days, fair_is_causal=True,
        spot_path=paths.SPOT, rtds_path=paths.RTDS_BTC)
        if ep.market_id in ids]
    if len(episodes) != len(ids):
        print("warning: %d of %d markets rebuilt" % (len(episodes), len(ids)))

    os.environ.setdefault("FV_VARIANT", "baseline")
    r = backtest(
        investigation_dir=HERE, model=MODEL,
        quote=QuoteParams(e_p=E_P, rpl_p=RPL_P, max_pos=MAX_POS, shares=10.0),
        execn=ExecConfig(), sample=Sample(),
        output=Output(seeds=(0,), emit_ticks=True, tick_markets=ids),
        episodes=episodes,
    )

    mapping = {"sample": r.run_dir}
    ticks, ledgers = load_ticks(mapping), load_ledgers(mapping)
    os.makedirs(OUT, exist_ok=True)
    for label, mid, day, pnl in picks:
        path = os.path.join(OUT, "%s.png" % label)
        market_detail(ticks, ledgers, mid, path,
                      title="%s -- %s, %s, pnl_net %+.2f" % (label, day, mid[:18], pnl))
        print("wrote %s" % path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else latest_run())
