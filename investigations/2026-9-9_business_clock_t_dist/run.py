"""Run the Chainlink fair-value model over the 5 m book, one variant at a time.

    python run.py                              baseline, one sampled day
    python run.py rho_off eps_x2               those two as well
    python run.py --days 12 --markets 0 baseline    the whole selection window

WHAT A RUN COSTS. The harness's tick loop is about **0.65 s per market per seed**
-- 3,000 buckets of Python per market -- and that dominates everything else by two
orders of magnitude: this model's blocks add 1.3 s once to load the export and then
about 5 ms per market. So the full selection window, 12 days at roughly 288 markets
a day, is ~2,250 market-seeds per seed: an hour and fifty minutes for one variant at
three seeds, and a day and a half for all sixteen.

The defaults are therefore a SAMPLE -- one day, 120 markets, one seed, about ninety
seconds -- which is enough to see that a variant runs and roughly where it sits.
Widen it deliberately with `--days` and `--markets` when you want the real number,
and expect to leave it going.

Each variant is a fair export built by `export/build_export.py`; this only reads
them. The model's blocks live in `models/chainlink_fv` and are shared -- change
one by dropping that filename into THIS folder, which shadows the model and
nothing else.

`fair_is_causal=True` is deliberate and is the one claim here that needs saying
out loud. The export's rows are already on the decision grid: a row carries
information through the instant `open_ts + t_s` and is not used before
`t_ms = max(0, t_s*1000 + 100)`. `harness/core/episode.py` asks that this be
confirmed in writing before the flag is set; it is, in
`backtesting_5m/docs/fair-export-handoff.md`. Leaving the flag off would shift the
export a second time and quietly cost 100 ms of edge.
"""
from __future__ import annotations

import os
import sys

from harness import ExecConfig, Output, QuoteParams, Sample, backtest, paths
from harness.build.episodes import load_episodes
from harness.streams import catalog

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, os.pardir, os.pardir, "models", "chainlink_fv")

#: The selection window. 2026-09-01 onward is unpriceable -- the model's perp and
#: spot inputs end 2026-08-31 -- and 08-26 to 08-31 is the reserved holdout, so a
#: routine run stops at 08-25. See `variants/README.md`.
SELECTION_DAYS = tuple("2026-08-%02d" % d for d in range(14, 26))


def one(variant: str, episodes, seeds):
    """One variant. `FV_VARIANT` is what the model's blocks read to find its export."""
    os.environ["FV_VARIANT"] = variant
    return backtest(
        investigation_dir=HERE,
        model=MODEL,
        quote=QuoteParams(e_p=0.01, rpl_p=0.0005, max_pos=50.0, shares=10.0),
        execn=ExecConfig(),
        sample=Sample(),
        output=Output(seeds=seeds),
        episodes=episodes,
        inputs=(paths.PANEL, paths.STRIKES,
                os.path.join(paths.DATA, "fair", "%s.parquet" % variant)),
    )


def main(variants, n_days=1, max_markets=120, seeds=(0,)):
    catalog.install()
    days = SELECTION_DAYS[:n_days] if n_days else SELECTION_DAYS
    episodes = load_episodes(days=days, fair_is_causal=True)
    if not episodes:
        raise SystemExit("no episodes built for %s" % (days,))
    if max_markets:
        episodes = episodes[:max_markets]
    print("%d market(s) over %d day(s), %d seed(s) -- about %.0f s per variant"
          % (len(episodes), len(days), len(seeds),
             0.65 * len(episodes) * len(seeds)), flush=True)

    runs = {}
    for variant in variants:
        r = one(variant, episodes, seeds)
        sample, head = r.summary["sample"], r.summary["headline"]
        # Two ways to get a clean-looking nothing, both of which exit 0: every
        # market dropped, or every market kept and never quoted because `s` came
        # back NaN. Neither is a result.
        if sample["n_selected"] == 0:
            raise SystemExit("%s: selected 0 markets -- dropped: %r"
                             % (variant, sample["dropped"]))
        if head["n_fills"] == 0:
            raise SystemExit(
                "%s: selected %d markets and filled none. The model quoted "
                "nothing, so check that its export exists and that `s` is finite."
                % (variant, sample["n_selected"]))
        print("%-20s n_selected=%d n_fills=%d pnl_per_market=%+.4f"
              % (variant, sample["n_selected"], head["n_fills"],
                 head["pnl_per_market"]))
        runs[variant] = r.run_dir

    if len(runs) > 1:
        from harness.report import cumulative_pnl, load_runs
        out = os.path.join(HERE, "cum_pnl.png")
        cumulative_pnl(load_runs(runs), out)
        print("wrote %s" % out)
    return runs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", default=["baseline"])
    ap.add_argument("--days", type=int, default=1,
                    help="days from the selection window; 0 for all 12")
    ap.add_argument("--markets", type=int, default=120,
                    help="cap on markets; 0 for every one in the day range")
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    main(a.variants or ["baseline"], n_days=a.days, max_markets=a.markets,
         seeds=tuple(range(a.seeds)))
