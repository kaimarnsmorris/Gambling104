"""Draw a run. Plotting is a separate step from running, so this takes run folders.

    python figures.py                      the most recent run
    python figures.py <run_dir> [more...]  those, overlaid where a figure supports it

Four figures, because four questions:

    calibration.png   does the model's own p match what happened? This is the one
                      that is about the MODEL. The other three are about the policy.
    markout.png       are the fills adversely selected, and how badly
    pnl_by_tte.png    where in a market's life the PnL is made or lost
    cum_pnl.png       the equity curve, gross and net

`market_detail` is deliberately not here: it needs per-tick output, which is off by
default because it is ~3,000 rows a market. Turn on `Output(emit_ticks=True)` for a
handful of markets when you want to watch one play out.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "figures")


def latest_run() -> str:
    dirs = [os.path.join(RUNS, d) for d in os.listdir(RUNS)
            if os.path.isdir(os.path.join(RUNS, d))
            and os.path.exists(os.path.join(RUNS, d, "summary.json"))]
    if not dirs:
        raise SystemExit("no runs under %s -- run `python run.py` first" % RUNS)
    return max(dirs, key=os.path.getmtime)


def main(run_dirs):
    from harness.report import (calibration, cumulative_pnl, load_ledgers,
                                load_runs, markout_distribution, pnl_by_tte)

    os.makedirs(OUT, exist_ok=True)
    mapping = {os.path.basename(d.rstrip(os.sep)): d for d in run_dirs}
    markets = load_runs(mapping)
    ledgers = load_ledgers(mapping)

    drawn = []
    for name, fn, args in (
            ("calibration", calibration, (ledgers, markets)),
            ("markout", markout_distribution, (ledgers,)),
            ("pnl_by_tte", pnl_by_tte, (ledgers,)),
            ("cum_pnl", cumulative_pnl, (markets,)),
    ):
        path = os.path.join(OUT, "%s.png" % name)
        fn(*args, path)
        drawn.append(path)
        print("wrote %s" % path)
    return drawn


if __name__ == "__main__":
    main(sys.argv[1:] or [latest_run()])
