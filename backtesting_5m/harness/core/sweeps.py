"""The sweeps that must attach to every headline, whether asked for or not.

Two results in this programme's history motivate this file. Taker edge measured
at 200 ms had a day-blocked CI of [+0.157, +0.587]; the same edge at 500 ms had
[-0.026, +0.309]. And no maker fill in this dataset is measured -- there is no
depth, no trade tape and no queue -- so a single maker number is a statement
about the fill block, not about the market.

Reporting one number for either is the failure mode. So the sweep runs.
"""
import json
import os
from dataclasses import replace

from harness.core import stats
from harness.core.run import run

LATENCY_LADDER_MS = (0.0, 100.0, 200.0, 250.0, 500.0)

FILL_ARMS = {
    "optimistic": {"penetration": 0.0},          # upper bound: front of queue
    "adverse_lag": {"penetration": 0.0},         # default; cancel loses races
    "penetration": {"penetration": 0.01},        # conservative queue proxy
}


def _headline(result):
    primary = result["markets"]
    primary = primary[primary["seed"] == primary["seed"].iloc[0]] \
        if len(primary) else primary
    return stats.headline(primary)


def run_with_sweeps(investigation_dir, quote, execn, sample, output, episodes):
    """The reporting entry point. `run()` is the single-arm primitive."""
    base = run(investigation_dir, quote, execn, sample, output, episodes)

    latency_arms = []
    for ms in LATENCY_LADDER_MS:
        arm_latency = replace(execn.latency, place_ms=ms, cancel_ms=ms,
                              take_ms=ms)
        arm = run(investigation_dir, quote, replace(execn, latency=arm_latency),
                  sample, replace(output, plots=False), episodes)
        latency_arms.append({"latency_ms": ms, "headline": _headline(arm),
                             "run_dir": arm["run_dir"]})

    fill_arms = []
    for name, params in FILL_ARMS.items():
        arm = run(investigation_dir, quote,
                  replace(execn, fill_params=params), sample,
                  replace(output, plots=False), episodes)
        fill_arms.append({"arm": name, "fill_params": params,
                          "headline": _headline(arm),
                          "run_dir": arm["run_dir"]})

    base["sweeps"] = {"latency": latency_arms, "fill": fill_arms}

    path = os.path.join(base["run_dir"], "summary.json")
    summary = json.loads(open(path).read())
    summary["sweeps"] = base["sweeps"]
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return base
