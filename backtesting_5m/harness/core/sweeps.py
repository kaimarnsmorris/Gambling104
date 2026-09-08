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

#: Adverse selection in this harness is modelled by the cancel race, not by
#: `penetration`: you decide to pull a quote at index i, the cancel lands at
#: i + cancel_latency, and any cross in between fills you anyway. So a
#: genuinely optimistic arm must also zero the cancel latency -- otherwise it
#: is byte-identical to the default arm and the sweep silently runs the same
#: configuration twice.
#:
#: Sentinel distinguishing "override to None" (optimistic's move_cancel_ms,
#: which falls back to cancel_ms in LatencyModel.draw) from "leave whatever
#: the caller configured alone" (adverse_lag and penetration, which must
#: track the caller's latency exactly, not a hardcoded default).
_KEEP = object()

#: Each arm carries both the fill-params dict and a latency override applied
#: with dataclasses.replace on execn.latency (cancel_ms, move_cancel_ms).
#: `_KEEP` means "leave the caller's configured value alone".
FILL_ARMS = {
    "optimistic": {
        "fill_params": {"penetration": 0.0},
        "cancel_ms": 0.0,
        "move_cancel_ms": None,
    },  # upper bound: you always pull in time, on the move path too
    "adverse_lag": {
        "fill_params": {"penetration": 0.0},
        "cancel_ms": _KEEP,
        "move_cancel_ms": _KEEP,
    },  # default: cancel latency exactly as configured
    "penetration": {
        "fill_params": {"penetration": 0.01},
        "cancel_ms": _KEEP,
        "move_cancel_ms": _KEEP,
    },  # conservative queue proxy, cancel latency as configured
}


def _resolve_fill_arm(execn, arm):
    """Apply an entry of FILL_ARMS to execn, returning a new ExecConfig."""
    latency_overrides = {k: v for k, v in
                          (("cancel_ms", arm["cancel_ms"]),
                           ("move_cancel_ms", arm["move_cancel_ms"]))
                          if v is not _KEEP}
    latency = (replace(execn.latency, **latency_overrides)
               if latency_overrides else execn.latency)
    return replace(execn, fill_params=arm["fill_params"], latency=latency)


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
    for name, arm_spec in FILL_ARMS.items():
        arm_execn = _resolve_fill_arm(execn, arm_spec)
        arm = run(investigation_dir, quote, arm_execn, sample,
                  replace(output, plots=False), episodes)
        fill_arms.append({"arm": name, "fill_params": arm_spec["fill_params"],
                          "cancel_ms": arm_execn.latency.cancel_ms,
                          "move_cancel_ms": arm_execn.latency.move_cancel_ms,
                          "headline": _headline(arm),
                          "run_dir": arm["run_dir"]})

    base["sweeps"] = {"latency": latency_arms, "fill": fill_arms}

    path = os.path.join(base["run_dir"], "summary.json")
    summary = json.loads(open(path).read())
    summary["sweeps"] = base["sweeps"]
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return base
