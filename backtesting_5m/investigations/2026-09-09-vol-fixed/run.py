"""Three vol blocks, one fair/f/link, fit days and test days scored apart.

    baseline    `vol_baseline.py` -- the EWMA + tau_eff block as it stands
    calibrated  `vol.py`          -- the empirical term structure of the
                                     forecast error, fitted on 08-17..18
    flat        `vol_flat.py`     -- k * sqrt(tau), the level-only control

A run is frozen one file per slot, so each arm is MATERIALISED into its own
folder under `runs/variants/<arm>/`: `fair.py`, `f.py` and `link.py` copied
byte-for-byte from here, the arm's vol block copied in as `vol.py`, and
`sigma_fit.json` alongside so the fitted blocks can read it. That way each
arm's `manifest.json` fingerprints exactly the four files that priced it, and
the only file that differs between arms is `vol.py`.

QUOTE PARAMETERS ARE NOT TUNED. `e_p=0.01, rpl_p=0.0005, max_pos=50,
shares=10` is the same unoptimised, round-numbered set the normal-QQ
evaluation used, carried over deliberately so the arms differ in sigma and in
nothing else. Anything that made money here would still have to survive a
sweep; nothing here is hill-climbed.

THE SPLIT IS BY CALENDAR, not by market. Fit on 2026-08-17/18, test on
2026-08-19/20/21. Every headline number below is reported for both halves and
the test half leads, because a sigma fitted and scored on the same days is a
description of those days, not a model.

WHY THOSE DAYS AND NOT THE OLD 08-19..24. `fair.py` learns its basis from the
Chainlink oracle and returns NaN without one; the RTDS capture stops at
2026-08-21 01:59 UTC. On the old panel the test half had NO oracle at all, so
the split was not a split. `paths.SPOT_ORACLE_WINDOW` is the panel rebuilt
over the overlap of the three feeds, and this cut is 511 scorable fit markets
against 591 test.

AND THE ONE THING THE SPLIT CANNOT AVOID: realised movement rises
monotonically across these five days -- 6.5, 5.5, 11.2, 14.6, 27.7 bp at
300 s -- so the test half is 2.28x more volatile than the fit half. An
UNCONDITIONAL sigma table carries none of that across, so the calibrated arm
enters the test half with a sigma roughly half the size the test days
warrant. That is a property of a five-day window with a trend in it, not of
the fit, and it is the largest caveat on every test-half number here.

WARM-UP IS ON, 900 s. `fair.py`'s basis halflife is 180 s and converges only
because of it (see `warmup_check.py` for the measurement), and
`vol_baseline.py`'s realised-variance EWMA burns in across it instead of
restarting at zero every open. Both fit and run must use it or the sigma
being scored is not the sigma that was fitted.

MARKETS WITHOUT AN ORACLE ARE DROPPED, NOT SCORED AS ZERO. `Sample`'s
`require_spot` filters on the venue feed; it knows nothing about Chainlink.
Left alone, 2026-08-21 would contribute 261 spot-covered markets of which
only 24 have an oracle line -- and `fair.py` returns NaN without one, so the
other 237 quote nothing, fill nothing, and enter `stats.headline` as 237
markets of exactly $0.00. That does not measure a model declining to trade;
it silently divides the loss by a larger number. They are filtered out here
instead, which is the same rule `require_spot` applies to the venue feed and
is knowable at decision time, not after the fact. The count before and after
the filter is printed so the size of the drop is on the record.
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402

from harness.build.episodes import load_episodes           # noqa: E402
from harness.core import provenance, stats                 # noqa: E402
from harness.core.config import (ExecConfig, Output,       # noqa: E402
                                 QuoteParams, Sample)
from harness.core.latency import LatencyModel              # noqa: E402
from harness.core.run import run                           # noqa: E402
from harness import paths                                  # noqa: E402

import calibration                                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANT_ROOT = os.path.join(HERE, "runs", "variants")

FIT_DAYS = ("2026-08-17", "2026-08-18")
TEST_DAYS = ("2026-08-19", "2026-08-20", "2026-08-21")
SPOT_DAYS = FIT_DAYS + TEST_DAYS

WARMUP_S = 900.0                # matches fit_sigma.py and the episode default

ARMS = {"baseline": "vol_baseline.py",
        "calibrated": "vol.py",
        "flat": "vol_flat.py"}

QUOTE = QuoteParams(e_p=0.01, rpl_p=0.0005, max_pos=50.0, shares=10.0)
SEEDS = (0, 1, 2)

#: THE ORACLE-OVERLAP PANEL. Same USD build as `paths.SPOT_USD` -- BTC/USDT
#: mid LESS the capture's `usdt_basis` -- rebuilt over 2026-08-17..21, the
#: window where the book panel, the venue L1 capture and the Chainlink RTDS
#: feed all exist at once. `fair.py` returns NaN without an oracle, so on the
#: 08-19..24 panel a six-day headline collapsed to a two-day one.
#:
#: The superseded `paths.SPOT_LEGACY_USDT` carries a ~+43 USD level bias
#: which, measured from this end, put the settling value below the model's
#: forecast on 827 of 827 fit-day markets. Everything in `*_uncorrected.*`
#: was measured on that panel and is kept only so the before/after in
#: REPORT.md reproduces.
SPOT_PATH = paths.SPOT_ORACLE_WINDOW
RESULTS_JSON = "results.json"
CURVES_PNG = "calibration_curves.png"

INPUTS = (paths.PANEL, paths.STRIKES, SPOT_PATH, paths.RTDS_BTC)


def materialise(arm, vol_file):
    """Assemble one arm's block folder. Only `vol.py` differs between arms."""
    out = os.path.join(VARIANT_ROOT, arm)
    os.makedirs(out, exist_ok=True)
    for name in ("fair.py", "f.py", "link.py"):
        shutil.copy2(os.path.join(HERE, name), os.path.join(out, name))
    shutil.copy2(os.path.join(HERE, vol_file), os.path.join(out, "vol.py"))
    shutil.copy2(os.path.join(HERE, "sigma_fit.json"),
                 os.path.join(out, "sigma_fit.json"))
    # A stale __pycache__ under an arm folder would be a different model
    # wearing this arm's name.
    shutil.rmtree(os.path.join(out, "__pycache__"), ignore_errors=True)
    return out


def main():
    t0 = time.time()
    episodes = load_episodes(spot_path=SPOT_PATH, days=SPOT_DAYS,
                             rtds_path=paths.RTDS_BTC,
                             warmup=True, warmup_s=WARMUP_S)
    n_loaded = len(episodes)
    episodes = [e for e in episodes if np.isfinite(e.chainlink).any()]
    print(f"loaded {n_loaded} episodes in {time.time()-t0:.1f}s; "
          f"{len(episodes)} carry a settlement oracle "
          f"({sum(e.has_warmup for e in episodes)} with a complete "
          f"{WARMUP_S:.0f} s warm-up)", flush=True)

    results = {}
    curves = {}

    for arm, vol_file in ARMS.items():
        arm_dir = materialise(arm, vol_file)
        modules = {slot: provenance.load_slot(path, slot)
                   for slot, path in provenance.resolve_slots(arm_dir).items()}
        results[arm] = {}

        for half, days in (("test", TEST_DAYS), ("fit", FIT_DAYS)):
            t1 = time.time()
            obs = calibration.observations(episodes, modules, days=days)
            model = calibration.score(obs, "p")
            book = calibration.score(obs, "book_mid")
            print(f"[{arm}/{half}] calibration {len(obs)} rows in "
                  f"{time.time()-t1:.1f}s  model brier {model['brier']:.4f}  "
                  f"book brier {book['brier']:.4f}  "
                  f"sat_hi {model['sat_hi_frac']:.3f} "
                  f"(realised {model['sat_hi_realised']:.3f})", flush=True)
            if half == "test":
                curves[arm] = calibration.calibration_curve(obs, "p")
                curves["_book"] = calibration.calibration_curve(obs, "book_mid")

            t2 = time.time()
            res = run(arm_dir, quote=QUOTE,
                      execn=ExecConfig(latency=LatencyModel()),
                      sample=Sample(require_spot=True, days=days),
                      output=Output(seeds=SEEDS),
                      episodes=episodes, inputs=INPUTS)
            head = res["summary"]["headline"]
            lo, hi = stats.day_blocked_ci(
                res["markets"][res["markets"]["seed"] == SEEDS[0]])
            print(f"[{arm}/{half}] backtest {time.time()-t2:.1f}s  "
                  f"{head['n_markets']} mkts  "
                  f"{head['c_per_share']:.3f} c/share  "
                  f"{head['pnl_per_market']:+.4f} $/market  "
                  f"CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"gates {res['summary']['gates']['passed']}", flush=True)

            results[arm][half] = {
                "model": {k: v for k, v in model.items() if k != "deciles"},
                "book": {k: v for k, v in book.items() if k != "deciles"},
                "model_deciles": model.get("deciles"),
                "headline": head,
                "day_blocked_ci": [lo, hi],
                "gates": res["summary"]["gates"],
                "run_dir": res["run_dir"],
            }

    with open(os.path.join(HERE, RESULTS_JSON), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nwrote {RESULTS_JSON}; total {time.time()-t0:.1f}s")

    plot_calibration(curves, os.path.join(HERE, CURVES_PNG))
    print(f"wrote {CURVES_PNG}")
    summarise(results)


def plot_calibration(curves, path):
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="perfect calibration")
    style = {"baseline": ("#c0392b", "o-"),
             "calibrated": ("#27853f", "s-"),
             "flat": ("#e08a1e", "^-")}
    for arm, (colour, marker) in style.items():
        if arm not in curves:
            continue
        pred, real, _ = curves[arm]
        ax.plot(pred, real, marker, color=colour, ms=5, label=f"model: {arm}")
    if "_book" in curves:
        pred, real, _ = curves["_book"]
        ax.plot(pred, real, "d-", color="#1b6ca8", ms=5,
                label="book mid (benchmark)")
    ax.set_xlabel("predicted P(up), decile mean")
    ax.set_ylabel("realised frequency")
    ax.set_title("Unconditional calibration, TEST days (08-19..21)\n"
                 "oracle-overlap BTC/USD panel, 900 s warm-up",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def summarise(results):
    print("\n" + "=" * 78)
    print("TEST DAYS (2026-08-19..21) -- leading numbers")
    print("=" * 78)
    print(f"{'arm':<11}{'brier':>8}{'book':>8}{'sat>=.999':>11}"
          f"{'sat realis':>11}{'worst gap':>11}{'c/share':>10}{'$/mkt':>9}")
    for half in ("test", "fit"):
        if half == "fit":
            print("-" * 78)
            print("FIT DAYS (2026-08-17..18) -- in sample for the calibrated arm")
            print("-" * 78)
        for arm, halves in results.items():
            r = halves[half]
            m, b, h = r["model"], r["book"], r["headline"]
            print(f"{arm:<11}{m['brier']:>8.4f}{b['brier']:>8.4f}"
                  f"{m['sat_hi_frac']:>11.3f}{m['sat_hi_realised']:>11.3f}"
                  f"{m['worst_decile_gap']:>11.3f}"
                  f"{h['c_per_share']:>10.3f}{h['pnl_per_market']:>9.4f}")


if __name__ == "__main__":
    main()
