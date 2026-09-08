"""Evaluate the normal-QQ fair-value model: one headline run, the mandatory
sweeps on a subsample, and a small tick-emitting run over four markets chosen
for the per-market detail plots.

There is no longer a maker arm and a taker arm to compare. The execution
policy is unified -- it rests on both sides and crosses when the book is
through the fee-adjusted threshold, in the same pass -- so a single headline
run contains both kinds of fill and `liquidity` on the ledger separates them.

Quote parameters (e_p=0.01, rpl_p=0.0005, max_pos=50, shares=10) are NOT
optimised. They are a modest, round-numbered choice, stated here and in the
report, and are not hill-climbed against this 6-day sample.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import numpy as np  # noqa: E402

from harness.build.episodes import load_episodes          # noqa: E402
from harness.core.config import ExecConfig, Output, QuoteParams, Sample  # noqa: E402
from harness.core.latency import LatencyModel             # noqa: E402
from harness.core.run import run                          # noqa: E402
from harness.core.sweeps import run_with_sweeps            # noqa: E402
from harness import paths                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

#: The only six days the spot panel covers (2026-08-18 is absent: a
#: source-archive schema defect on that day, deliberate, not a bug here).
SPOT_DAYS = ("2026-08-19", "2026-08-20", "2026-08-21",
            "2026-08-22", "2026-08-23", "2026-08-24")

QUOTE = QuoteParams(e_p=0.01, rpl_p=0.0005, max_pos=50.0, shares=10.0)

SWEEP_SAMPLE_SIZE = 450
SWEEP_SEED_RNG = 20260909


def main():
    t0 = time.time()
    episodes = load_episodes(spot_path=paths.SPOT, days=SPOT_DAYS)
    print(f"loaded {len(episodes)} episodes with spot in {time.time()-t0:.1f}s")

    sample = Sample(require_spot=True)
    results = {}

    t1 = time.time()
    headline = run(
        HERE, quote=QUOTE,
        execn=ExecConfig(latency=LatencyModel()),
        sample=sample,
        output=Output(seeds=(0, 1, 2), plots=True),
        episodes=episodes,
    )
    print(f"headline: {time.time()-t1:.1f}s -> {headline['run_dir']}")
    print(headline["summary"]["headline"])
    print("gates passed:", headline["summary"]["gates"]["passed"])
    results["headline"] = headline

    # -- sweeps, on a subsample, one seed, clearly labelled -----------------
    rng = np.random.default_rng(SWEEP_SEED_RNG)
    idx = rng.choice(len(episodes), size=min(SWEEP_SAMPLE_SIZE, len(episodes)),
                     replace=False)
    sweep_episodes = [episodes[i] for i in sorted(idx)]
    print(f"sweep sample: {len(sweep_episodes)} markets "
         f"(seed {SWEEP_SEED_RNG}), single sweep seed 0")

    t2 = time.time()
    sweep_res = run_with_sweeps(
        HERE, quote=QUOTE,
        execn=ExecConfig(latency=LatencyModel()),
        sample=sample,
        output=Output(seeds=(0,), plots=False),
        episodes=sweep_episodes,
    )
    print(f"sweeps: {time.time()-t2:.1f}s -> {sweep_res['run_dir']}")
    results["sweeps"] = sweep_res

    # -- pick 4 markets that actually traded, for the per-market detail -----
    headline_markets = headline["markets"]
    primary = headline_markets[headline_markets["seed"] == 0]
    traded = primary[primary["n_fills"] > 0]["market_id"].tolist()
    pick_rng = np.random.default_rng(20260909)
    n_pick = min(4, len(traded))
    pick_idx = pick_rng.choice(len(traded), size=n_pick, replace=False)
    chosen = sorted(traded[i] for i in pick_idx)
    print("chosen markets for per-market detail:", chosen)

    detail_episodes = [ep for ep in episodes if ep.market_id in chosen]
    t3 = time.time()
    detail_res = run(
        HERE, quote=QUOTE,
        execn=ExecConfig(latency=LatencyModel()),
        sample=Sample(require_spot=True, markets=tuple(chosen)),
        output=Output(emit_ticks=True, tick_markets=tuple(chosen),
                      seeds=(0,), plots=False),
        episodes=detail_episodes,
    )
    print(f"detail (ticks): {time.time()-t3:.1f}s -> {detail_res['run_dir']}")
    results["detail"] = detail_res

    manifest = {
        "quote": QUOTE.__dict__,
        "spot_days": SPOT_DAYS,
        "sweep_sample_size": len(sweep_episodes),
        "sweep_sample_seed": SWEEP_SEED_RNG,
        "chosen_detail_markets": chosen,
        "run_dirs": {k: v["run_dir"] for k, v in results.items()},
    }
    with open(os.path.join(HERE, "last_run_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
