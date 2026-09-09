"""Evaluate the normal-QQ fair-value model: one headline run, the mandatory
sweeps on a subsample, and a small tick-emitting run over four markets chosen
for the per-market detail plots.

There is no longer a maker arm and a taker arm to compare. The execution
policy is unified -- it rests on both sides and crosses when the book is
through the fee-adjusted threshold, in the same pass -- so a single headline
run contains both kinds of fill and `liquidity` on the ledger separates them.

Quote parameters (e_p=0.01, rpl_p=0.0005, max_pos=50, shares=10) are NOT
optimised. They are a modest, round-numbered choice, stated here and in the
report, and are not hill-climbed against this 5-day sample.
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

#: THE ORACLE-OVERLAP WINDOW. `paths.SPOT_ORACLE_WINDOW` is the BTC/USD spot
#: panel rebuilt over 2026-08-17..21, the days on which the book panel, the
#: venue L1 capture and the Chainlink RTDS feed all exist at once. It replaces
#: the old 08-19..24 sample because `fair.py` learns its basis from the oracle
#: and returns NaN without one: on the old panel only 08-19 and 08-20 had a
#: Chainlink line, so a six-day headline was really a two-day one. Two of
#: these five days are partial at the ends, both from the source captures --
#: the venue L1 stream starts 04:19 UTC on 08-17, and the oracle stops at
#: 01:59 UTC on 08-21.
SPOT_DAYS = ("2026-08-17", "2026-08-18", "2026-08-19",
            "2026-08-20", "2026-08-21")

#: Pre-open history handed to the signal blocks. `fair.py`'s basis halflife is
#: 180 s and only converges because of this; `vol.py`'s realised-variance EWMA
#: burns in across it rather than restarting at zero every 300 s. Measured, not
#: assumed -- see `../2026-09-09-vol-fixed/warmup_check.py`.
WARMUP_S = 900.0

QUOTE = QuoteParams(e_p=0.01, rpl_p=0.0005, max_pos=50.0, shares=10.0)

#: EVERY data file this investigation reads, fingerprinted into each run's
#: manifest.json. `load_episodes` is called with an explicit spot path and
#: takes the panel and the strikes from their defaults, so all three are read
#: -- and the panel is the primary data behind every number here. Listing only
#: the spot left the book these results were traded against unrecorded.
SPOT_PATH = paths.SPOT_ORACLE_WINDOW
INPUTS = (paths.PANEL, paths.STRIKES, SPOT_PATH, paths.RTDS_BTC)

#: Chainlink's 1 s RTDS feed (harness.paths.RTDS_BTC) stops at 01:59 UTC on
#: 2026-08-21, so these four of the five SPOT_DAYS carry a full-day twap60
#: line for the per-market detail plot. The four detail markets are drawn
#: from this subset so every detail figure's BTC panel is populated.
RTDS_COVERED_DAYS = ("2026-08-17", "2026-08-18",
                    "2026-08-19", "2026-08-20")

SWEEP_SAMPLE_SIZE = 450
SWEEP_SEED_RNG = 20260909


def main():
    t0 = time.time()
    episodes = load_episodes(spot_path=SPOT_PATH, days=SPOT_DAYS,
                             rtds_path=paths.RTDS_BTC,
                             warmup=True, warmup_s=WARMUP_S)
    # MARKETS WITHOUT AN ORACLE ARE DROPPED, NOT SCORED AS ZERO. `require_spot`
    # filters on the venue feed and knows nothing about Chainlink, but
    # `fair.py` returns NaN without an oracle -- so 2026-08-21's 237
    # oracle-less markets would quote nothing, fill nothing and still land in
    # `stats.headline` as 237 markets of exactly $0.00, dividing the loss by a
    # larger number. The filter is knowable at decision time, which is what
    # makes it a sample rule and not a selection effect.
    n_loaded = len(episodes)
    episodes = [ep for ep in episodes if np.isfinite(ep.chainlink).any()]
    print(f"loaded {n_loaded} episodes with spot in {time.time()-t0:.1f}s; "
          f"{len(episodes)} carry a settlement oracle "
          f"({sum(e.has_warmup for e in episodes)} with a complete "
          f"{WARMUP_S:.0f} s warm-up)")

    sample = Sample(require_spot=True)
    results = {}

    t1 = time.time()
    headline = run(
        HERE, quote=QUOTE,
        execn=ExecConfig(latency=LatencyModel()),
        sample=sample,
        output=Output(seeds=(0, 1, 2), plots=True),
        episodes=episodes,
        inputs=INPUTS,
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
        inputs=INPUTS,
    )
    print(f"sweeps: {time.time()-t2:.1f}s -> {sweep_res['run_dir']}")
    results["sweeps"] = sweep_res

    # -- pick 4 markets that actually traded, for the per-market detail -----
    headline_markets = headline["markets"]
    primary = headline_markets[headline_markets["seed"] == 0]
    traded = primary[(primary["n_fills"] > 0)
                     & primary["day"].isin(RTDS_COVERED_DAYS)]["market_id"].tolist()
    pick_rng = np.random.default_rng(20260909)
    n_pick = min(4, len(traded))
    pick_idx = pick_rng.choice(len(traded), size=n_pick, replace=False)
    chosen = sorted(traded[i] for i in pick_idx)
    print(f"chosen markets for per-market detail (from {RTDS_COVERED_DAYS}, "
         "the Chainlink-covered days):", chosen)

    detail_episodes = [ep for ep in episodes if ep.market_id in chosen]
    t3 = time.time()
    detail_res = run(
        HERE, quote=QUOTE,
        execn=ExecConfig(latency=LatencyModel()),
        sample=Sample(require_spot=True, markets=tuple(chosen)),
        output=Output(emit_ticks=True, tick_markets=tuple(chosen),
                      seeds=(0,), plots=False),
        episodes=detail_episodes,
        inputs=INPUTS,
    )
    print(f"detail (ticks): {time.time()-t3:.1f}s -> {detail_res['run_dir']}")
    results["detail"] = detail_res

    manifest = {
        "quote": QUOTE.__dict__,
        "spot_path": SPOT_PATH,
        "warmup_s": WARMUP_S,
        "spot_days": SPOT_DAYS,
        "sweep_sample_size": len(sweep_episodes),
        "sweep_sample_seed": SWEEP_SEED_RNG,
        "rtds_covered_days": RTDS_COVERED_DAYS,
        "chosen_detail_markets": chosen,
        "run_dirs": {k: v["run_dir"] for k, v in results.items()},
    }
    with open(os.path.join(HERE, "last_run_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
