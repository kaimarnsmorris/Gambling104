"""Resolve blocks, select a sample, replay it, score it, write it down."""
import json
import os
from dataclasses import asdict, is_dataclass

import pandas as pd

from harness.core import provenance, stats
from harness.core.loop import run_episode

#: Measured twice on this exact substrate: the 100 ms grid arm earns more than
#: an events arm by this much per market (CI [+0.053, +0.239]; replicated on
#: red_fast at +0.124, CI [+0.025, +0.229]). Carried on every run so no result
#: leaves here pretending the replay clock is free.
GRID_BIAS_USD_PER_MARKET = 0.13


def _fee_fields(schedule):
    """The fee schedule as plain fields, for hashing. Never its identity."""
    if is_dataclass(schedule):
        return asdict(schedule)
    try:
        return {k: v for k, v in sorted(vars(schedule).items())
                if not k.startswith("_")}
    except TypeError:
        return {"repr": repr(schedule)}


def config_dict(quote, execn, sample, output, fee_schedule):
    """Everything that makes this run a different run.

    The run-folder suffix is a hash of this, so anything omitted here makes two
    genuinely different configurations look identical on disk. `fill_params`,
    the time-to-expiry window and the fee schedule were all omitted, which is
    how the `adverse_lag` and `penetration` sweep arms -- differing ONLY in
    fill_params -- shipped run folders with the same suffix and byte-identical
    config blocks. That inverts what the hash is for.
    """
    return {
        "quote": asdict(quote),
        "sample": asdict(sample),
        "output": asdict(output),
        "mode": execn.mode,
        "latency": asdict(execn.latency),
        "max_book_age_ms": execn.max_book_age_ms,
        "requote_every": execn.requote_every,
        "min_tte_s": execn.min_tte_s,
        "max_tte_s": execn.max_tte_s,
        "fill_params": dict(execn.fill_params),
        "fees": _fee_fields(fee_schedule),
    }


def _select(episodes, sample):
    out = []
    for ep in episodes:
        if sample.t0 is not None and ep.open_ts < sample.t0:
            continue
        if sample.t1 is not None and ep.open_ts >= sample.t1:
            continue
        if sample.days and ep.day not in sample.days:
            continue
        if sample.markets and ep.market_id not in sample.markets:
            continue
        if sample.require_spot and not ep.has_spot.any():
            continue
        out.append(ep)
    out.sort(key=lambda e: e.open_ts)
    if sample.max_markets is not None:
        out = out[: sample.max_markets]
    return out


def run(investigation_dir, quote, execn, sample, output, episodes):
    resolved = provenance.resolve_slots(investigation_dir)
    modules = {slot: provenance.load_slot(path, slot)
               for slot, path in resolved.items()}

    # A schedule on the config is a run parameter and wins; otherwise the
    # resolved `fees` block supplies it, so an investigation can still
    # override the schedule the way it overrides any other slot.
    fee_schedule = (execn.fees if execn.fees is not None
                    else modules["fees"].FeeSchedule())

    config = config_dict(quote, execn, sample, output, fee_schedule)

    run_dir = provenance.new_run_dir(investigation_dir, config)
    provenance.write_manifest(run_dir, config, resolved, seeds=output.seeds)

    blocks = {
        "f": modules["f"].standardise,
        "link": modules["link"].link,
        "quote": modules["quote"].quotes,
        "execution": modules["execution"].decide,
        "fill": modules["fill"].resolve,
        "fees": fee_schedule,
        "fill_params": dict(execn.fill_params),
    }

    selected = _select(episodes, sample)
    tick_ids = set(output.tick_markets)

    all_fills, all_markets, all_ticks, per_seed = [], [], [], []

    for seed in output.seeds:
        rows = []
        for ep in selected:
            eb = dict(blocks)
            eb["s"] = modules["fair"].precompute(ep)
            eb["sigma"] = modules["vol"].precompute(ep)
            emit = output.emit_ticks and (
                not tick_ids or ep.market_id in tick_ids)
            res = run_episode(ep, eb, quote, execn, seed=seed, emit_ticks=emit)

            all_fills.extend(res.pop("fills"))
            all_ticks.extend(res.pop("ticks"))
            rows.append({**res, "seed": seed})

        frame = pd.DataFrame(rows)
        all_markets.append(frame)
        per_seed.append({"seed": seed, **stats.headline(frame)})

    markets = pd.concat(all_markets, ignore_index=True)
    ledger = pd.DataFrame(all_fills)

    primary = markets[markets["seed"] == output.seeds[0]]
    summary = {
        "headline": stats.headline(primary),
        "per_seed": per_seed,
        "gates": stats.run_gates(primary),
        "caveats": {
            "grid_bias_usd_per_market": GRID_BIAS_USD_PER_MARKET,
            "grid_bias_note":
                "The 100 ms replay clock flatters results by roughly this much "
                "per market, measured on two independent models. Subtract it "
                "before believing any headline.",
            "maker_fills_are_modelled":
                "No depth, no trade tape, no queue exists in this data. Any "
                "maker number is conditional on the fill block and must be "
                "reported as a range across fill optimism.",
        },
    }

    ledger.to_parquet(os.path.join(run_dir, "ledger.parquet"), index=False)
    markets.to_parquet(os.path.join(run_dir, "markets.parquet"), index=False)
    if all_ticks:
        pd.DataFrame(all_ticks).to_parquet(
            os.path.join(run_dir, "ticks.parquet"), index=False)
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    if output.plots and len(primary):
        from harness.core import plots
        plots.cumulative_pnl(primary, os.path.join(run_dir, "cum_pnl.png"))

    return {"run_dir": run_dir, "summary": summary,
            "ledger": ledger, "markets": markets}
