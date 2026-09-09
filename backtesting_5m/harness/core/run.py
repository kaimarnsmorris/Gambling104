"""Resolve blocks, select a sample, replay it, score it, write it down."""
import json
import os
from dataclasses import asdict, is_dataclass, replace

import pandas as pd

from harness import paths
from harness.core import provenance, stats
from harness.core.loop import run_episode
from harness.core.signals import (block_signature, normalise_params,
                                  precompute_signals)
from harness.streams import registry

#: Measured twice on this exact substrate: the 100 ms grid arm earns more than
#: an events arm by this much per market (CI [+0.053, +0.239]; replicated on
#: red_fast at +0.124, CI [+0.025, +0.229]). Carried on every run so no result
#: leaves here pretending the replay clock is free.
GRID_BIAS_USD_PER_MARKET = 0.13

#: Where the spot build records the offset it was TOLD to use, per day.
VENUE_OFFSETS_TSV = os.path.join(paths.RESULTS, "venue_vantage_offsets.tsv")

#: The bound this repo can actually defend: data/results/vantage_offsets.tsv
#: measured same-book inter-host drift on this infrastructure at roughly
#: 0-74 ms. It is a bound on the plausible range, not a measurement of THIS
#: offset, which no property of these two streams can recover.
CLOCK_OFFSET_PLAUSIBLE_RANGE_MS = (0.0, 74.0)

CLOCK_OFFSET_NOTE = (
    "The venue-to-panel clock offset is a CONFIGURED ASSUMPTION, not a "
    "measurement. It cannot be measured from these two streams: the panel "
    "records receipt of Polymarket book updates and the venue feed records "
    "receipt of Binance/Coinbase/OKX/Bybit updates on a different host, so "
    "they share no event to align on, and cross-correlating them would "
    "confound the offset with the spot-to-book response lag this harness "
    "exists to study. The default is 0.0 s. This repo's own same-book "
    "inter-host drift measurements bound the plausible range at 0-74 ms. "
    "Every spot-derived number here is conditional on that assumption and "
    "must be swept over that band the way latency and fill optimism are."
)


def _clock_offset_caveat(days):
    """What offset this run's spot actually rests on, or an admission."""
    caveat = {
        "clock_offset_is_assumed_not_measured": CLOCK_OFFSET_NOTE,
        "clock_offset_default_s": 0.0,
        "clock_offset_plausible_range_ms": list(
            CLOCK_OFFSET_PLAUSIBLE_RANGE_MS),
    }

    by_day = {}
    if os.path.exists(VENUE_OFFSETS_TSV):
        try:
            table = pd.read_csv(VENUE_OFFSETS_TSV, sep="\t")
            by_day = {str(d): float(o) for d, o
                      in zip(table["day"], table["offset_s"])}
            caveat["clock_offset_source"] = VENUE_OFFSETS_TSV
        except Exception:                       # noqa: BLE001 - a caveat must
            by_day = {}                         # never be able to fail a run

    used = {d: by_day[d] for d in days if d in by_day}
    caveat["clock_offset_s_by_day"] = used or None
    if not used:
        caveat["clock_offset_unknown_to_this_run"] = (
            "No configured offset is recorded for the days in this run, so "
            "the offset these results rest on is unknown. Read them as "
            "assuming 0.0 s, with the 0-74 ms band as the sensitivity.")
    return caveat


def _fee_fields(schedule):
    """The fee schedule as plain fields, for hashing. Never its identity."""
    if is_dataclass(schedule):
        return asdict(schedule)
    try:
        return {k: v for k, v in sorted(vars(schedule).items())
                if not k.startswith("_")}
    except TypeError:
        return {"repr": repr(schedule)}


def config_dict(quote, execn, sample, output, fee_schedule, streams=(),
                signal_params=()):
    """Everything that makes this run a different run.

    The run-folder suffix is a hash of this, so anything omitted here makes two
    genuinely different configurations look identical on disk. `fill_params`,
    the time-to-expiry window and the fee schedule were all omitted, which is
    how the `adverse_lag` and `penetration` sweep arms -- differing ONLY in
    fill_params -- shipped run folders with the same suffix and byte-identical
    config blocks. That inverts what the hash is for.

    `streams` is recorded too, so the manifest shows which named streams a
    run actually used rather than leaving that to be inferred.

    So is `signal_params`. Two arms of a vol-scale sweep differ in NOTHING
    else -- same quote, same execution, same blocks -- so omitting it here
    would give them one config hash, one run folder, and the second arm
    silently overwriting the first. That is the same failure `fill_params`
    already caused once.
    """
    return {
        "quote": asdict(quote),
        "signal_params": {slot: dict(pairs) for slot, pairs
                          in normalise_params(signal_params)},
        "sample": asdict(sample),
        "output": asdict(output),
        "latency": asdict(execn.latency),
        "max_book_age_ms": execn.max_book_age_ms,
        "requote_every": execn.requote_every,
        "min_tte_s": execn.min_tte_s,
        "max_tte_s": execn.max_tte_s,
        "fill_params": dict(execn.fill_params),
        "fees": _fee_fields(fee_schedule),
        "streams": list(streams),
    }


#: `require=("spot",)` is answered by `ep.has_spot`, not by `ep.streams` --
#: the panel predates the registry and `load_episodes` reads it by its own
#: path. It is the ONE pre-gridded name a require clause can use.
_REQUIRABLE_PRE_GRIDDED = ("spot",)


def check_require(sample):
    """Refuse a `require` clause that can only ever select nothing.

    `load_episodes` never attaches a pre-gridded stream to `ep.streams` --
    there is nothing to grid, and gridding it would drop every row -- so
    `require=("book",)` or `require=("spot_london_usdt",)` matches no episode
    and silently drops 100 % of the sample, reporting a drop count and a
    headline over zero markets. Selecting zero markets is a failure this
    branch has now hit twice, and it reads exactly like a real result.
    """
    for name in sample.require:
        if name in _REQUIRABLE_PRE_GRIDDED:
            continue
        try:
            reg = registry.resolve(name)
        except registry.StreamNotRegistered:
            continue                    # unregistered: reported downstream
        if reg.pre_gridded:
            raise ValueError(
                f"require={name!r} can never match: {name!r} is a pre-gridded "
                f"panel, already on the decision grid, so it is never "
                f"attached to ep.streams and every episode would be dropped. "
                f"Only {_REQUIRABLE_PRE_GRIDDED[0]!r} is answerable this way, "
                f"via ep.has_spot.")


def select_episodes(episodes, sample):
    """(kept, dropped_counts). Drop counts are reported in summary.json so a
    require clause that halves the sample is visible rather than inferred."""
    check_require(sample)
    dropped = {}
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
            dropped["spot"] = dropped.get("spot", 0) + 1
            continue
        missing = None
        for name in sample.require:
            if name == "spot":
                ok = bool(ep.has_spot.any())
            else:
                ok = name in ep.streams and bool(ep.streams[name]["has"].any())
            if not ok:
                missing = name
                break
        if missing:
            dropped[missing] = dropped.get(missing, 0) + 1
            continue
        out.append(ep)

    out.sort(key=lambda e: e.open_ts)
    if sample.max_markets is not None:
        out = out[: sample.max_markets]
    return out, dropped


def _seed_frame(markets, seed):
    """The market rows for one seed.

    An empty `selected` (every episode dropped by `sample`) leaves `markets`
    with no columns at all, so indexing by "seed" would raise KeyError rather
    than yield the empty frame the summary already knows how to score.
    """
    if "seed" not in markets.columns:
        return markets
    return markets[markets["seed"] == seed]


def _withhold_daily_minimum(fees_module, ledger, markets):
    """Apply the daily rebate floor to the ledger AND to the market rows.

    `markets` is accumulated inside the seed loop out of each episode's own
    `fees_paid`, so it is already final by the time the ledger is adjusted.
    Adjusting only the ledger left ledger.parquet and markets.parquet
    disagreeing by exactly the withheld dust, and left the headline and the
    gates -- both computed from `markets` -- reporting the UNADJUSTED number
    under a caveat stating that the minimum had been applied.

    `loop.run_episode` defines pnl_net = pnl_gross - fees, so fees enter
    linearly and the correction is exact: the per-(market_id, seed) change in
    charged fees is added to `fees` and subtracted from `pnl_net`. `pnl_gross`
    does not move -- withholding a rebate is a fee change, not a change to PnL
    before fees. Markets with no fills never appear in the ledger, so their
    delta is zero and their rows stand exactly as the loop scored them.
    """
    adjusted = fees_module.apply_daily_minimum(ledger)
    if not len(ledger) or not len(markets):
        return adjusted, markets

    keys = [k for k in ("market_id", "seed")
            if k in ledger.columns and k in markets.columns]
    if not keys:
        return adjusted, markets

    delta = (adjusted.groupby(keys)["fee_usd"].sum()
             .subtract(ledger.groupby(keys)["fee_usd"].sum(), fill_value=0.0))
    idx = (pd.MultiIndex.from_frame(markets[keys]) if len(keys) > 1
           else pd.Index(markets[keys[0]]))
    moved = delta.reindex(idx).fillna(0.0).to_numpy()

    markets = markets.copy()
    markets["fees"] = markets["fees"] + moved
    markets["pnl_net"] = markets["pnl_net"] - moved
    return adjusted, markets


def run(investigation_dir, quote, execn, sample, output, episodes, inputs=(),
        model_dir=None, streams=(), signal_cache=None,
        signal_params=()):
    """Replay one configuration. `inputs` is the data files this run read.

    Every path in `inputs` is fingerprinted into the manifest, which is how a
    run folder answers "which panel, which spot build, which fair export?"
    later. Passing nothing leaves `manifest["inputs"]` empty and the run
    unidentifiable against its data.

    `model_dir` names a shared block directory: slots resolve
    investigation-first, then the model, then harness defaults. `streams` is
    recorded into the config so the manifest shows which named streams this
    run used.

    `signal_cache` is a dict shared across arms of a grid. `s` and `sigma`
    depend on the episode, the `fair`/`vol` blocks and `signal_params` alone
    -- not on quote parameters, not on the seed -- and cost 61 % of an arm,
    so recomputing them per arm is the single largest waste in a sweep.
    `signal_params` is `{slot: {name: value}}` and reaches that block's
    `precompute` as keyword arguments, which is how a grid sweeps a model
    parameter (`{"vol": {"scale": 1.6}}`, say) without editing the block per
    point. Both are recorded in the config. See
    harness/core/signals.py for what the cache key covers and why it must.
    """
    resolved = provenance.resolve_slots(investigation_dir, model_dir=model_dir)
    modules = {slot: provenance.load_slot(path, slot)
               for slot, path in resolved.items()}

    # A schedule on the config is a run parameter and wins; otherwise the
    # resolved `fees` block supplies it, so an investigation can still
    # override the schedule the way it overrides any other slot.
    fee_schedule = (execn.fees if execn.fees is not None
                    else modules["fees"].FeeSchedule())
    # Policy prices against fees, so it must see the SAME schedule the
    # ledger charges -- including one that came from an overridden block.
    execn = replace(execn, fees=fee_schedule)

    config = config_dict(quote, execn, sample, output, fee_schedule, streams,
                         signal_params)

    # Selected BEFORE the run folder is created: a `require` clause that
    # cannot match raises here, and a run that never starts should not leave
    # a folder and a manifest behind claiming it did.
    selected, dropped = select_episodes(episodes, sample)

    run_dir = provenance.new_run_dir(investigation_dir, config)
    provenance.write_manifest(run_dir, config, resolved, inputs=tuple(inputs),
                              seeds=output.seeds)

    blocks = {
        "f": modules["f"].standardise,
        "link": modules["link"].link,
        "quote": modules["quote"].quotes,
        "execution": modules["execution"].decide,
        "fill": modules["fill"].resolve,
        "fees": fee_schedule,
        "fill_params": dict(execn.fill_params),
    }

    tick_ids = set(output.tick_markets)

    all_fills, all_markets, all_ticks = [], [], []

    # ONCE, not once per seed. Every seed replays the same market against the
    # same signal; only the latency draws differ. Computing these inside the
    # seed loop -- as this did -- made a 3-seed run do the expensive 61 %
    # three times over for identical arrays.
    signals = precompute_signals(
        selected, modules, block_signature(resolved), signal_cache,
        signal_params)

    for seed in output.seeds:
        rows = []
        for ep in selected:
            eb = dict(blocks)
            eb["s"], eb["sigma"] = signals[ep.market_id]
            emit = output.emit_ticks and (
                not tick_ids or ep.market_id in tick_ids)
            res = run_episode(ep, eb, quote, execn, seed=seed, emit_ticks=emit)

            all_fills.extend(res.pop("fills"))
            all_ticks.extend(res.pop("ticks"))
            rows.append({**res, "seed": seed})

        all_markets.append(pd.DataFrame(rows))

    markets = pd.concat(all_markets, ignore_index=True)
    ledger = pd.DataFrame(all_fills)
    if execn.apply_daily_minimum:
        ledger, markets = _withhold_daily_minimum(
            modules["fees"], ledger, markets)

    # `per_seed` is built HERE, off the same corrected frame the headline is
    # taken from, rather than appended inside the seed loop -- which ran
    # before the ledger adjustment existed. Patching those dicts afterwards
    # would mean re-deriving pnl_per_market and c_per_share in a second copy
    # of stats.headline's arithmetic; recomputing leaves per_seed and headline
    # agreeing by construction.
    per_seed = [{"seed": seed, **stats.headline(_seed_frame(markets, seed))}
                for seed in output.seeds]
    primary = _seed_frame(markets, output.seeds[0])
    summary = {
        "headline": stats.headline(primary),
        "per_seed": per_seed,
        "gates": stats.run_gates(primary),
        "sample": {"n_selected": len(selected), "dropped": dropped},
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
            "daily_rebate_minimum_applied": bool(execn.apply_daily_minimum),
            **_clock_offset_caveat(sorted({ep.day for ep in selected})),
        },
    }

    ledger.to_parquet(os.path.join(run_dir, "ledger.parquet"), index=False)
    markets.to_parquet(os.path.join(run_dir, "markets.parquet"), index=False)
    if all_ticks:
        pd.DataFrame(all_ticks).to_parquet(
            os.path.join(run_dir, "ticks.parquet"), index=False)
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    result = {"run_dir": run_dir, "summary": summary,
              "ledger": ledger, "markets": markets}
    if all_ticks:
        result["ticks"] = pd.DataFrame(all_ticks)
    return result
