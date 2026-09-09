"""Many arms over one set of episodes.

`backtest()` is the single-arm primitive and stays that; this is the shape a
parameter search actually has, and it exists because doing it with a plain
loop over `backtest()` wastes most of the machine.

WHAT IT SAVES, MEASURED. On one day (284 markets) an arm costs 59.5 ms per
episode of `fair`/`vol` precompute against 37.3 ms of feed loop. The
precompute cannot depend on a quote parameter or a seed -- `s` and `sigma`
take the episode and the two SignalBlocks and nothing else -- so a naive
grid recomputes 61 % of its total work at every point. One shared
`signal_cache` removes all of it after the first arm.

An arm that sweeps a MODEL parameter is different: `signal_params` is part
of the key, so those arms each compute their own signal, as they must. Grids
mixing the two share what they can.

WHAT PARALLELISM COSTS. Arms are independent and episodes are read-only, so
this is embarrassingly parallel; the catch is that Windows spawns rather
than forks, so each worker gets its own pickled copy of the episodes AND of
the warmed cache. That is paid once per worker through the pool initialiser,
not once per arm, but it is a fixed cost the grid has to be long enough to
amortise. Measured on this machine (11 workers), against a plain loop over
`backtest()`:

    60 markets,   6 arms:  39.3 s -> 15.4 s cached (2.6x) -> 11.6 s (3.4x)
    120 markets, 24 arms: 260.4 s -> 101.4 s cached (2.6x) -> 34.7 s (7.5x)

The cache is worth 2.6x at any size. Parallelism is worth almost nothing on
six arms and 2.9x on twenty-four, which is why `workers` defaults to 1: a
short grid pays the copies and gets little back.

SEEDS. One seed by default, and that is not a shortcut. A seed changes only
the latency draws, so a multi-seed grid pays N times for a dispersion
estimate it does not need while RANKING arms. Rank on one seed, then re-run
the handful of finalists with `Output(seeds=(0, 1, 2))` and read the spread
there.

Each arm still writes its own run folder with its own manifest, and
`signal_params` is in the config hash, so two arms differing only in vol
scale are two folders rather than one silently overwritten.

WHAT THIS MAKES EASY, AND THE RISK. Twenty-four arms over one day will
produce a best arm with a positive headline whether or not any edge exists;
that is what a grid does. The best arm of a grid is an IN-SAMPLE maximum and
is not a result. `summary["gates"]` on the chosen arm -- sign surviving
across periods, a bootstrap CI excluding zero, the headline surviving
deletion of its best ten markets -- is what separates the two, and a grid is
precisely the situation those gates were written for.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

from harness.core import provenance
from harness.core.api import backtest
from harness.core.config import QuoteParams
from harness.core.signals import (block_signature, normalise_params,
                                  precompute_signals)

#: Set in each worker by `_init`, so the episodes cross the process boundary
#: once per worker rather than once per arm.
_WORKER = {}


@dataclass(frozen=True)
class Arm:
    """One point of a grid.

    `quote` is the usual case. `signal_params` sweeps a model parameter --
    it reaches `fair.precompute` and `vol.precompute` as keyword arguments,
    so a block that wants to be swept declares the parameter and a block
    that takes `(ep)` alone is untouched. `execn` overrides execution for
    this arm only; leave it None to share the grid's.
    """
    label: str
    quote: QuoteParams = field(default_factory=QuoteParams)
    signal_params: tuple = ()
    execn: object = None

    def __post_init__(self):
        object.__setattr__(self, "signal_params",
                           normalise_params(self.signal_params))


def _init(episodes, signal_cache, common):
    _WORKER["episodes"] = episodes
    _WORKER["signal_cache"] = signal_cache
    _WORKER["common"] = common


def _run_one(arm):
    r = _one(arm, _WORKER["episodes"], _WORKER["signal_cache"],
             _WORKER["common"])
    # Only the run folder and the summary come back. A BacktestResult also
    # carries the ledger and markets frames, and pickling those out of every
    # worker would cost more than the arm did -- they are on disk in
    # `run_dir`, which is what `load_runs` reads anyway.
    return arm.label, r.run_dir, r.summary


def _one(arm, episodes, signal_cache, common):
    kw = dict(common)
    if arm.execn is not None:
        kw["execn"] = arm.execn
    return backtest(quote=arm.quote, signal_params=arm.signal_params,
                    episodes=episodes, signal_cache=signal_cache, **kw)


def as_arms(grid):
    """Accept `Arm`s, bare `QuoteParams`, or `(label, QuoteParams)` pairs.

    An unlabelled arm gets a positional name. Labels name run folders in the
    report, so duplicates would plot as one line and quietly discard an arm.
    """
    arms = []
    for n, item in enumerate(grid):
        if isinstance(item, Arm):
            arms.append(item)
        elif isinstance(item, tuple) and len(item) == 2:
            arms.append(Arm(label=str(item[0]), quote=item[1]))
        else:
            arms.append(Arm(label=f"arm {n}", quote=item))
    labels = [a.label for a in arms]
    if len(set(labels)) != len(labels):
        raise ValueError("grid labels must be unique -- they name the runs "
                         "in the report and would plot as one line")
    return arms


def grid_search(*, grid, investigation_dir, execn, sample, output, episodes,
                model=None, streams=(), inputs=(), workers=1,
                signal_cache=None):
    """Run every arm in `grid`, sharing one signal cache. Returns a mapping.

    `grid` is a list of `Arm`, of `QuoteParams`, or of `(label, QuoteParams)`
    pairs. Everything not on the arm is held fixed across them -- vary one
    thing at a time or the comparison means nothing.

    Returns `{label: {"run_dir": ..., "summary": ...}}` in the order given,
    which feeds the report layer directly:

        results = grid_search(...)
        cumulative_pnl(load_runs({k: v["run_dir"]
                                  for k, v in results.items()}), "grid.png")

    `workers` > 1 runs arms in processes; read the module docstring on what
    that copies first. `output` defaults to one seed on purpose -- see the
    same docstring on why ranking does not want three.
    """
    arms = as_arms(grid)
    common = dict(investigation_dir=investigation_dir, execn=execn,
                  sample=sample, output=output, inputs=inputs,
                  model=model, streams=streams)

    if signal_cache is None:
        signal_cache = {}

    if workers and workers > 1:
        # Warm the cache in the PARENT so it crosses the boundary already
        # populated. Otherwise each worker recomputes the 61 % for whichever
        # arms it is handed, the caches never merge back, and the grid pays
        # more than it would have sequentially.
        _warm(investigation_dir, model, episodes, arms, signal_cache)
        n = max(1, min(int(workers), len(arms)))
        with ProcessPoolExecutor(max_workers=n, initializer=_init,
                                 initargs=(episodes, signal_cache,
                                           common)) as pool:
            done = list(pool.map(_run_one, arms))
        return {label: {"run_dir": d, "summary": s} for label, d, s in done}

    return {arm.label: _summarise(_one(arm, episodes, signal_cache, common))
            for arm in arms}


def _summarise(result):
    return {"run_dir": result.run_dir, "summary": result.summary}


def _warm(investigation_dir, model, episodes, arms, signal_cache):
    """Fill `signal_cache` before the pool spawns, once per distinct params.

    Uses the same slot resolution `run()` will, so the signature matches and
    the workers hit the cache rather than silently rebuilding it. The sample
    is not applied: an arm's selection can only remove episodes, so warming
    the whole list is at worst wasted work on episodes no arm reads.
    """
    resolved = provenance.resolve_slots(investigation_dir, model_dir=model)
    modules = {slot: provenance.load_slot(resolved[slot], slot)
               for slot in ("fair", "vol") if slot in resolved}
    signature = block_signature(resolved)
    for params in dict.fromkeys(arm.signal_params for arm in arms):
        precompute_signals(episodes, modules, signature, signal_cache, params)


def default_workers():
    """Cores minus one, floor 1 -- leave the machine usable while it runs."""
    return max(1, (os.cpu_count() or 2) - 1)


def linear_grid(base, name, values, *, label=None):
    """`len(values)` arms varying one `QuoteParams` field. The common case.

        linear_grid(QuoteParams(e_p=0.02), "e_z", [0.1, 0.2, 0.3])

    `label` formats the arm name from the value; the default prints the
    field and the value, which is what a legend wants.
    """
    fmt = label or (lambda v: f"{name}={v:g}")
    return [Arm(label=fmt(v), quote=replace(base, **{name: v}))
            for v in values]
