"""One entry point.

An investigation should declare what it wants and call one function, rather
than importing from six modules and knowing which.
"""
import os
from dataclasses import dataclass

import pandas as pd

from harness.core.run import run
from harness.streams import catalog, registry


@dataclass(frozen=True)
class BacktestResult:
    run_dir: str
    summary: dict
    ledger: pd.DataFrame
    markets: pd.DataFrame
    ticks: pd.DataFrame | None = None


def _fingerprinted_inputs(inputs, streams):
    """`inputs`, then the registry path of every named stream, deduped.

    The manifest fingerprints whatever is in `inputs`, and `backtest()` used
    to pass nothing at all -- so every run made through the public entry point
    was unidentifiable against the data it read. Passing `inputs` alone would
    not fix it in practice either: the caller who names their streams is
    exactly the caller who will not also list those streams' paths by hand.
    The registry already knows each path, so naming a stream is enough.

    An explicitly-passed path wins: it keeps the caller's own spelling of a
    path the registry may hold in another form. A name nothing has registered
    is skipped rather than raised -- the streams themselves are resolved
    downstream, where a missing one is reported properly, and a manifest
    detail must never be what fails a run.
    """
    resolved = []
    for name in streams:
        try:
            resolved.append(registry.resolve(name).path)
        except registry.StreamNotRegistered:
            continue

    paths, seen = [], set()
    for path in (*inputs, *resolved):
        key = os.path.abspath(path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return tuple(paths)


def backtest(*, model=None, streams=(), inputs=(), quote, execn, sample,
             output, episodes, investigation_dir):
    """Run one backtest and return its artefacts.

    `model` names a shared model directory; blocks resolve investigation-first,
    then the model, then harness defaults. The standard stream catalog is
    installed (idempotently) before the run, so `require=(...)` on `sample`
    can name any of the standard streams without the caller having to
    remember to register them first.

    `inputs` is the data files this run read; they are fingerprinted into the
    manifest, and the registry-resolved path of every name in `streams` is
    fingerprinted with them, so a run folder can be matched to its data
    afterwards without the caller listing paths twice.
    """
    catalog.install()
    out = run(investigation_dir, quote, execn, sample, output, episodes,
              inputs=_fingerprinted_inputs(inputs, streams),
              model_dir=model, streams=streams)
    return BacktestResult(run_dir=out["run_dir"], summary=out["summary"],
                          ledger=out["ledger"], markets=out["markets"],
                          ticks=out.get("ticks"))
