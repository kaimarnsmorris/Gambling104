"""One entry point.

An investigation should declare what it wants and call one function, rather
than importing from six modules and knowing which.
"""
from dataclasses import dataclass

import pandas as pd

from harness.core.run import run
from harness.streams import catalog


@dataclass(frozen=True)
class BacktestResult:
    run_dir: str
    summary: dict
    ledger: pd.DataFrame
    markets: pd.DataFrame
    ticks: pd.DataFrame | None = None


def backtest(*, model=None, streams=(), quote, execn, sample, output,
             episodes, investigation_dir):
    """Run one backtest and return its artefacts.

    `model` names a shared model directory; blocks resolve investigation-first,
    then the model, then harness defaults. The standard stream catalog is
    installed (idempotently) before the run, so `require=(...)` on `sample`
    can name any of the standard streams without the caller having to
    remember to register them first.
    """
    catalog.install()
    out = run(investigation_dir, quote, execn, sample, output, episodes,
              model_dir=model, streams=streams)
    return BacktestResult(run_dir=out["run_dir"], summary=out["summary"],
                          ledger=out["ledger"], markets=out["markets"],
                          ticks=out.get("ticks"))
