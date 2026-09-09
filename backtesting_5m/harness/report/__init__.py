"""Reporting: loads run folders and draws figures. Never imported by the engine.

`run()` sees exactly one run, so it structurally cannot draw the comparison a
hyperparameter sweep needs. `load_runs`/`load_ledgers` take a mapping of
label -> run directory, making side-by-side the default shape and a single
run the one-element case.
"""
from harness.report.figures import (calibration, cumulative_pnl,
                                    markout_distribution, pnl_by_tte)
from harness.report.load import load_ledgers, load_runs, load_summaries

__all__ = ["calibration", "cumulative_pnl", "load_ledgers", "load_runs",
           "load_summaries", "markout_distribution", "pnl_by_tte"]
