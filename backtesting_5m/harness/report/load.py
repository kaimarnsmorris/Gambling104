"""Read run folders into tidy frames, many at a time.

The mapping is label -> run directory, so a comparison is the default shape
and a single run is the one-element case.
"""
import json
import os

import pandas as pd

from harness.io import read_parquet


def _tidy(mapping, filename):
    frames = []
    for label, run_dir in mapping.items():
        path = os.path.join(run_dir, filename)
        if not os.path.exists(path):
            continue
        df = read_parquet(path)
        df.insert(0, "run", label)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_runs(mapping):
    """Per-market rows for each run, tagged with a `run` column."""
    return _tidy(mapping, "markets.parquet")


def load_ledgers(mapping):
    """Per-fill rows for each run, tagged with a `run` column."""
    return _tidy(mapping, "ledger.parquet")


def load_summaries(mapping):
    out = {}
    for label, run_dir in mapping.items():
        with open(os.path.join(run_dir, "summary.json")) as fh:
            out[label] = json.load(fh)
    return out


def load_ticks(mapping):
    """Per-tick rows for each run, tagged with a `run` column.

    Only present for runs made with `Output(emit_ticks=True)`, and only for
    the markets named in `tick_markets` -- a full day is ~3,000 rows per
    market, so an unrestricted tick dump is not what you want. Runs without
    the file are skipped rather than raising, so a mapping can mix runs that
    emitted ticks with runs that did not.
    """
    return _tidy(mapping, "ticks.parquet")
