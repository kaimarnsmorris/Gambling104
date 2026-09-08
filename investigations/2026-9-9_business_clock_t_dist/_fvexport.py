"""Shared loader for the four block files.

Deliberately NOT named after a harness block slot: the investigation root is the
slot namespace, so a helper called `fees.py` here would silently replace the
harness's fee model.

The export is one second per row; the harness grid is 100 ms. `for_episode` expands
one to the other through `export.build_export.bucket_owner`, so the causality shift
lives in exactly one place and is not restated here.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from export.build_export import MARKET_LEN, bucket_owner
from harness_paths import FAIR_DIR

N_BUCKET = MARKET_LEN * 10
OWNER = bucket_owner()
COLS = ("s", "sigma", "nu", "mu", "sigma_t", "p_model", "p_quoted", "omega", "ok")


def variant() -> str:
    return os.environ.get("FV_VARIANT", "baseline")


@functools.lru_cache(maxsize=4)
def _table(name: str) -> pl.DataFrame:
    p = Path(FAIR_DIR) / ("%s.parquet" % name)
    if not p.exists():
        raise FileNotFoundError(
            "no fair export for variant %r at %s; build it with "
            "`python -m export.build_export --variant %s`" % (name, p, name))
    return pl.read_parquet(p).sort(["market_id", "t_s"])


@functools.lru_cache(maxsize=4)
def temperature(name: str | None = None) -> float:
    """The variant's quoting-layer temperature, read from the export's own sidecar.

    `link.py` recomputes `p_model` from `(z, nu, mu, sigma_t)` because `f` perturbs
    the level, so the quote it produces does not come back through the export's
    `p_quoted` column. But the spec's quoting layer wants `p_quoted`, which differs
    from `p_model` exactly by this knob (`fvmodel.overrides.quoted_prob`). Reading it
    from `overrides_non_default` here - rather than threading it through every
    caller - keeps `link.py` a pure function of the export's own state.
    """
    name = name or variant()
    p = Path(FAIR_DIR) / ("%s.json" % name)
    meta = json.loads(p.read_text(encoding="utf-8"))
    return float(meta["provenance"]["overrides_non_default"].get("temperature", 1.0))


@functools.lru_cache(maxsize=4)
def tail_family(name: str | None = None) -> str:
    """The variant's settlement-tail family ("t" or "normal"), read from the export's
    own sidecar - the same mechanism as `temperature` above, and for the same reason:
    it is a per-variant constant (`Overrides.tail_family`), not a per-row quantity, so
    it belongs in the sidecar JSON rather than a parquet column that would just repeat
    the same string on every row.

    This is load-bearing, not cosmetic: `fvmodel/overrides.py::_scaled_tail` leaves
    `(nu, mu, sigma)` numerically UNTOUCHED when `tail_family == "normal"` - it only
    flips an in-memory `.family` flag on the `SettlementTail` object, which the export
    (a table of numbers) cannot carry. Without reading this flag back out of the
    sidecar, `link._prob` would recompute the Student-t form on those unchanged
    columns and silently price every `normal_tail`-style variant as baseline - which
    is exactly the defect this function exists to close (see fix-round 1 in
    .superpowers/sdd/2026-09-09-fv-consolidation/task-10-report.md).
    """
    name = name or variant()
    p = Path(FAIR_DIR) / ("%s.json" % name)
    meta = json.loads(p.read_text(encoding="utf-8"))
    return str(meta["provenance"]["overrides_non_default"].get("tail_family", "t"))


@functools.lru_cache(maxsize=4096)
def _for_market(name: str, market_id: str) -> dict:
    d = _table(name).filter(pl.col("market_id") == market_id).sort("t_s")
    if len(d) == 0:
        return {c: np.full(N_BUCKET, np.nan) for c in COLS}
    # gather, do not scatter: for every bucket, take the row OWNER says serves it.
    # A row missing from the export leaves its buckets NaN, which the harness already
    # treats as "no fair value here" rather than forward-filling.
    t_s = d["t_s"].to_numpy()
    pos = np.searchsorted(t_s, OWNER)
    have = (pos < t_s.size) & (t_s[np.clip(pos, 0, t_s.size - 1)] == OWNER)
    pos = np.clip(pos, 0, t_s.size - 1)
    out = {}
    for c in COLS:
        v = d[c].to_numpy().astype(np.float64)
        out[c] = np.where(have, v[pos], np.nan)
    return out


def for_episode(ep) -> dict:
    return _for_market(variant(), ep.market_id)
