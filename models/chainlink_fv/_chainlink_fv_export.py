"""Read the fair export and hand a market's two harness columns to the blocks.

Not a block slot -- the leading underscore keeps it out of the harness's slot
namespace, which is filenames.

The export is one row per market second; the harness decides on a 100 ms grid, so
each row serves ten buckets. `bucket_owner` is that mapping, and it is a copy of
`export/build_export.bucket_owner` rather than an import: a model directory is
meant to be shareable across investigations, so it must not reach back into the
one that happens to hold the builder. `tests/test_harness_blocks.py` pins the copy
against the original, which is the honest way to hold a duplicate.

The variant is chosen by `FV_VARIANT` and the export located by `FV_FAIR_DIR`,
both with the same defaults the rest of this investigation uses.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
import polars as pl

from tailfold import harness_columns

MARKET_LEN = 300
BUCKET_MS = 100
N_BUCKET = MARKET_LEN * 1000 // BUCKET_MS
COLS = ("s", "sigma", "nu", "mu", "sigma_t")


def bucket_owner() -> np.ndarray:
    """For each 100 ms bucket, the export's `t_s` that is allowed to serve it.

    A row carries information through the instant `open_ts + t_s` and becomes
    usable one bucket later, so bucket `i` is served by `(i*100 - 100) // 1000`:
    bucket 0 by the pre-open row, buckets 1..10 by `t_s = 0`, and so on.
    """
    i = np.arange(N_BUCKET, dtype=np.int64)
    return (i * BUCKET_MS - 100) // 1000


def fair_dir() -> Path:
    from harness import paths
    return Path(os.environ.get("FV_FAIR_DIR",
                               os.path.join(paths.DATA, "fair")))


def variant() -> str:
    return os.environ.get("FV_VARIANT", "baseline")


@functools.lru_cache(maxsize=2)
def _table(name: str) -> pl.DataFrame:
    path = fair_dir() / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"no fair export for variant {name!r} at {path}; build it with "
            f"`python -m export.build_export --variant {name}`")
    return (pl.read_parquet(path, columns=["market_id", "t_s", *COLS])
            .sort(["market_id", "t_s"]))


def tail_family(name: str | None = None) -> str:
    """The variant's tail family, from the export's sidecar. Defaults to `t`.

    Constant per variant, so `link.py` reads it once and picks its shape for the
    whole run. An export written before the sidecar carried the key is a Student-t
    one, which is why the default is `t` rather than an error.
    """
    return _non_default(name or variant()).get("tail_family", "t")


@functools.lru_cache(maxsize=2)
def _non_default(name: str) -> dict:
    """The variant's non-default overrides, from the export's sidecar.

    Takes a RESOLVED name, never `None`: caching on `None` would key every variant
    to one entry and hand back whichever was read first.

    An export written before a key existed simply lacks it, so every reader here
    supplies its own default rather than treating absence as an error.
    """
    import json

    meta = fair_dir() / f"{name}.json"
    if not meta.exists():
        return {}
    prov = json.loads(meta.read_text()).get("provenance", {})
    return prov.get("overrides_non_default", {})


def temperature(name: str | None = None) -> float:
    """The variant's quoting temperature, from the sidecar. Defaults to 1.0."""
    return float(_non_default(name or variant()).get("temperature", 1.0))


@functools.lru_cache(maxsize=1)
def _by_market(name: str) -> dict:
    # polars hands back a one-element tuple as the group key, not the bare value;
    # keying the dict on the tuple makes every market_id lookup miss and every
    # column come back all-NaN, which the harness would read as "no fair value
    # anywhere" rather than as an error
    return {k[0]: v for k, v in
            _table(name).group_by("market_id", maintain_order=True)}


def columns_for(market_id: str):
    """See `_columns_for`; this resolves the variant so the cache key carries it."""
    return _columns_for(variant(), market_id)


@functools.lru_cache(maxsize=8192)
def _columns_for(name: str, market_id: str):
    """`(s_prime, scale)`, each length `N_BUCKET`, for one market.

    A bucket whose serving row is missing from the export stays NaN. The harness
    treats an absent fair value as absent and simply does not quote there, which
    is what we want -- it never forward-fills, and neither does this.
    """
    rows = _by_market(name).get(market_id)
    if rows is None:
        nan = np.full(N_BUCKET, np.nan)
        return nan, nan.copy()

    own = bucket_owner()
    t_s = rows["t_s"].to_numpy()
    pos = np.searchsorted(t_s, own)
    have = (pos < t_s.size) & (t_s[np.clip(pos, 0, t_s.size - 1)] == own)
    pos = np.clip(pos, 0, t_s.size - 1)

    col = {c: rows[c].to_numpy().astype(np.float64)[pos] for c in COLS}
    s_prime, scale = harness_columns(family=tail_family(name), **col)
    return (np.where(have, s_prime, np.nan), np.where(have, scale, np.nan))
