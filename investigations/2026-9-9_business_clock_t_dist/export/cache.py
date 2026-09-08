"""The layers that are shared across variant runs.

L1 - the register bank - is the expensive one and, because kappa_vol moved to the
forward curve, it does not depend on any override. It is swept once for the window
and every variant reads it. That is the warm start: the 14-day burn-in is paid once
rather than once per variant, and no market is lost to it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from fvmodel.base import CACHE


def _key(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def register_bank(win, idx: np.ndarray, params_sha: str) -> np.ndarray:
    """`logv` at `idx`, cached on disk. Variant-invariant by construction."""
    idx = np.unique(np.asarray(idx, dtype=np.int64))
    path = Path(CACHE) / ("logv_%s.npz" % _key(params_sha, int(win.ts[0]), win.n,
                                               idx.size, int(idx[0]), int(idx[-1])))
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["idx"], idx):
            win._cache_idx = idx
            win._cache_logv = z["logv"].astype(np.float64)
            return win._cache_logv
    win.prepare(idx)
    np.savez_compressed(path, idx=idx, logv=win._cache_logv.astype(np.float32))
    return win._cache_logv
