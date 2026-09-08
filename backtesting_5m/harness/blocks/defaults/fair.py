"""s -- the expected settling TWAP, in USD.

The harness does NOT own the fair value. This default reads the column the
Gambling102 export supplies. If it is absent the run stops here rather than
quietly scoring a model that does not exist.

For plumbing tests before the export lands, use
`harness/blocks/placeholders/fair_flat.py`.
"""
import numpy as np


def precompute(ep):
    s = np.asarray(ep.s, dtype="float64")
    if not np.isfinite(s).any():
        raise ValueError(
            f"{ep.market_id}: no fair value. Supply a fair export, or drop a "
            f"fair.py into the investigation folder.")
    return s
