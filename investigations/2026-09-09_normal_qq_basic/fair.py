"""s -- the expected settling TWAP, in USD.

The 5 m market settles on Chainlink's 60 s TWAP, and this is the cheapest
honest proxy for it: a 60 s halflife EWM of the venue mid. The smoother is not
noise reduction, it is an impersonation -- the settling number is an average
over the last minute, so a spot estimate that tracks the tick is estimating the
wrong quantity.

WHY IT UPDATES ON OBSERVATIONS, NOT ON INDICES: `shift_to_decision_grid`
carries the last spot forward, so `ep.spot` has no gaps -- it repeats. Feeding
a repeat back into the EWM would pull s toward a stale price at the 10 Hz grid
rate purely because nothing happened. `spot_age_ms == 0` is exactly the "the
bucket before this index held news" flag, so that is the update trigger, and dt
is the real elapsed time since the previous update.

The scan is duplicated in `vol.py` rather than shared. Blocks are frozen into a
run one file per slot, so a block that imports its neighbour is a block whose
frozen copy no longer means what it meant.
"""
import math

import numpy as np

from harness.paths import BUCKET_MS

S_HALFLIFE_S = 60.0             # Chainlink's own TWAP window


def precompute(ep):
    """The EWM level visible at each decision index. NaN before the first."""
    spot = np.asarray(ep.spot, dtype="float64")
    age = np.asarray(ep.spot_age_ms, dtype="float64")
    bucket_s = BUCKET_MS / 1000.0

    out = np.full(len(ep), np.nan)
    level = float("nan")
    prev = -1

    for i in range(len(ep)):
        if age[i] == 0.0 and np.isfinite(spot[i]) and spot[i] > 0.0:
            if prev < 0:
                level = float(spot[i])
            else:
                dt = (i - prev) * bucket_s
                alpha = 1.0 - math.exp(-dt * math.log(2.0) / S_HALFLIFE_S)
                level = alpha * float(spot[i]) + (1.0 - alpha) * level
            prev = i
        out[i] = level

    return out
