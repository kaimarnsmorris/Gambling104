"""s -- the expected settling TWAP, in USD.

The market does not settle on a price, it settles on a 60 s Chainlink TWAP.
So the quantity to forecast is an AVERAGE over [T-w, T], and this block is its
conditional expectation, E[A_T | F_t], which has two regimes:

    tau >= w   E[A] = S_t
               the whole average is still in the future and S is a martingale,
               so today's spot IS the forecast

    tau <  w   E[A] = (1 - tau/w) * TWAP_realised + (tau/w) * S_t
               the elapsed part of the window is no longer a forecast, it is a
               measurement, so it enters at full weight

An EWM lagged to "impersonate" the TWAP would be a biased estimator of the
first regime, which is 80 % of the window. The lag is not the answer; the
explicit average is. What remains of the smoother here is a short denoise on
the venue mid -- 2 s, against the 60 s of the averaging -- because the
bookTicker tick is not itself the Chainlink feed. Set it to 0 for raw spot.

TWO OPPOSITE READINGS OF THE SAME ARRAY, both deliberate:

  * S_t updates only where `spot_age_ms == 0`. The panel carries the last spot
    forward, and a repeat is not news; letting it update would drag the level
    at 10 Hz because nothing happened.
  * TWAP_realised averages EVERY bucket, carried values included. A flat mean
    over a uniform 100 ms grid IS the time-weighted average, and time passes
    whether or not a quote arrives. Skipping carried buckets here would
    silently reweight the average toward the busy moments.

`vol.py` carries its own copy of the window length, and its own scan. Blocks
are frozen into a run one file per slot, so a block that imports its neighbour
is a block whose frozen copy no longer means what it meant.
"""
import math

import numpy as np

from harness.paths import BUCKET_MS

TWAP_WINDOW_S = 60.0            # Chainlink's lookback, per the 2026-08-14 era
SPOT_SMOOTH_HALFLIFE_S = 2.0    # denoise on the venue mid; 0 disables


def precompute(ep):
    """E[settling TWAP] at each decision index. NaN before the first spot."""
    spot = np.asarray(ep.spot, dtype="float64")
    age = np.asarray(ep.spot_age_ms, dtype="float64")
    bucket_s = BUCKET_MS / 1000.0

    out = np.full(len(ep), np.nan)
    level = float("nan")        # S_t, the smoothed current spot
    prev = -1
    window_sum = 0.0
    window_n = 0

    for i in range(len(ep)):
        px = float(spot[i])
        usable = np.isfinite(px) and px > 0.0

        if usable and age[i] == 0.0:
            if prev < 0 or SPOT_SMOOTH_HALFLIFE_S <= 0.0:
                level = px
            else:
                dt = (i - prev) * bucket_s
                alpha = 1.0 - math.exp(
                    -dt * math.log(2.0) / SPOT_SMOOTH_HALFLIFE_S)
                level = alpha * px + (1.0 - alpha) * level
            prev = i

        tau = ep.tte_s(i)
        if tau > TWAP_WINDOW_S:
            out[i] = level
            continue

        if usable:
            window_sum += px
            window_n += 1

        if window_n == 0 or not np.isfinite(level):
            out[i] = level
            continue

        frac = max(0.0, tau) / TWAP_WINDOW_S
        out[i] = (1.0 - frac) * (window_sum / window_n) + frac * level

    return out
