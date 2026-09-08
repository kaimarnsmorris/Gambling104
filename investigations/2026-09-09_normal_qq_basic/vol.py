"""sigma -- the scale that turns log moneyness into a standardised distance.

An EWMA of realised variance, per observation:

    r2_per_sec = ln(S_i / S_prev)^2 / dt
    alpha      = 1 - exp(-dt / RV_HALFLIFE_S)
    rv_ewm_var = alpha * r2_per_sec + (1 - alpha) * rv_ewm_var

Rates are per second throughout, so an irregular spot feed costs nothing: dt
carries the spacing, and alpha is drawn from the same clock. Returns are taken
on the RAW spot, not on the smoothed level in `fair.py` -- smoothing suppresses
variance by a halflife-dependent factor, and a vol estimate that inherits it is
measuring the smoother.

WHAT THIS RETURNS is sigma_T, the TOTAL log-return sd to expiry, not an
annualised rate. `f.standardise` is handed no time argument by the loop, so
time-to-expiry is folded in here. It is annualised and de-annualised rather
than cancelled algebraically because the annualised number is the one that is
worth reading in a diagnostic.

There is NO shrink and NO clamp: a run that wants those can express them as a
different vol block. Before MIN_UPDATES returns have landed the estimate is
NaN rather than a guess -- the loop declines to quote on a non-finite sigma, so
the model stands aside during warm-up instead of pricing off one observation.

The observation scan is duplicated from `fair.py` on purpose; see the note
there.
"""
import math

import numpy as np

from harness.paths import BUCKET_MS

RV_HALFLIFE_S = 100.0
MIN_UPDATES = 5
SECONDS_PER_YEAR = 365.25 * 24 * 3600


def precompute(ep):
    """sigma_T at each decision index. NaN until the EWMA has warmed up."""
    spot = np.asarray(ep.spot, dtype="float64")
    age = np.asarray(ep.spot_age_ms, dtype="float64")
    bucket_s = BUCKET_MS / 1000.0

    out = np.full(len(ep), np.nan)
    rv_ewm_var = 0.0
    updates = 0
    prev = -1

    for i in range(len(ep)):
        if age[i] == 0.0 and np.isfinite(spot[i]) and spot[i] > 0.0:
            if prev >= 0:
                dt = (i - prev) * bucket_s
                log_ret = math.log(float(spot[i]) / float(spot[prev]))
                r2_per_sec = log_ret * log_ret / dt
                alpha = 1.0 - math.exp(-dt / RV_HALFLIFE_S)
                rv_ewm_var = alpha * r2_per_sec + (1.0 - alpha) * rv_ewm_var
                updates += 1
            prev = i

        if updates >= MIN_UPDATES:
            ewm_rv = math.sqrt(rv_ewm_var * SECONDS_PER_YEAR)
            tte = max(ep.tte_s(i), 0.0)
            out[i] = ewm_rv * math.sqrt(tte / SECONDS_PER_YEAR)

    return out
