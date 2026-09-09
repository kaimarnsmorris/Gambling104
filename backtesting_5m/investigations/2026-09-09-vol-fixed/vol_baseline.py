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

AND THE TIME THAT GETS FOLDED IN IS NOT tte. The market settles on a 60 s
TWAP, and an average is less variable than its endpoint. Writing the settling
average as A = (1/w) * integral of S over [T-w, T] and taking the geometric
approximation for its log,

    Var[ln A] = (sigma^2 / w^2) * Var[ integral of W ]

and with Var[ integral of W from a to b ] = (b-a)^2 * a + (b-a)^3 / 3 this
collapses to sigma^2 * tau_eff, in two regimes:

    tau >= w    tau_eff = tau - 2w/3      the window has not opened; the whole
                                          average is ahead, but averaging it
                                          costs two thirds of a window
    tau <  w    tau_eff = tau^3 / 3w^2    averaging is under way, so only the
                                          unelapsed tail is still random

They agree at tau = w, both giving w/3, so sigma has no step in it. The effect
is mild early (-7 % at 300 s) and dominant late (-90 % at 10 s), because near
expiry most of what settles the market has already happened.

That sharpening is only safe because `fair.py` centres on E[A] rather than on
a lagged proxy: sigma_T now falls like tau^1.5, so a biased level would be
priced with near-certainty in the last seconds.

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
TWAP_WINDOW_S = 60.0            # Chainlink's lookback, per the 2026-08-14 era


def effective_tte(tau, window=TWAP_WINDOW_S):
    """The time the settling AVERAGE is exposed to, not the time to expiry."""
    tau = max(tau, 0.0)
    if tau >= window:
        return tau - 2.0 * window / 3.0
    return tau ** 3 / (3.0 * window * window)


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
            tau_eff = effective_tte(ep.tte_s(i))
            out[i] = ewm_rv * math.sqrt(tau_eff / SECONDS_PER_YEAR)

    return out
