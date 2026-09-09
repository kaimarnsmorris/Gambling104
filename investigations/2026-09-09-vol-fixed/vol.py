# DELIBERATE LOCAL SHADOW: this vol.py is NOT models/normal_qq/vol.py. It is
# the fitted lookup table this investigation exists to evaluate; the
# investigation-first block resolution order picking it over the model's
# EWMA baseline is the point, not an oversight. Do not "tidy" it away.
"""sigma -- CALIBRATED. The empirical term structure of the forecast error.

THE CONTROL THIS REPLACES, `vol_baseline.py`, builds sigma_T from an EWMA of
realised spot variance and then shrinks it by a TWAP-averaging correction
`tau_eff`. Measured against realised movement (`fit_sigma.py`), that number is
between 2x and 8x too small across the whole window and collapses toward zero
in the last minute, where the correction is most aggressive. That collapse is
what pins `link.py` at p = 1.000.

WHAT THIS BLOCK RETURNS INSTEAD is the measured dispersion of the model's own
forecast error,

    r = ln(A_T / s_t)

where A_T is the settling value and s_t is `fair.py`'s E[A_T | F_t] -- the same
level `f.standardise` divides by. So sigma_T is exactly the sd of the thing it
is asked to standardise, per time-to-expiry bucket, and nothing else. The table
lives in `sigma_fit.json`, written by `fit_sigma.py` from the FIT DAYS ONLY
(2026-08-19..21); the days this is scored on never enter it.

A TABLE, NOT A FORMULA, on purpose. The suspicion under investigation is that
the baseline's analytic term structure is wrong; answering it with a different
analytic term structure would beg the question. Interpolation is linear in
log sigma against log tau, which is exact for any power law and honest about
everything else.

THE FLOOR IS THE POINT. Below the shortest tabulated bucket sigma is HELD, not
extrapolated. The realised error does not vanish as tau -> 0: it settles onto a
floor of about 0.9 bp, roughly $7, because the market settles on a Chainlink
TWAP while this model watches a venue composite, and the residual basis between
them does not shrink with elapsed time. Extrapolating a decaying term structure
through that floor is precisely the bug being fixed, so the block refuses to do
it. (On the ORIGINAL, uncorrected BTC/USDT panel this floor read ~7 bp, eight
times larger, because it was dominated by a ~+43 USD level bias in the spot
build rather than by anything the market did. The table is panel-conditional:
refit it whenever the panel is rebuilt.)

WARM-UP IS PRESERVED DELIBERATELY. This sigma needs no spot history -- it is
unconditional in tau. It nonetheless returns NaN until MIN_UPDATES distinct
spot observations have landed, exactly as the baseline does, so that the two
blocks decline to quote on the same indices. The variable under test is the
LEVEL of sigma; leaving the quoting window free to move as well would confound
them.

AND IT READS NO PRE-OPEN HISTORY, for the same reason. Episodes now carry a
warm-up region and `vol_baseline.py` burns its EWMA in across it; this block
has no state to burn in, so there is nothing here for the region to change.
The MIN_UPDATES gate deliberately still counts IN-WINDOW observations in all
three arms -- see the note in `vol_baseline.py` for the measurement that says
it costs 0.34 % of the window to leave it alone.

WHAT THE WARM-UP DID CHANGE, indirectly and importantly: the table below is
refitted against a `fair.py` whose basis now runs a 180 s halflife burnt in
over 900 s of history. That model's forecast error is a different, smaller
quantity than the cold-start one's -- the near-expiry floor fell from ~0.9 bp
to ~0.26 bp -- so the numbers in `sigma_fit.json` are not comparable across
that change. The table is conditional on the panel AND on the fair block;
refit it whenever either moves.

WHAT THIS BLOCK CANNOT FIX, and the report must not pretend otherwise: it can
only widen around a level error, never move it. On the corrected BTC/USD panel
the residual bias is small (-0.71 bp on the fit half) and changes sign across
the window, so there is little left to absorb; on the uncorrected panel it was
-5.6 bp at every tau and negative on 827 of 827 markets, and this block's
mean-absolute scale silently swallowed it into sigma's magnitude. That was the
conservative thing to do with a bias that could not be removed, and it was not
the same as removing it. If a persistent bias ever returns, it belongs in
`fair.py` as a drift term, not here as extra width.

WHAT THIS BLOCK LEAVES ON THE TABLE: it is unconditional in tau, and the
baseline's EWMA -- worthless-looking at corr 0.0043 on the biased panel -- is
worth corr 0.3805 on the corrected one. A block that kept that per-market
conditioning and pinned its LEVEL to this table should beat both this and
`vol_flat.py`. It has not been built or measured, so it is not claimed.
"""
import bisect
import json
import math
import os

import numpy as np

MIN_UPDATES = 5         # matched to vol_baseline.py, so warm-up is identical
FIT_FILE = "sigma_fit.json"
KEY = "table"           # which fitted object in the file this block reads


def _find_fit():
    """`sigma_fit.json`, beside this file or above it.

    A run freezes a copy of this block into its own folder; that copy is
    provenance, never executed. What IS executed is this file where it lives,
    or the copy a variant folder was assembled from -- and `run.py` puts the
    fit file next to that copy. The walk upward is the safety net.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        candidate = os.path.join(here, FIT_FILE)
        if os.path.exists(candidate):
            return candidate
        here = os.path.dirname(here)
    raise FileNotFoundError(
        f"{FIT_FILE} not found beside {__file__} or above it. Run "
        "`python fit_sigma.py` to write it; this block is a lookup, not a "
        "model, and declines to invent a term structure.")


def _load():
    with open(_find_fit()) as fh:
        fit = json.load(fh)
    tau = [float(t) for t in fit[KEY]["tte"]]
    sig = [float(s) for s in fit[KEY]["sigma"]]
    order = sorted(range(len(tau)), key=lambda k: tau[k])
    tau = [tau[k] for k in order]
    sig = [sig[k] for k in order]
    return tau, [math.log(max(s, 1e-12)) for s in sig], [math.log(t) for t in tau]


TAU, LOG_SIGMA, LOG_TAU = _load()


def sigma_of_tau(tau):
    """sigma_T at time-to-expiry `tau`, by log-log interpolation on the table.

    Held flat outside the tabulated range at BOTH ends: the short end because
    the error floor is real, the long end because 300 s is the whole market.
    """
    if tau <= TAU[0]:
        return math.exp(LOG_SIGMA[0])
    if tau >= TAU[-1]:
        return math.exp(LOG_SIGMA[-1])
    j = bisect.bisect_right(TAU, tau)
    lt = math.log(tau)
    w = (lt - LOG_TAU[j - 1]) / (LOG_TAU[j] - LOG_TAU[j - 1])
    return math.exp(LOG_SIGMA[j - 1] + w * (LOG_SIGMA[j] - LOG_SIGMA[j - 1]))


def precompute(ep):
    """sigma_T at each decision index. NaN until the spot feed has warmed up."""
    spot = np.asarray(ep.spot, dtype="float64")
    age = np.asarray(ep.spot_age_ms, dtype="float64")

    out = np.full(len(ep), np.nan)
    updates = 0
    prev = -1

    for i in range(len(ep)):
        if age[i] == 0.0 and np.isfinite(spot[i]) and spot[i] > 0.0:
            if prev >= 0:
                updates += 1
            prev = i
        if updates >= MIN_UPDATES:
            out[i] = sigma_of_tau(ep.tte_s(i))

    return out
