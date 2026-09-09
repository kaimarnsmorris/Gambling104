"""sigma -- FLAT. One constant total vol, scaled by sqrt(tte). The control.

This block exists to separate a LEVEL error from a TERM-STRUCTURE error. It is
`vol.py`'s fit with one constraint added: the exponent is pinned at 0.5, so
sigma_T = k * sqrt(tau) with no TWAP correction, no floor, no shape. `k` is
fitted by the same weighted regression on the same fit days, and lives in the
same `sigma_fit.json` under `"flat"`.

READ THE THREE ARMS TOGETHER:

  * if `vol_flat.py` recovers most of what `vol_baseline.py` loses, the
    baseline's problem is that sigma is simply too small everywhere, and
    `tau_eff` is a side issue;
  * if only `vol.py`'s measured term structure recovers it, the problem is the
    SHAPE -- `tau_eff` crushing sigma toward zero near expiry -- and the level
    at 300 s was never the complaint.

Being a control, it is allowed to be wrong in a known way, and it is: sqrt(tau)
goes to zero at expiry, so this arm reintroduces the saturation `vol.py`'s
floor is there to prevent. That is the experiment, not an oversight.

Warm-up is MIN_UPDATES spot observations, matched to both neighbours so all
three arms quote on the same indices.
"""
import json
import math
import os

import numpy as np

MIN_UPDATES = 5         # matched to vol_baseline.py and vol.py
FIT_FILE = "sigma_fit.json"


def _find_fit():
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        candidate = os.path.join(here, FIT_FILE)
        if os.path.exists(candidate):
            return candidate
        here = os.path.dirname(here)
    raise FileNotFoundError(
        f"{FIT_FILE} not found beside {__file__} or above it. Run "
        "`python fit_sigma.py` to write it.")


with open(_find_fit()) as _fh:
    _FIT = json.load(_fh)

K = float(_FIT["flat"]["k"])            # sigma_T = K * sqrt(tau)


def sigma_of_tau(tau):
    return K * math.sqrt(max(tau, 0.0))


def precompute(ep):
    """sigma_T = K * sqrt(tte). NaN until the spot feed has warmed up."""
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
