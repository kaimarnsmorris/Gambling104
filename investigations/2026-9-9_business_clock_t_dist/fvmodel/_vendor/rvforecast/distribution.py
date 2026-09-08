"""Step 5 - the conditional distribution of the return over the remaining horizon.

The autotrader needs P(S_{t+h} > K | S_t), so a variance forecast is not enough. We
standardise realised returns by the model's own forecast,

    z = r(t, h) / sqrt(RV_hat(t, h)),

and fit a location-scale Student-t to z as a function of the *business-time* horizon
x = log dT_h (kurtosis decays as shocks accumulate, and it is business time, not clock
time, that counts the shocks). Optionally the degrees of freedom are additionally
conditioned on a vol-of-vol proxy, log v_fast - log v_slow.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import minimize


def _nll_t(theta, z):
    log_nu_m2, loc, log_scale = theta
    nu = 2.0 + np.exp(log_nu_m2)
    sc = np.exp(log_scale)
    return -np.sum(stats.t.logpdf(z, df=nu, loc=loc, scale=sc))


def fit_student_t(z: np.ndarray, max_n: int = 400_000, seed: int = 0):
    """ML fit of a location-scale Student-t. Returns (nu, loc, scale)."""
    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size > max_n:
        rng = np.random.default_rng(seed)
        z = z[rng.choice(z.size, max_n, replace=False)]
    if z.size < 100:
        return np.nan, np.nan, np.nan
    s0 = np.std(z)
    x0 = np.array([np.log(3.0), 0.0, np.log(max(s0 * 0.7, 1e-6))])
    res = minimize(_nll_t, x0, args=(z,), method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-4})
    a, loc, ls = res.x
    return float(2.0 + np.exp(a)), float(loc), float(np.exp(ls))


def fit_skew_two_piece(z: np.ndarray, nu: float):
    """Two-piece scale: separate scale left and right of the location."""
    z = z[np.isfinite(z)]
    med = float(np.median(z))
    lo, hi = z[z < med], z[z >= med]
    # match the half-variance of a symmetric t on each side
    sl = float(np.sqrt(np.mean((lo - med) ** 2))) if lo.size else np.nan
    sr = float(np.sqrt(np.mean((hi - med) ** 2))) if hi.size else np.nan
    return med, sl, sr


@dataclass
class Distribution:
    """Student-t tails indexed by business-time horizon x = log dT."""
    x_grid: np.ndarray                # (m,) ascending
    nu: np.ndarray                    # (m,)
    loc: np.ndarray                   # (m,)
    scale: np.ndarray                 # (m,)
    kurt_slope: float = 0.0           # d log(nu-2) / d volvol_z, 0 disables conditioning
    volvol_center: float = 0.0
    volvol_sd: float = 1.0
    family: str = "student_t"         # or "normal"
    meta: dict = field(default_factory=dict)

    def _interp(self, a, x):
        return np.interp(np.asarray(x, dtype=np.float64), self.x_grid, a)

    def params(self, x, volvol=None):
        nu = self._interp(self.nu, x)
        loc = self._interp(self.loc, x)
        sc = self._interp(self.scale, x)
        if volvol is not None and self.kurt_slope != 0.0:
            zz = (np.asarray(volvol, dtype=np.float64) - self.volvol_center) / self.volvol_sd
            nu = 2.0 + (nu - 2.0) * np.exp(self.kurt_slope * zz)
            nu = np.clip(nu, 2.05, 200.0)
        return nu, loc, sc

    def cdf(self, z, x, volvol=None):
        if self.family == "normal":
            return stats.norm.cdf(z)
        nu, loc, sc = self.params(x, volvol)
        return stats.t.cdf(z, df=nu, loc=loc, scale=sc)

    def prob_up(self, log_moneyness, rv_hat, x, volvol=None):
        """P(S_{t+h} > K) where log_moneyness = log(K / S_t)."""
        z = np.asarray(log_moneyness, dtype=np.float64) / np.sqrt(np.maximum(rv_hat, 1e-300))
        return 1.0 - self.cdf(z, x, volvol)

    def to_dict(self):
        return {"x_grid": self.x_grid.tolist(), "nu": self.nu.tolist(),
                "loc": self.loc.tolist(), "scale": self.scale.tolist(),
                "kurt_slope": self.kurt_slope, "volvol_center": self.volvol_center,
                "volvol_sd": self.volvol_sd, "family": self.family, "meta": self.meta}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["x_grid"]), np.asarray(d["nu"]), np.asarray(d["loc"]),
                   np.asarray(d["scale"]), d.get("kurt_slope", 0.0),
                   d.get("volvol_center", 0.0), d.get("volvol_sd", 1.0),
                   d.get("family", "student_t"), d.get("meta", {}))

    @classmethod
    def normal(cls):
        g = np.linspace(-11.0, -1.0, 3)
        return cls(g, np.full(3, 1e6), np.zeros(3), np.ones(3), family="normal")
