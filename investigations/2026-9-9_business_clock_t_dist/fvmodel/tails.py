"""The tail of the standardised settlement residual.

    q = ( y* / sqrt(Var_Y) - mu(z) ) / sigma(z),     P(up) = 1 - F_nu(z)(q)

fitted on `(Y - m_Y - eps_bar) / sqrt(Var_Y)` and indexed by **z = log D(n)**, the
business time left in the market, not by the number of seconds. That is the same choice
v2.1 makes for its own horizon distribution and for the same reason: what thins the tail
is the number of shocks accumulated, and one clock second buys an order of magnitude
more shocks in the US macro window than on a Sunday morning.

Each settlement kind gets its own curve. They are genuinely different distributions - a
sixty-second average of a filtered price is a much smoother object than a single print,
and at the short end the Chainlink settlement is dominated by the EWMA's own state
rather than by future returns.

`nu` is floored at 2.5. Below about 2 a Student-t has no variance and the standardisation
the whole construction rests on stops meaning anything; 2.5 keeps a finite variance with
room to spare, and the fitted values only approach it at the very shortest horizons.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

NU_FLOOR = 2.5
NU_CAP = 200.0


@dataclass
class SettlementTail:
    """(nu, mu, sigma) of the standardised residual, as a function of z = log D."""
    z_grid: np.ndarray
    nu: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    family: str = "student_t"
    kind: str = ""
    meta: dict = field(default_factory=dict)

    def params(self, z):
        z = np.asarray(z, dtype=np.float64)
        return (np.interp(z, self.z_grid, self.nu),
                np.interp(z, self.z_grid, self.mu),
                np.interp(z, self.z_grid, self.sigma))

    def prob_up(self, y_star, var_y, z):
        """P(Y > y*) with Y the standardised settlement residual."""
        s = np.sqrt(np.maximum(np.asarray(var_y, dtype=np.float64), 1e-300))
        q = np.asarray(y_star, dtype=np.float64) / s
        if self.family == "normal":
            return 1.0 - stats.norm.cdf(q)
        nu, mu, sg = self.params(z)
        return 1.0 - stats.t.cdf(q, df=nu, loc=mu, scale=sg)

    def to_dict(self):
        return {"z_grid": self.z_grid.tolist(), "nu": self.nu.tolist(),
                "mu": self.mu.tolist(), "sigma": self.sigma.tolist(),
                "family": self.family, "kind": self.kind, "meta": self.meta}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["z_grid"]), np.asarray(d["nu"]), np.asarray(d["mu"]),
                   np.asarray(d["sigma"]), d.get("family", "student_t"),
                   d.get("kind", ""), d.get("meta", {}))

    @classmethod
    def normal(cls, kind: str = ""):
        g = np.linspace(-16.0, -2.0, 3)
        return cls(g, np.full(3, 1e6), np.zeros(3), np.ones(3), "normal", kind)


def fit_tail(z: np.ndarray, resid: np.ndarray, kind: str = "",
             n_bins: int = 9, min_n: int = 2000, boot: int = 25,
             seed: int = 0, max_fit: int = 40_000,
             max_boot_fit: int = 8_000) -> SettlementTail:
    """Bin the standardised residuals by business time and fit a Student-t in each bin.

    The bins are quantiles of z, so each carries a comparable number of observations.
    `boot` resamples per bin give the confidence band on nu that the report plots; the
    resampling is over contiguous blocks of 200 rows, because neighbouring synthetic
    markets overlap. Each fit is a Nelder-Mead maximum likelihood, so the sample sizes
    are capped: a location-scale t on 40,000 draws is already pinned far tighter than
    the sampling variation the bootstrap is measuring.
    """
    from rvforecast.distribution import fit_student_t

    m = np.isfinite(z) & np.isfinite(resid)
    z, resid = z[m], resid[m]
    order = np.argsort(z)
    z, resid = z[order], resid[order]
    edges = np.quantile(z, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    rng = np.random.default_rng(seed)

    zs, nus, mus, sgs, ns, lo, hi = [], [], [], [], [], [], []
    for k in range(n_bins):
        sel = (z > edges[k]) & (z <= edges[k + 1])
        if sel.sum() < min_n:
            continue
        r = resid[sel]
        nu, mu, sg = fit_student_t(r, max_n=max_fit, seed=seed)
        if not np.isfinite(nu):
            continue
        zs.append(float(np.median(z[sel])))
        nus.append(float(np.clip(nu, NU_FLOOR, NU_CAP)))
        mus.append(float(mu))
        sgs.append(float(sg))
        ns.append(int(sel.sum()))
        if boot:
            blk = 200
            nb = max(min(r.size, max_boot_fit) // blk, 2)
            top = max(r.size - blk, 1)
            bs = []
            for _ in range(boot):
                pick = rng.integers(0, top, nb)
                idx = (pick[:, None] + np.arange(blk)[None, :]).ravel()
                idx = idx[idx < r.size]
                v = fit_student_t(r[idx], max_n=max_boot_fit, seed=seed)[0]
                if np.isfinite(v):
                    bs.append(min(max(v, NU_FLOOR), NU_CAP))
            lo.append(float(np.quantile(bs, 0.05)) if bs else np.nan)
            hi.append(float(np.quantile(bs, 0.95)) if bs else np.nan)
        else:
            lo.append(np.nan)
            hi.append(np.nan)

    if not zs:
        return SettlementTail.normal(kind)
    return SettlementTail(np.array(zs), np.array(nus), np.array(mus), np.array(sgs),
                          "student_t", kind,
                          {"n": ns, "nu_lo": lo, "nu_hi": hi,
                           "sd": float(np.std(resid)),
                           "kurtosis": float(((resid - resid.mean()) ** 4).mean()
                                             / max(resid.var(), 1e-300) ** 2)})
