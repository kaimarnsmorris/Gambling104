"""Step 4 - the forecast regression.

    log RV_hat(t, h) = beta_0(x) + sum_k beta_k(x) * log v_k(t) + bias(x),   x = log dT_h

Each beta_k(.) is a smooth function of the *business-time* horizon, expanded in a small
cubic B-spline basis B_j(x), so the whole model is a linear regression on the J*(K+1)
interaction columns  B_j(x) * [1, log v_1..log v_K].

Two fitting losses are supported:
  * "ls"    - least squares on log RV (fast, closed form, GLS-weighted by horizon)
  * "qlike" - direct minimisation of QLIKE on the variance scale (convex in beta)

`bias(x)` is the multiplicative correction that takes exp(E[log RV]) to E[RV]. We use the
QLIKE-optimal scale, which is exactly the local mean of RV/RV_hat_raw, and also store the
lognormal 0.5*sigma^2 value for reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

LOG_TINY = -300.0


# ------------------------------------------------------------------------- basis
def make_knots(x_lo: float, x_hi: float, n_interior: int = 2) -> np.ndarray:
    """Clamped cubic B-spline knot vector covering [x_lo, x_hi]."""
    inner = np.linspace(x_lo, x_hi, n_interior + 2)[1:-1]
    return np.concatenate([[x_lo] * 4, inner, [x_hi] * 4])


def spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), knots[0], knots[-1])
    return np.asarray(BSpline.design_matrix(x, knots, 3, extrapolate=False).todense())


def n_basis(knots: np.ndarray) -> int:
    return knots.size - 4


# ------------------------------------------------------------------- the model
@dataclass
class Forecaster:
    knots: np.ndarray
    coef: np.ndarray                  # (J, K+1)
    logv_center: np.ndarray           # (K,)
    bias_knots: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bias_coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bias_mean_coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bias_lognormal: np.ndarray = field(default_factory=lambda: np.zeros(0))
    meta: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return self.coef.shape[1] - 1

    @property
    def J(self) -> int:
        return self.coef.shape[0]

    # ---- prediction ----------------------------------------------------
    def betas(self, x: np.ndarray) -> np.ndarray:
        """beta_k(x) for every k, shape (n, K+1) with column 0 the intercept."""
        return spline_basis(np.atleast_1d(x), self.knots) @ self.coef

    def predict_log_raw(self, x: np.ndarray, logv: np.ndarray) -> np.ndarray:
        b = self.betas(x)
        z = np.column_stack([np.ones(len(x)), np.asarray(logv) - self.logv_center])
        return np.einsum("ij,ij->i", b, z)

    def log_bias(self, x: np.ndarray) -> np.ndarray:
        if self.bias_coef.size == 0:
            return np.zeros(np.atleast_1d(x).shape)
        return spline_basis(np.atleast_1d(x), self.bias_knots) @ self.bias_coef

    def log_bias_mean(self, x: np.ndarray) -> np.ndarray:
        if self.bias_mean_coef.size == 0:
            return self.log_bias(x)
        return spline_basis(np.atleast_1d(x), self.bias_knots) @ self.bias_mean_coef

    def predict_rv(self, x: np.ndarray, logv: np.ndarray) -> np.ndarray:
        """Realised variance over the horizon whose business length is exp(x), on the
        QLIKE-optimal scale (the model's primary metric)."""
        return np.exp(self.predict_log_raw(x, logv) + self.log_bias(x))

    def predict_rv_mean(self, x: np.ndarray, logv: np.ndarray) -> np.ndarray:
        """The same forecast rescaled so that mean(RV_hat) matches mean(RV): a literal
        E[RV]. Use this if you need an unbiased expectation rather than the QLIKE
        optimum; see fit_bias for why the two differ."""
        return np.exp(self.predict_log_raw(x, logv) + self.log_bias_mean(x))

    def to_dict(self) -> dict:
        return {
            "knots": self.knots.tolist(),
            "coef": self.coef.tolist(),
            "logv_center": self.logv_center.tolist(),
            "bias_knots": self.bias_knots.tolist(),
            "bias_coef": self.bias_coef.tolist(),
            "bias_mean_coef": np.asarray(self.bias_mean_coef).tolist(),
            "bias_lognormal": np.asarray(self.bias_lognormal).tolist(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Forecaster":
        return cls(np.asarray(d["knots"]), np.asarray(d["coef"]),
                   np.asarray(d["logv_center"]),
                   np.asarray(d.get("bias_knots", [])),
                   np.asarray(d.get("bias_coef", [])),
                   np.asarray(d.get("bias_mean_coef", [])),
                   np.asarray(d.get("bias_lognormal", [])),
                   d.get("meta", {}))


# ------------------------------------------------------------------- fitting
CHUNK = 500_000


def design_rows(x, lv, knots, center):
    """kron(B_row, Z_row) -> (n, J*(K+1))."""
    B = spline_basis(x, knots)
    Z = np.column_stack([np.ones(x.size), lv - center])
    return (B[:, :, None] * Z[:, None, :]).reshape(x.size, B.shape[1] * Z.shape[1])


def _accumulate(xs, logvs, ys, knots, center, weights=None):
    """Normal equations for the interaction design, chunked over rows and horizons."""
    J = n_basis(knots)
    K = logvs[0].shape[1]
    P = J * (K + 1)
    A = np.zeros((P, P))
    b = np.zeros(P)
    n_tot = 0
    for i, (x, lv, y) in enumerate(zip(xs, logvs, ys)):
        w = 1.0 if weights is None else float(weights[i])
        for a in range(0, x.size, CHUNK):
            sl = slice(a, a + CHUNK)
            D = design_rows(x[sl], lv[sl], knots, center)
            A += w * (D.T @ D)
            b += w * (D.T @ y[sl])
        n_tot += x.size
    return A, b, n_tot


def fit_ls(xs, logvs, ys, knots, center, ridge=1e-8, gls_passes=2):
    """Joint least squares on log RV, optionally GLS-weighted by horizon."""
    P = n_basis(knots) * (logvs[0].shape[1] + 1)
    w = None
    coef = None
    for _ in range(max(1, gls_passes)):
        A, b, n = _accumulate(xs, logvs, ys, knots, center, w)
        A.flat[:: P + 1] += ridge * np.trace(A) / P
        coef = np.linalg.solve(A, b)
        # residual variance per horizon -> GLS weights
        w = []
        for x, lv, y in zip(xs, logvs, ys):
            resid = []
            for a in range(0, x.size, CHUNK):
                sl = slice(a, a + CHUNK)
                resid.append(y[sl] - design_rows(x[sl], lv[sl], knots, center) @ coef)
            w.append(1.0 / max(np.var(np.concatenate(resid)), 1e-12))
        w = np.asarray(w)
        w = w / w.mean()
    return coef.reshape(n_basis(knots), logvs[0].shape[1] + 1)


def predict_flat(x, lv, knots, center, coef):
    out = np.empty(x.size)
    flat = np.asarray(coef).ravel()
    for a in range(0, x.size, CHUNK):
        sl = slice(a, a + CHUNK)
        out[sl] = design_rows(x[sl], lv[sl], knots, center) @ flat
    return out


def fit_qlike(xs, logvs, ys_rv, knots, center, coef0, maxiter=60,
              max_rows_per_horizon=250_000, seed=0):
    """Direct QLIKE minimisation (convex in the coefficients).

    Materialises the design, so we subsample each horizon to keep memory bounded; the
    coefficient count is ~45 so a quarter-million rows per horizon is ample.
    """
    J = n_basis(knots)
    K = logvs[0].shape[1]
    P = J * (K + 1)
    rng = np.random.default_rng(seed)
    Ds, RVs = [], []
    for x, lv, rv in zip(xs, logvs, ys_rv):
        if x.size > max_rows_per_horizon:
            sel = rng.choice(x.size, max_rows_per_horizon, replace=False)
            sel.sort()
            x, lv, rv = x[sel], lv[sel], rv[sel]
        Ds.append(design_rows(x, lv, knots, center))
        RVs.append(rv)
    ntot = sum(d.shape[0] for d in Ds)

    def fg(theta):
        f_val = 0.0
        g = np.zeros(P)
        for D, rv in zip(Ds, RVs):
            f = D @ theta
            e = rv * np.exp(-f)
            f_val += np.sum(e - (np.log(rv) - f) - 1.0)
            g += D.T @ (1.0 - e)
        return f_val / ntot, g / ntot

    res = minimize(fg, coef0.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter})
    return res.x.reshape(J, K + 1)


def fit_bias(x_all: np.ndarray, rv: np.ndarray, rv_hat_raw: np.ndarray,
             knots: np.ndarray, n_bins: int = 24):
    """Two multiplicative corrections, both smooth functions of x = log dT.

    `exp(E[log RV])` is not `E[RV]`, and there is no single scale that fixes both things
    a user might want:

      * **QLIKE-optimal**: the scale that minimises QLIKE is exactly the local mean of
        `RV / RV_hat`. This is what the model's own primary metric wants.
      * **Mean-matching**: the scale that makes `mean(RV_hat) == mean(RV)` locally, which
        is what "the autotrader needs E[RV]" literally asks for and what the
        `mean(RV)/mean(RV_hat)` diagnostic measures.

    They differ because realised variance is violently right-skewed, so the mean of the
    ratio is not the ratio of the means - at the 10-second horizon they differ by a
    factor of two. We fit and store both; the model applies the QLIKE-optimal one by
    default and exposes the other through `predict_rv_mean`.

    Note that the binary-market probabilities are invariant to this choice: the tail is
    fitted to `r / sqrt(RV_hat)`, so any monotone rescaling of RV_hat is absorbed by the
    fitted scale parameter.
    """
    ok = np.isfinite(rv) & np.isfinite(rv_hat_raw) & (rv > 0) & (rv_hat_raw > 0)
    x, rvo, rho = x_all[ok], rv[ok], rv_hat_raw[ok]
    ratio = rvo / rho
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(x, edges) - 1, 0, edges.size - 2)
    nb = edges.size - 1
    cnt = np.bincount(idx, minlength=nb).astype(float)
    s_ratio = np.bincount(idx, weights=ratio, minlength=nb)
    s_rv = np.bincount(idx, weights=rvo, minlength=nb)
    s_rh = np.bincount(idx, weights=rho, minlength=nb)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    good = cnt > 50
    bk = make_knots(float(x.min()), float(x.max()), n_interior=2)
    if good.sum() < n_basis(bk):
        # Too few rows to identify a smooth correction (only ever happens on toy
        # inputs). Fall back to a single constant, which is still the right estimator,
        # just without the horizon dependence.
        n_b = n_basis(bk)
        q = np.log(max(float(np.mean(ratio)), 1e-12))
        m_ = np.log(max(float(np.mean(rvo) / max(np.mean(rho), 1e-300)), 1e-12))
        return bk, np.full(n_b, q), np.full(n_b, m_)
    B = spline_basis(ctr[good], bk)
    A = B.T @ (B * cnt[good, None])
    A.flat[:: A.shape[0] + 1] += 1e-6 * np.trace(A) / A.shape[0]

    def _solve(yv):
        return np.linalg.solve(A, B.T @ (yv * cnt[good]))

    y_q = np.log(np.maximum(s_ratio[good] / cnt[good], 1e-12))
    y_m = np.log(np.maximum(s_rv[good] / np.maximum(s_rh[good], 1e-300), 1e-12))
    return bk, _solve(y_q), _solve(y_m)


# -------------------------------------------------------------- rough-vol kernel
def rough_weights(hl_bdays: np.ndarray, H: float) -> np.ndarray:
    """Exponential-mixture approximation of the fractional kernel.

    The registers are EWMAs with rates lam_k = ln2/HL_k. Writing the power-law memory
    kernel as a Bernstein mixture of exponentials, u^{gamma-1} ~ int lam^{-gamma}
    e^{-lam u} dlam, and using geometrically spaced nodes (constant d log lam), the
    quadrature weight on register k is proportional to lam_k^{1-gamma} with
    gamma = H + 1/2. H -> 0 concentrates the weight on the fast registers (rough),
    H = 1/2 gives equal log-spaced weight (long memory / HAR-like).
    """
    lam = np.log(2.0) / np.asarray(hl_bdays, dtype=np.float64)
    w = lam ** (0.5 - H)
    return w / w.sum()
