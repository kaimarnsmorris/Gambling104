"""v2 section 5 (original brief section 11) - forward variance, the return ACF kernel
and TWAP uncertainty.

The autotrader does not only need the integrated variance over a fixed grid of horizons.
It needs, for an arbitrary window of the next N seconds, the variance of the *TWAP* of
second-by-second prices over that window, including part-way through when some of the
window is already observed. That changes the link function, not the registers.

1. Forward-variance link
------------------------
v1 predicts integrated variance directly, one number per horizon:

    log IV_hat(t, h) = beta_0(x) + sum_k beta_k(x) log v_k(t),   x = log dT_h

v2 predicts a *density* and integrates it:

    xi(t, u)  = s(t+u)/86400 * exp( b_0(dT_u) + sum_k b_k(dT_u) log v_k(t) )
    IV(t, h)  = sum_{u=1..h} xi(t, u)

with `dT_u` the business time elapsed from t to t+u. This is positive and monotone by
construction, and it hands you the per-second forward variance for free.

**The integral collapses.** Because `dT_u - dT_{u-1}` is exactly `s(t+u)/86400`, the sum
telescopes into a pure business-time integral:

    IV(t, h) = int_0^{D} exp( g(log tau) ) dtau ,     D = dT_h
             = int_{-inf}^{log D} exp( g(z) + z ) dz

where `g(z) = b_0(z) + sum_k b_k(z) log v_k(t)`. The clock time drops out entirely: the
whole term structure is one univariate integral in business time whose upper limit is the
only thing the horizon controls. That makes the fit cheap - a fixed log-tau quadrature
grid, one exponential per (node, row), and a cumulative sum - and it makes the identity
`IV(t, h) = int xi` exact rather than approximate.

The direct v1 fit is kept as a comparison row; at the grid horizons the two should agree
closely, and the report checks that they do.

2. The 1-second return ACF kernel
---------------------------------
A TWAP is a weighted sum of consecutive 1-second returns, so its variance depends on the
autocovariance of those returns, not only on their variances.

Both the original brief ("expect negative rho(1), bid-ask bounce") and NOTES D12 ("lag-1
autocorrelation of 1s returns is strongly negative, so 1s RV overstates integrated
variance") predict a negative lag-1. **On this instrument at this sampling frequency it
is positive**: rho(1) = +0.081 measured directly on raw 1-second last-trade bars, and
+0.085 on consecutive non-zero price changes. At roughly fifty prints a second, the
bounce averages out *inside* the one-second bar and what survives to the next bar is
order-splitting continuation. Bounce lives at tick frequency, not at second frequency.

Two consequences, both of which the report states rather than buries:

  * D12's inference is backwards - a positive lag-1 means 1-second RV *understates* the
    variance of lower-frequency returns, not overstates it;
  * the TWAP cross term is positive, so the fitted kernel makes a TWAP *more* volatile
    than the independent-increments formula, not less. The rho == 0 ablation therefore
    under-predicts, which is the direction QLIKE punishes hardest.

3. TWAP variance
----------------
For a window of N seconds of which the first m are already observed, the unknown part is
`sum_{i=m+1..N} w_i r_i` with `w_i = (N - i + 1)/N`, so

    Var = sum_i w_i^2 xi_i + 2 sum_{i<j} w_i w_j rho(j-i) sqrt(xi_i xi_j)

The double sum is over a band of width 120 s (the kernel is zero beyond that), so it
costs O(N * 120) rather than O(N^2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .config import SEC_PER_DAY, SEC_PER_YEAR
from .regression import make_knots, n_basis, spline_basis

MAX_LAG = 120


# ============================================================ the forward-variance fit
def quad_grid(z_lo: float, z_hi: float, n: int = 240):
    """Trapezoid nodes and weights for int exp(g(z) + z) dz on a log-tau grid.

    The lower limit is -inf in principle; in practice `z_lo` is set below the smallest
    business second in the sample, where the integrand is ~exp(z) and contributes
    nothing measurable.
    """
    z = np.linspace(z_lo, z_hi, n)
    w = np.full(n, z[1] - z[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    return z, w


@dataclass
class ForwardModel:
    """b_k(.) as a cubic B-spline in z = log dT, plus the quadrature grid."""
    knots: np.ndarray
    coef: np.ndarray                  # (J, K+1)
    logv_center: np.ndarray           # (K,)
    z_grid: np.ndarray
    w_grid: np.ndarray
    bias_knots: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bias_coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bias_mean_coef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    meta: dict = field(default_factory=dict)

    # ---- core -------------------------------------------------------------
    def _theta(self) -> np.ndarray:
        """B(z_grid) @ coef -> (M, K+1)."""
        return spline_basis(self.z_grid, self.knots) @ self.coef

    def density(self, z: np.ndarray, logv: np.ndarray) -> np.ndarray:
        """exp(g(z)) - the variance rate per business day at business age exp(z)."""
        B = spline_basis(np.atleast_1d(z), self.knots)
        Z = np.concatenate([[1.0], np.asarray(logv) - self.logv_center])
        return np.exp((B @ self.coef) @ Z)

    def cumulative(self, logv: np.ndarray) -> np.ndarray:
        """Running integral of exp(g(z)+z) over the quadrature grid, for one row."""
        Z = np.concatenate([[1.0], np.asarray(logv) - self.logv_center])
        f = np.exp(self._theta() @ Z + self.z_grid)
        return np.cumsum(self.w_grid * f)

    def iv_raw(self, x, logv: np.ndarray) -> np.ndarray:
        """IV before the bias correction, at x = log dT (scalar or array)."""
        cum = self.cumulative(logv)
        return np.interp(np.atleast_1d(x), self.z_grid, cum)

    def log_bias(self, x) -> np.ndarray:
        if self.bias_coef.size == 0:
            return np.zeros(np.atleast_1d(x).shape)
        return spline_basis(np.atleast_1d(x), self.bias_knots) @ self.bias_coef

    def log_bias_mean(self, x) -> np.ndarray:
        if self.bias_mean_coef.size == 0:
            return self.log_bias(x)
        return spline_basis(np.atleast_1d(x), self.bias_knots) @ self.bias_mean_coef

    def predict_rv(self, x, logv: np.ndarray) -> np.ndarray:
        return self.iv_raw(x, logv) * np.exp(self.log_bias(x))

    # ---- the forward curve ------------------------------------------------
    def forward_curve_business(self, dT_cum: np.ndarray,
                               logv: np.ndarray) -> np.ndarray:
        """xi for a sequence of seconds whose cumulative business ages are `dT_cum`.

        `dT_cum[u]` is the business time from the forecast origin to the end of second
        u (u = 0 is the first future second). The increments are taken of the
        **bias-corrected** integral, so that

            sum(xi over the first h seconds) == predict_rv(log dT_h)

        holds exactly. Differencing the raw integral instead would silently drop the
        correction that turns exp(E[log RV]) into E[RV] - a factor of 3.5 at a
        ten-second horizon, falling to 1.2 by five minutes - and every TWAP variance
        built from the curve would inherit it.
        """
        cum = self.cumulative(logv)
        z = np.log(np.maximum(dT_cum, 1e-12))
        g = np.interp(z, self.z_grid, cum) * np.exp(self.log_bias(z))
        return np.maximum(np.diff(np.concatenate([[0.0], g])), 0.0)

    def to_dict(self) -> dict:
        return {"knots": self.knots.tolist(), "coef": self.coef.tolist(),
                "logv_center": self.logv_center.tolist(),
                "z_grid": self.z_grid.tolist(), "w_grid": self.w_grid.tolist(),
                "bias_knots": np.asarray(self.bias_knots).tolist(),
                "bias_coef": np.asarray(self.bias_coef).tolist(),
                "bias_mean_coef": np.asarray(self.bias_mean_coef).tolist(),
                "meta": self.meta}

    @classmethod
    def from_dict(cls, d) -> "ForwardModel":
        return cls(np.asarray(d["knots"]), np.asarray(d["coef"]),
                   np.asarray(d["logv_center"]), np.asarray(d["z_grid"]),
                   np.asarray(d["w_grid"]),
                   np.asarray(d.get("bias_knots", [])),
                   np.asarray(d.get("bias_coef", [])),
                   np.asarray(d.get("bias_mean_coef", [])),
                   d.get("meta", {}))


def _pack_rows(xs: dict, logv: np.ndarray, ys: dict, horizons, sel: np.ndarray,
               max_rows: int = 150_000, seed: int = 0):
    """One row per forecast origin, with all horizons attached.

    The forward fit shares a single quadrature per origin across every horizon, so the
    design is organised by origin rather than by (origin, horizon) pair.
    """
    rng = np.random.default_rng(seed)
    ok = sel.copy()
    for h in horizons:
        ok &= ys[h]["ok"]
    idx = np.flatnonzero(ok)
    if idx.size > max_rows:
        take = rng.choice(idx.size, max_rows, replace=False)
        take.sort()
        idx = idx[take]
    X = np.column_stack([xs[h][idx] for h in horizons])          # (N, H)
    Y = np.column_stack([ys[h]["rv"][idx].astype(np.float64) for h in horizons])
    return idx, X, Y, logv[idx]


def fit_forward(xs, logv, targets, horizons, sel, n_knots: int = 2,
                max_rows: int = 80_000, n_nodes: int = 200, loss: str = "ls",
                maxiter: int = 300, seed: int = 0, ridge: float = 1e-6,
                verbose: bool = False) -> ForwardModel:
    """Fit b(.) so that the integrated density matches ex-post RV at every horizon.

    One row per forecast origin with all eleven horizons attached, because the
    quadrature is shared across horizons. The gradient is analytic: with

        IV(n,h) = sum_{j <= j0(n,h)} w_j F(j,n) + frac(n,h) * w_{j0+1} F(j0+1,n)

    and F(j,n) = exp( (B_j . coef) . Z_n + z_j ), the derivative with respect to
    coef[j', k] is  sum_j B[j,j'] * w_j F(j,n) * S(j,n) * Z[n,k], where S(j,n) collects
    the per-horizon loss derivatives whose upper limit reaches node j. S is built by a
    difference-and-cumulative-sum trick so the whole gradient costs one (M x N) pass.
    """
    idx, X, Y, LV = _pack_rows(xs, logv, targets, horizons, sel, max_rows, seed)
    N, H = X.shape
    K = LV.shape[1]
    center = LV.mean(0)
    Z = np.ascontiguousarray(np.column_stack([np.ones(N), LV - center]))   # (N, K+1)

    z_lo = float(X.min()) - 3.0
    z_hi = float(X.max()) + 0.05
    zg, wg = quad_grid(z_lo, z_hi, n_nodes)
    knots = make_knots(float(np.quantile(X, 0.001)), float(np.quantile(X, 0.999)),
                       n_knots)
    J = n_basis(knots)
    B = np.ascontiguousarray(spline_basis(zg, knots))                      # (M, J)
    M = zg.size
    dz = zg[1] - zg[0]

    pos = (X - z_lo) / dz
    j0 = np.clip(np.floor(pos).astype(np.int64), 0, M - 2)                 # (N, H)
    frac = np.clip(pos - j0, 0.0, 1.0)
    rows = np.arange(N)[:, None]
    flat1 = ((j0 + 1) * N + rows).ravel()
    flat2 = ((j0 + 2) * N + rows).ravel()
    nflat = (M + 2) * N

    logY = np.log(Y)
    ntot = float(N * H)
    ez = np.exp(zg)

    def _forward(theta):
        F = np.exp(B @ theta.reshape(J, K + 1) @ Z.T)                      # (M, N)
        F *= (wg * ez)[:, None]
        C = np.cumsum(F, axis=0)
        lo = C[j0, rows]
        hi = C[j0 + 1, rows]
        return lo + frac * (hi - lo), F

    def fg(theta):
        IV, F = _forward(theta)
        IV = np.maximum(IV, 1e-300)
        if loss == "ls":
            resid = np.log(IV) - logY
            val = float(np.sum(resid ** 2)) / ntot
            c = 2.0 * resid / IV / ntot
        else:
            ratio = Y / IV
            val = float(np.sum(ratio - np.log(ratio) - 1.0)) / ntot
            c = (1.0 - ratio) / IV / ntot
        d = np.zeros(nflat)
        d[:N] = c.sum(axis=1)
        np.add.at(d, flat1, ((frac - 1.0) * c).ravel())
        np.add.at(d, flat2, (-frac * c).ravel())
        S = np.cumsum(d[: M * N].reshape(M, N), axis=0)
        S *= F
        grad = B.T @ (S @ Z)
        if ridge:
            val += ridge * float(theta @ theta)
            grad = grad + 2.0 * ridge * theta.reshape(J, K + 1)
        return val, grad.ravel()

    lvl = float(np.mean(logY - X))
    theta0 = np.zeros((J, K + 1))
    theta0[:, 0] = lvl
    theta0[:, 1:] = 1.0 / K
    res = minimize(fg, theta0.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxcor": 30,
                            "disp": bool(verbose)})
    coef = res.x.reshape(J, K + 1)
    fm = ForwardModel(knots, coef, center, zg, wg,
                      meta={"loss": loss, "n_rows": int(N), "n_nodes": int(n_nodes),
                            "n_knots": int(n_knots), "obj": float(res.fun),
                            "nit": int(res.nit), "success": bool(res.success)})

    IV, _ = _forward(res.x)
    from .regression import fit_bias
    bk, bc, bm = fit_bias(X.ravel(), Y.ravel(), np.maximum(IV.ravel(), 1e-300), knots)
    fm.bias_knots, fm.bias_coef, fm.bias_mean_coef = bk, bc, bm
    return fm


def predict_forward_rv(fm: ForwardModel, x: np.ndarray, logv: np.ndarray) -> np.ndarray:
    """Vectorised IV over many rows (each row has its own logv)."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    logv = np.atleast_2d(logv)
    B = spline_basis(fm.z_grid, fm.knots)
    Th = B @ fm.coef                                             # (M, K+1)
    out = np.empty(x.size)
    ez = np.exp(fm.z_grid)
    # one (n_nodes x step) exponential per chunk; 100k keeps that under ~160 MB and is
    # an order of magnitude faster than a small chunk on this shape
    step = 100_000
    for a in range(0, x.size, step):
        sl = slice(a, min(a + step, x.size))
        Z = np.column_stack([np.ones(logv[sl].shape[0]),
                             logv[sl] - fm.logv_center])
        F = np.exp(Th @ Z.T) * ez[:, None]
        C = np.cumsum(F * fm.w_grid[:, None], axis=0)
        pos = (x[sl] - fm.z_grid[0]) / (fm.z_grid[1] - fm.z_grid[0])
        j0 = np.clip(np.floor(pos).astype(np.int64), 0, fm.z_grid.size - 2)
        fr = np.clip(pos - j0, 0.0, 1.0)
        rows = np.arange(j0.size)
        out[sl] = C[j0, rows] + fr * (C[j0 + 1, rows] - C[j0, rows])
    return out * np.exp(fm.log_bias(x))


# ==================================================================== ACF kernel
def acf_kernel(r: np.ndarray, valid: np.ndarray, s: np.ndarray, mask: np.ndarray = None,
               max_lag: int = MAX_LAG, chunk: int = 4_000_000) -> dict:
    """rho(l), l = 1..max_lag, of the deseasonalised 1-second returns e = r / sqrt(s).

    Invalid seconds are set to zero, which drops them from every product they would have
    entered; with 0.013% of returns invalid on this sample the effect on the
    normalisation is far below the estimation error.
    """
    n = r.size
    e = np.zeros(n, dtype=np.float64)
    m = valid if mask is None else (valid & mask)
    e[m] = r[m] / np.sqrt(np.maximum(s[m], 1e-300))
    den = float(e @ e)
    cnt = int(m.sum())
    num = np.empty(max_lag)
    for l in range(1, max_lag + 1):
        num[l - 1] = float(e[l:] @ e[:-l])
    rho = num / max(den, 1e-300)
    se = 1.0 / np.sqrt(max(cnt, 1))
    return {"lag": np.arange(1, max_lag + 1), "rho": rho, "se": np.full(max_lag, se),
            "n": cnt, "var": den / max(cnt, 1)}


def acf_by_group(r, valid, s, groups: dict, max_lag: int = MAX_LAG) -> dict:
    return {k: acf_kernel(r, valid, s, mask=v, max_lag=max_lag)
            for k, v in groups.items()}


def kernel_from_table(rho: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    out = np.zeros(max_lag + 1)
    out[0] = 1.0
    out[1:min(max_lag, rho.size) + 1] = rho[:max_lag]
    return out


# ==================================================================== TWAP variance
def twap_weights(N: int, m: int = 0) -> np.ndarray:
    """w_i = (N - i + 1)/N for the unobserved seconds i = m+1..N (1-based)."""
    i = np.arange(m + 1, N + 1, dtype=np.float64)
    return (N - i + 1.0) / N


def twap_variance_from_xi(xi: np.ndarray, rho: np.ndarray, N: int,
                          m: int = 0) -> tuple:
    """Var of the remaining TWAP contribution, and the per-second contributions.

    `xi` holds the forward variance of each of the N - m unobserved seconds, `rho` the
    autocorrelation kernel with rho[0] = 1.
    """
    w = twap_weights(N, m)
    n = w.size
    if n == 0:
        return 0.0, np.zeros(0)
    sd = np.sqrt(np.maximum(xi[:n], 0.0))
    a = w * sd
    var = float(a @ a)
    contrib = a * a
    L = min(len(rho) - 1, n - 1)
    for l in range(1, L + 1):
        if rho[l] == 0.0:
            continue
        c = 2.0 * rho[l] * float(a[l:] @ a[:-l])
        var += c
        # split the cross term evenly over the two seconds involved, so the
        # contributions still sum to the total
        half = rho[l] * a[l:] * a[:-l]
        contrib[l:] += half
        contrib[:-l] += half
    return max(var, 0.0), contrib


def twap_realised(close: np.ndarray, i0: int, N: int) -> tuple:
    """(log(TWAP/S_t), S_t) for the window of N seconds after index i0."""
    s_t = close[i0]
    seg = close[i0 + 1:i0 + N + 1]
    return float(np.log(seg.mean() / s_t)), float(s_t)


def forward_curves_batch(fm, logv: np.ndarray, cum: np.ndarray, grid_idx: np.ndarray,
                         idx: np.ndarray, n_rem: int, step: int = 4000) -> np.ndarray:
    """xi for the `n_rem` seconds after every origin in `idx`, shape (len(idx), n_rem).

    The batched twin of `ForwardModel.forward_curve_business`: same integral, same bias
    correction inside it, evaluated for many origins at once by cumulating the density
    over the z grid once per batch. `cum` is the register bank's cumulative business
    time and `grid_idx` maps a sampling-grid row to its second.
    """
    from .regression import spline_basis
    Bz = spline_basis(fm.z_grid, fm.knots) @ fm.coef              # (M, K+1)
    out = np.empty((idx.size, n_rem))
    dz = fm.z_grid[1] - fm.z_grid[0]
    ez = np.exp(fm.z_grid) * fm.w_grid
    for a in range(0, idx.size, step):
        ii = idx[a:a + step]
        Z = np.column_stack([np.ones(ii.size), logv[ii] - fm.logv_center])
        F = np.exp(Bz @ Z.T) * ez[:, None]
        C = np.cumsum(F, axis=0).T                                # (n, M)
        j = grid_idx[ii]
        dT = (cum[j[:, None] + np.arange(2, n_rem + 2)[None, :]] - cum[j[:, None] + 1])
        pos = (np.log(np.maximum(dT, 1e-12)) - fm.z_grid[0]) / dz
        j0 = np.clip(np.floor(pos).astype(np.int64), 0, fm.z_grid.size - 2)
        fr = np.clip(pos - j0, 0.0, 1.0)
        lo = np.take_along_axis(C, j0, axis=1)
        hi = np.take_along_axis(C, j0 + 1, axis=1)
        # the bias correction is part of the forecast, so it belongs inside the
        # integral whose increments become the per-second variances
        G = (lo + fr * (hi - lo)) * np.exp(
            fm.log_bias(np.log(np.maximum(dT, 1e-12)).ravel()).reshape(dT.shape))
        out[a:a + step] = np.diff(np.concatenate(
            [np.zeros((ii.size, 1)), G], axis=1), axis=1)
    return np.maximum(out, 1e-30)


def twap_variance_batch(xi: np.ndarray, rho: np.ndarray, N: int,
                        m: int = 0) -> np.ndarray:
    """`twap_variance_from_xi`'s variance, for a whole block of origins at once."""
    w = twap_weights(N, m)
    a = w[None, :] * np.sqrt(xi)
    var = np.einsum("ij,ij->i", a, a)
    L = min(len(rho) - 1, w.size - 1)
    for l in range(1, L + 1):
        if rho[l] == 0.0:
            continue
        var += 2.0 * rho[l] * np.einsum("ij,ij->i", a[:, l:], a[:, :-l])
    return np.maximum(var, 1e-30)
