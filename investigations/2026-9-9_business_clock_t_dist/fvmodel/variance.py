"""Variance of a weighted sum of future one-second returns.

    Var_Y = sum_k W_k^2 xi_k + 2 sum_{k<l} W_k W_l rho(l-k) sqrt(xi_k xi_l)

with `xi` from the v2.1 forward curve and `rho` the perp return autocorrelation kernel.

Why this is not just one call to `twap_variance_from_xi`
--------------------------------------------------------
Every settlement this model prices has `W_k = 1` for all but the last `L` seconds - the
whole window before a TWAP opens loads the answer one-for-one, and a single print loads
it one-for-one throughout. Evaluating the banded quadratic form over all 14,400 seconds
of a four-hour market to discover that 14,340 of the weights are one costs about four
orders of magnitude more than it needs to.

So the sum is decomposed rather than truncated. Write `W = 1 + u`, where `u = W - 1` is
zero everywhere except the last `L` seconds:

    Var(sum W_k r_k) = Var(S_n) + 2 Cov(S_n, sum u_k r_k) + Var(sum u_k r_k)

* `Var(S_n) = IV(t, n) * [1 + 2 sum_l rho_l (1 - l/n)]` - the plain n-second return,
  closed form, exact whenever `xi` is smooth over the kernel's 120-second reach, which
  it is: it is a smooth function of business time.
* the other two terms only involve seconds within 120 of a non-zero `u`, so they need
  the forward curve over the last `L + 120` seconds and nothing else.

Nothing is approximated except the flatness of `xi` inside `kernel_inflation`, and
`check_split` scores that against the fully exact computation before anything is fitted.
"""
from __future__ import annotations

import numpy as np

try:                                                          # pragma: no cover
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:                                           # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        return (lambda f: f) if not a else a[0]

    prange = range

PAD = 120          # head seconds carried into the exact block, = the kernel's reach


# =============================================================== the exact banded form
@njit(cache=True, parallel=True, fastmath=False)
def _banded(a, rho, out):
    R, M = a.shape
    L = min(rho.shape[0] - 1, M - 1)
    for i in prange(R):
        s = 0.0
        for k in range(M):
            s += a[i, k] * a[i, k]
        for l in range(1, L + 1):
            rl = rho[l]
            if rl == 0.0:
                continue
            t = 0.0
            for k in range(M - l):
                t += a[i, k] * a[i, k + l]
            s += 2.0 * rl * t
        out[i] = s if s > 0.0 else 0.0


def banded_variance_numpy(a: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """The reference implementation. `banded_variance` must match it exactly enough."""
    a = np.atleast_2d(a)
    var = np.einsum("ij,ij->i", a, a)
    L = min(rho.size - 1, a.shape[1] - 1)
    for l in range(1, L + 1):
        if rho[l] == 0.0:
            continue
        var += 2.0 * rho[l] * np.einsum("ij,ij->i", a[:, l:], a[:, :-l])
    return np.maximum(var, 0.0)


def banded_variance(a: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Var(sum_k a_k) for rows of `a = W * sd`, with autocorrelation `rho` (rho[0]=1).

    A hundred and twenty lags over half a million quotes is thirty billion multiply-adds
    in the natural numpy form; the same loop compiled and threaded is seconds. The numpy
    version stays as `banded_variance_numpy` and the two are pinned together by a test.
    """
    a = np.atleast_2d(a)
    if not HAVE_NUMBA or a.shape[0] < 64:
        return banded_variance_numpy(a, rho)
    out = np.empty(a.shape[0], dtype=np.float64)
    _banded(np.ascontiguousarray(a, dtype=np.float64),
            np.ascontiguousarray(rho, dtype=np.float64), out)
    return out


@njit(cache=True, parallel=True, fastmath=False)
def _banded_cross(a, b, rho, out):
    R, M = a.shape
    L = min(rho.shape[0] - 1, M - 1)
    for i in prange(R):
        s = 0.0
        for k in range(M):
            s += a[i, k] * b[i, k]
        for l in range(1, L + 1):
            rl = rho[l]
            if rl == 0.0:
                continue
            t = 0.0
            for k in range(M - l):
                t += a[i, k] * b[i, k + l] + a[i, k + l] * b[i, k]
            s += rl * t
        out[i] = s


def banded_cross_numpy(a: np.ndarray, b: np.ndarray, rho: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    out = np.einsum("ij,ij->i", a, b)
    L = min(rho.size - 1, a.shape[1] - 1)
    for l in range(1, L + 1):
        if rho[l] == 0.0:
            continue
        out = out + rho[l] * (np.einsum("ij,ij->i", a[:, l:], b[:, :-l])
                              + np.einsum("ij,ij->i", a[:, :-l], b[:, l:]))
    return out


def banded_cross(a: np.ndarray, b: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """`sum_{j,k} rho(|j-k|) a_k b_j` for rows of two aligned vectors."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    if not HAVE_NUMBA or a.shape[0] < 64:
        return banded_cross_numpy(a, b, rho)
    out = np.empty(a.shape[0], dtype=np.float64)
    _banded_cross(np.ascontiguousarray(a, dtype=np.float64),
                  np.ascontiguousarray(b, dtype=np.float64),
                  np.ascontiguousarray(rho, dtype=np.float64), out)
    return out


def kernel_inflation(rho: np.ndarray, H) -> np.ndarray:
    """`Var(sum of H consecutive returns) / (H * xi)` under a flat forward curve."""
    H = np.atleast_1d(np.maximum(np.asarray(H, dtype=np.float64), 1.0))
    L = rho.size - 1
    l = np.arange(1, L + 1, dtype=np.float64)
    out = np.ones(H.shape)
    for i, li in enumerate(l):
        out = out + 2.0 * rho[i + 1] * np.maximum(1.0 - li / H, 0.0)
    return np.maximum(out, 1e-6)


# ================================================================== the split form
def settlement_variance(iv_head: np.ndarray, xi_block: np.ndarray, W: np.ndarray,
                        rho: np.ndarray, n: int, pad: int = PAD) -> np.ndarray:
    """Var of `sum_{k=1..n} W_k r_k` where `W_k = 1` for every k before the block.

    `xi_block` is the forward curve over the **last** `m` seconds of the horizon, `W`
    the weights over the whole horizon (length n), and `iv_head` the integrated variance
    over the first `n - m` seconds. `m` must reach at least `pad` seconds further back
    than the first second where `W` departs from one, which `block_length` guarantees.
    """
    xi_block = np.atleast_2d(xi_block)
    rows, m = xi_block.shape
    sd = np.sqrt(np.maximum(xi_block, 0.0))
    Wb = np.asarray(W[-m:] if m > 0 else W, dtype=np.float64)
    iv_n = np.asarray(iv_head, dtype=np.float64).reshape(-1) + xi_block.sum(axis=1)
    if m >= n:
        return banded_variance(sd * Wb[None, :], rho)
    du = sd * (Wb - 1.0)[None, :]
    return np.maximum(iv_n * kernel_inflation(rho, n)
                      + 2.0 * banded_cross(sd, du, rho)
                      + banded_variance(du, rho), 0.0)


def block_length(kind: str, n: int, L: int = 60, pad: int = PAD) -> int:
    """How many trailing seconds of the forward curve the exact block needs.

    `W` departs from one only inside the settlement window, so the block has to cover
    that window plus the kernel's reach on either side of its opening edge.
    """
    if kind == "perp_single":
        return min(n, 1)                     # W is one throughout; only IV is needed
    return int(min(n, L + pad))


# ===================================================================== the harness
def check_split(xi_full: np.ndarray, W: np.ndarray, rho: np.ndarray,
                block: int) -> dict:
    """Exact banded variance against the split form, on the same rows."""
    xi_full = np.atleast_2d(xi_full)
    n = xi_full.shape[1]
    exact = banded_variance(np.sqrt(np.maximum(xi_full, 0.0)) * W[None, :], rho)
    iv_head = xi_full[:, :n - block].sum(axis=1)
    appr = settlement_variance(iv_head, xi_full[:, n - block:], W, rho, n)
    rel = (appr - exact) / np.maximum(exact, 1e-300)
    return {"n": int(n), "block": int(block), "max_abs_rel": float(np.max(np.abs(rel))),
            "mean_rel": float(np.mean(rel)),
            "p99_abs_rel": float(np.quantile(np.abs(rel), 0.99))}


# ================================================================ QLIKE / MZ scoring
QLIKE_FLOOR = 1.2704034809047095    # E[chi2_1] - E[ln chi2_1] - 1, one squared draw


def qlike(y2: np.ndarray, v: np.ndarray) -> float:
    r = np.maximum(y2, 1e-30) / np.maximum(v, 1e-30)
    return float(np.mean(r - np.log(r) - 1.0))


def mz(y2: np.ndarray, v: np.ndarray) -> tuple:
    A = np.column_stack([np.ones(v.size), v])
    b = np.linalg.lstsq(A, y2, rcond=None)[0]
    yh = A @ b
    r2 = 1.0 - np.var(y2 - yh) / max(np.var(y2), 1e-300)
    return float(b[1]), float(b[0]), float(r2)
