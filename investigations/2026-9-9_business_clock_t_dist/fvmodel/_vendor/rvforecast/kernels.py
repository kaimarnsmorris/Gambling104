"""Numba kernels shared by the vectorised fitting path and the streaming path.

Everything here is written so that the streaming class in `rvforecast.streaming` performs
*exactly* the same floating-point operations in the same order as the batch loop, which is
what makes the bit-identity test in tests/test_streaming.py pass.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

LN2 = 0.6931471805599453


@njit(cache=True, fastmath=False)
def register_step(v, r2, dT, valid, hl):
    """One update of the whole bank, in place. Mirrors `run_bank`'s inner body exactly."""
    K = hl.shape[0]
    for k in range(K):
        lam = np.exp(-LN2 * dT / hl[k])
        if valid:
            v[k] = lam * v[k] + (1.0 - lam) * (r2 / dT)
        else:
            v[k] = lam * v[k]
    return v


@njit(cache=True, parallel=True, fastmath=False)
def run_bank(r2, dT, valid, hl, v_init, grid_idx, out):
    """Run the register bank over the whole sample, sampling at `grid_idx`.

    r2, dT, valid : (n,) float64 / float64 / bool  -- per step
    hl            : (K,) float64 half-lives in business days
    v_init        : (K,) float64 initial variance rates
    grid_idx      : (m,) int64 ascending sample points (state AFTER step grid_idx[j])
    out           : (m, K) float64 output buffer
    Returns the final state (K,).
    """
    n = r2.shape[0]
    K = hl.shape[0]
    final = np.empty(K, dtype=np.float64)
    for k in prange(K):
        v = v_init[k]
        h = hl[k]
        j = 0
        m = grid_idx.shape[0]
        for i in range(n):
            d = dT[i]
            lam = np.exp(-LN2 * d / h)
            if valid[i]:
                v = lam * v + (1.0 - lam) * (r2[i] / d)
            else:
                v = lam * v
            while j < m and grid_idx[j] == i:
                out[j, k] = v
                j += 1
        final[k] = v
    return final


@njit(cache=True, fastmath=False)
def ewma_ratio(x, expected, lam, floor, init, cap=10.0):
    """EWMA of x/expected in clock time, clipped to [floor, cap].

    The clip matters: without it a single burst of prints can make the business clock
    run tens of times too fast for a few seconds, which both starves the fast registers
    (r^2/dT collapses) and pushes the horizon map far outside the fitted range. Clipping
    the activity factor to one order of magnitude either side keeps the clock a clock.
    """
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    acc = init
    om = 1.0 - lam
    for i in range(n):
        e = expected[i]
        rr = x[i] / e if e > 1e-12 else 1.0
        acc = lam * acc + om * rr
        if acc < floor:
            acc = floor
        elif acc > cap:
            acc = cap
        out[i] = acc
    return out


@njit(cache=True, parallel=True, fastmath=False)
def rolling_sum_forward(csum, idx, h):
    """csum[idx + h] - csum[idx] for a cumulative array, vectorised."""
    m = idx.shape[0]
    out = np.empty(m, dtype=np.float64)
    for j in prange(m):
        out[j] = csum[idx[j] + h] - csum[idx[j]]
    return out


@njit(cache=True, fastmath=False)
def aggregate_steps(r, valid, dT, step):
    """Aggregate 1s returns into `step`-second returns.

    A coarse return is valid only if every 1s return inside it is valid. Business time
    is summed. Returns (r2_c, dT_c, valid_c, end_idx) where end_idx[j] is the index of
    the last 1s bar inside coarse bar j.
    """
    n = r.shape[0]
    m = n // step
    r2c = np.empty(m, dtype=np.float64)
    dTc = np.empty(m, dtype=np.float64)
    vc = np.empty(m, dtype=np.bool_)
    ei = np.empty(m, dtype=np.int64)
    for j in range(m):
        a = j * step
        s = 0.0
        d = 0.0
        ok = True
        for i in range(a, a + step):
            s += r[i]
            d += dT[i]
            if not valid[i]:
                ok = False
        r2c[j] = s * s
        dTc[j] = d
        vc[j] = ok
        ei[j] = a + step - 1
    return r2c, dTc, vc, ei


@njit(cache=True, fastmath=False)
def run_single_full(r2, dT, valid, hl, v_init):
    """Run one register over the whole sample and keep every value (for truncation)."""
    n = r2.shape[0]
    out = np.empty(n, dtype=np.float64)
    v = v_init
    for i in range(n):
        d = dT[i]
        lam = np.exp(-LN2 * d / hl)
        if valid[i]:
            v = lam * v + (1.0 - lam) * (r2[i] / d)
        else:
            v = lam * v
        out[i] = v
    return out


@njit(cache=True, parallel=True, fastmath=False)
def run_bank_trunc(r2, dT, valid, hl, v_init, grid_idx, out, v_ref, c):
    """Register bank with the observation capped at c * v_ref (jump truncation)."""
    n = r2.shape[0]
    K = hl.shape[0]
    final = np.empty(K, dtype=np.float64)
    for k in prange(K):
        v = v_init[k]
        h = hl[k]
        j = 0
        m = grid_idx.shape[0]
        for i in range(n):
            d = dT[i]
            lam = np.exp(-LN2 * d / h)
            if valid[i]:
                x = r2[i] / d
                cap = c * v_ref[i]
                if x > cap:
                    x = cap
                v = lam * v + (1.0 - lam) * x
            else:
                v = lam * v
            while j < m and grid_idx[j] == i:
                out[j, k] = v
                j += 1
        final[k] = v
    return final


@njit(cache=True, fastmath=False)
def csum_f8(x):
    n = x.shape[0]
    out = np.empty(n + 1, dtype=np.float64)
    acc = 0.0
    out[0] = 0.0
    for i in range(n):
        acc += x[i]
        out[i + 1] = acc
    return out


@njit(cache=True, fastmath=False)
def reverse_ewma(s, lam):
    """B[t] = s[t] + lam * B[t+1]; the forward-looking exponential integral of a
    DETERMINISTIC profile. Used for the hybrid clock's horizon mapping, where the future
    shape of s is known but future activity is not."""
    n = s.shape[0]
    out = np.empty(n, dtype=np.float64)
    acc = 0.0
    for i in range(n - 1, -1, -1):
        acc = s[i] + lam * acc
        out[i] = acc
    return out


@njit(cache=True, fastmath=False)
def garch_recursion(x, w, a, b, s2_0):
    """sigma^2_i = w + a*x_{i-1}^2 + b*sigma^2_{i-1}."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    s2 = s2_0
    out[0] = s2
    for i in range(1, n):
        s2 = w + a * x[i - 1] * x[i - 1] + b * s2
        out[i] = s2
    return out


@njit(cache=True, fastmath=False)
def garch_nll(x, w, a, b, s2_0):
    n = x.shape[0]
    s2 = s2_0
    acc = 0.0
    for i in range(n):
        if s2 <= 0.0:
            return 1e10
        acc += np.log(s2) + x[i] * x[i] / s2
        s2 = w + a * x[i] * x[i] + b * s2
    return 0.5 * acc
