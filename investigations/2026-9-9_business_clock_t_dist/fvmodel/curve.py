"""Forward variance over just the seconds the settlement actually weights.

`rvforecast` gives the whole forward curve `xi(t, 1..n)`. For a four-hour market that is
14,400 numbers of which 14,340 carry weight one, and `fv.variance` only needs the last
few hundred plus one integrated number for the rest. These helpers evaluate the same
integral on an arbitrary set of future seconds, so nothing longer than the block is ever
materialised.

The identity that has to hold - and that `check_tail` asserts - is that the increments
of the **bias-corrected** integral are what get summed, exactly as
`ForwardModel.forward_curve_business` does. Differencing the raw integral instead drops
the factor that turns `exp(E[log RV])` into `E[RV]` (NOTES D34), which is a factor of
3.5 at ten seconds.

One wrinkle that has to be reproduced rather than tidied away. The bias correction is a
spline in log business time and it falls steeply at the very short end, steeply enough
that the *corrected* integral is occasionally non-monotone over the first second or two
of a quiet market. `ForwardModel.forward_curve_business` clips those increments at zero,
so the sum of the curve is very slightly above the integral it came from - about 0.1% at
a five-minute horizon on the worst origins. The head integral here is clipped the same
way, over the first `CLIP_HEAD` seconds where it can happen at all, so that the split
form and the shipped curve agree to floating point rather than to a tenth of a per cent.
"""
from __future__ import annotations

import numpy as np

SEC_PER_DAY = 86400.0
CLIP_HEAD = 24     # seconds over which the corrected integral can go backwards


# ============================================================ business time, one row
def dT_at(model, state, t_now: int, u: np.ndarray) -> np.ndarray:
    """Business time from `t_now` to `t_now + u`, for an array of offsets u >= 1.

    Deterministic seasonal profile for the future plus the activity factor known at the
    origin decayed back to one - the same no-lookahead construction the shipped horizon
    map uses.
    """
    u = np.asarray(u, dtype=np.int64)
    h = int(u.max())
    t_now = int(t_now)
    model._cache(t_now, t_now + h + 1)
    i = t_now - model._c0
    seg = model._s[i + 1:i + h + 1] / SEC_PER_DAY
    cum = np.cumsum(seg)
    if model.gamma == 0.0 or state.act == 1.0:
        return cum[u - 1]
    lam = float(np.exp(-1.0 / model.blend_tau_s))
    w = lam ** np.arange(1, h + 1, dtype=np.float64)
    kick = state.act ** model.gamma - 1.0
    return cum[u - 1] + kick * np.cumsum(seg * w)[u - 1]


def _iv_from_cum(fm, dT, logv: np.ndarray) -> np.ndarray:
    """The bias-corrected integrated variance at business ages `dT`, one register row."""
    cum = fm.cumulative(np.asarray(logv, dtype=np.float64))
    z = np.log(np.maximum(np.asarray(dT, dtype=np.float64), 1e-12))
    # `log_bias` fits a spline via scipy's BSpline.design_matrix, which insists on a
    # 1-D evaluation array; the batch path (unconditional_xi under shrink_w) hands
    # this a 2-D block of (market, second), so it is flattened for the spline and
    # reshaped back rather than restricting every caller to a single row.
    bias = fm.log_bias(z.ravel()).reshape(z.shape) if z.ndim > 1 else fm.log_bias(z)
    return np.interp(z, fm.z_grid, cum) * np.exp(bias)


def slow_index(hl_business, target_s: float = 3600.0) -> int:
    """Index of the register whose business half-life is nearest `target_s` seconds.

    The cap below limits the forward curve at short business ages to a multiple of the
    curve the *slower* registers alone would produce. This picks the register that
    defines "slower": by default the one nearest an hour.
    """
    hl = np.asarray(hl_business, dtype=np.float64) * 86400.0
    return int(np.argmin(np.abs(np.log(np.maximum(hl, 1e-9))
                                - np.log(max(target_s, 1e-9)))))


def _slow_logv(logv: np.ndarray, cap_i: int) -> np.ndarray:
    """`logv` with every register faster than `cap_i` clipped to the one at `cap_i`.

    Only *downward* excursions of the fast registers are left alone: the cap exists to
    stop a burst of second-scale volatility propagating into the forecast at full
    strength, not to stop the fast bank reporting calm.
    """
    out = np.array(logv, dtype=np.float64, copy=True)
    if out.ndim == 1:
        out[:cap_i] = np.minimum(out[:cap_i], out[cap_i])
    else:
        out[:, :cap_i] = np.minimum(out[:, :cap_i], out[:, cap_i][:, None])
    return out


def forward_block(model, state, t_now: int, n: int, m: int,
                  cap_c: float = 0.0, cap_i: int = None):
    """`(iv_head, xi_block)` for a horizon of `n` seconds with an exact block of `m`.

    `iv_head` is the integrated variance over the first `n - m` seconds and `xi_block`
    the per-second forward variances of the last `m`. Together they are exactly the
    quantities `fv.variance.settlement_variance` consumes.

    `cap_c > 0` limits each per-second forward variance to `cap_c` times the variance
    the registers at and below `cap_i` alone would give. Off by default; see report
    section 11.2 for what it is for and how `cap_c` is chosen.
    """
    n, m = int(n), int(min(m, n))
    H = max(n - m, 0)
    logv = np.log(np.maximum(state.v, 1e-300))
    cap = cap_c > 0.0 and cap_i is not None
    if H == 0:
        dT = dT_at(model, state, t_now, np.arange(1, n + 1, dtype=np.int64))
        g = _iv_from_cum(model.forward, dT, logv)
        inc = np.maximum(np.diff(np.concatenate([[0.0], g])), 0.0)
        if cap:
            gs = _iv_from_cum(model.forward, dT, _slow_logv(logv, cap_i))
            inc = np.minimum(inc, cap_c * np.maximum(
                np.diff(np.concatenate([[0.0], gs])), 0.0))
        return 0.0, inc
    k0 = min(CLIP_HEAD, H)
    u = np.concatenate([np.arange(1, k0 + 1), np.arange(H, n + 1)]).astype(np.int64)
    dT = dT_at(model, state, t_now, u)
    g = _iv_from_cum(model.forward, dT, logv)
    hd = np.maximum(np.diff(np.concatenate([[0.0], g[:k0]])), 0.0)
    jump = max(float(g[k0]) - float(g[k0 - 1]), 0.0) if H > k0 else 0.0
    blk = np.maximum(np.diff(g[k0:]), 0.0)
    if cap:
        gs = _iv_from_cum(model.forward, dT, _slow_logv(logv, cap_i))
        hd = np.minimum(hd, cap_c * np.maximum(
            np.diff(np.concatenate([[0.0], gs[:k0]])), 0.0))
        if H > k0:
            jump = min(jump, cap_c * max(float(gs[k0]) - float(gs[k0 - 1]), 0.0))
        blk = np.minimum(blk, cap_c * np.maximum(np.diff(gs[k0:]), 0.0))
    return float(hd.sum()) + jump, blk


# =============================================================== business time, batch
class BatchClock:
    """Vectorised `dT_at` / `forward_block` over many origins of one `BatchV2`."""

    def __init__(self, bv):
        self.bv = bv
        self.fm = bv.forward
        self.lam = bv.lam_b
        self.gamma = bv.gamma

    def dT_at(self, i: np.ndarray, u: np.ndarray) -> np.ndarray:
        """(len(i), len(u)) business time from each origin over the given offsets."""
        bv = self.bv
        i = np.asarray(i, dtype=np.int64)[:, None]
        u = np.asarray(u, dtype=np.int64)[None, :]
        base = bv.cum[i + u + 1] - bv.cum[i + 1]
        if self.gamma == 0.0:
            return base
        kick = (bv.act[i[:, 0]] ** self.gamma - 1.0)[:, None]
        lam = self.lam
        tail = lam * (bv.B[i + 1] - (lam ** u.astype(np.float64)) * bv.B[i + u + 1])
        return base + kick * tail

    def forward_block(self, logv: np.ndarray, i: np.ndarray, n: int, m: int,
                      chunk: int = 4000, cap_c: float = 0.0, cap_i: int = None):
        """`(iv_head (r,), xi_block (r, m))` for many origins at one horizon.

        `cap_c > 0` applies the same short-business-time cap as the streaming path.
        """
        from rvforecast.regression import spline_basis

        fm = self.fm
        n, m = int(n), int(min(m, n))
        H = max(n - m, 0)
        k0 = min(CLIP_HEAD, H)
        if H == 0:
            u = np.arange(1, n + 1, dtype=np.int64)
        else:
            u = np.concatenate([np.arange(1, k0 + 1),
                                np.arange(H, n + 1)]).astype(np.int64)
        head = H > 0
        r = np.asarray(i).size
        Th = np.ascontiguousarray(spline_basis(fm.z_grid, fm.knots) @ fm.coef)
        ez = np.exp(fm.z_grid) * fm.w_grid
        z0, dz = fm.z_grid[0], fm.z_grid[1] - fm.z_grid[0]
        M = fm.z_grid.size
        iv_head = np.zeros(r)
        xi = np.empty((r, u.size - k0 - 1 if head else u.size))
        ii = np.asarray(i, dtype=np.int64)
        cap = cap_c > 0.0 and cap_i is not None
        for a in range(0, r, chunk):
            sl = slice(a, min(a + chunk, r))
            dT = self.dT_at(ii[sl], u)
            zz = np.log(np.maximum(dT, 1e-12))
            pos = (zz - z0) / dz
            j0 = np.clip(np.floor(pos).astype(np.int64), 0, M - 2)
            fr = np.clip(pos - j0, 0.0, 1.0)
            bias = np.exp(fm.log_bias(zz.ravel()).reshape(zz.shape))

            def _curve(lv):
                Z = np.column_stack([np.ones(lv.shape[0]), lv - fm.logv_center])
                F = np.exp(Th @ Z.T) * ez[:, None]
                C = np.cumsum(F, axis=0).T                  # (c, M)
                lo = np.take_along_axis(C, j0, axis=1)
                hi = np.take_along_axis(C, j0 + 1, axis=1)
                return (lo + fr * (hi - lo)) * bias

            def _split(g):
                if head:
                    first = np.maximum(np.diff(np.concatenate(
                        [np.zeros((g.shape[0], 1)), g[:, :k0]], axis=1), axis=1), 0.0)
                    jump = (np.maximum(g[:, k0] - g[:, k0 - 1], 0.0) if H > k0
                            else np.zeros(g.shape[0]))
                    return first, jump, np.maximum(np.diff(g[:, k0:], axis=1), 0.0)
                return (None, None, np.maximum(np.diff(
                    np.concatenate([np.zeros((g.shape[0], 1)), g], axis=1), axis=1), 0.0))

            first, jump, blk = _split(_curve(logv[sl]))
            if cap:
                fs, js, bs = _split(_curve(_slow_logv(logv[sl], cap_i)))
                blk = np.minimum(blk, cap_c * bs)
                if head:
                    first = np.minimum(first, cap_c * fs)
                    jump = np.minimum(jump, cap_c * js)
            if head:
                iv_head[sl] = first.sum(axis=1) + jump
            xi[sl] = blk
        return iv_head, xi


def unconditional_xi(model, dT: np.ndarray, dT_prev=None) -> np.ndarray:
    """The forward curve the unconditional register bank would give, at ages `dT`.

    This is the shrink target (spec ruling R6): the seasonal level with no volatility
    news in it at all, which is what `registers.initial_value` encodes. `dT` is in
    business days, matching `dT_at`.

    ALIGNMENT. The returned curve is per-second *increments* over exactly the seconds
    `dT` names, so `forward_block`'s block `xi[k]` and `xi_bar[k]` are the same second
    and the shrink in `xi_adjust` mixes like with like. The block starts at offset
    `H + 1 = n - m + 1`, not at the quote origin, so the first increment is
    `g(dT[0]) - g(dT_prev)` where `dT_prev` is the business age of second `H` - the
    second immediately BEFORE the block. `dT_prev=None` means the block really does
    start at the origin (`H == 0`), and only then is the first increment the whole
    integral to `dT[0]`. Differencing against a prepended zero when `H > 0` puts the
    entire head integral into `xi_bar[0]`, which at n=300 is ~87x its neighbour.

    The increments are clipped at zero for the same reason `forward_block` clips its
    own: the bias-corrected integral is occasionally non-monotone at the very short end.
    """
    v0 = np.log(np.maximum(np.asarray(model.params["registers"]["initial_value"],
                                      dtype=np.float64), 1e-300))
    dT = np.asarray(dT, dtype=np.float64)
    g = np.atleast_2d(_iv_from_cum(model.forward, dT, v0))
    if dT_prev is None:
        g0 = np.zeros((g.shape[0], 1))
    else:
        # one extra evaluation of the SAME integral, at the second before the block;
        # reshaped to a column so the 1-D and the (market, second) batch case share
        # one code path
        prev = np.reshape(np.asarray(dT_prev, dtype=np.float64), (-1, 1))
        g0 = np.atleast_2d(_iv_from_cum(model.forward, prev, v0))
    inc = np.maximum(np.diff(np.concatenate([g0, g], axis=1), axis=1), 0.0)
    return inc if dT.ndim > 1 else inc[0]


# ======================================================================== the harness
def check_tail(model, state, t_now: int, n: int, m: int, tol: float = 1e-7) -> float:
    """`forward_block` must agree with the shipped full curve on the same seconds."""
    ref = model.forward_curve(state, t_now, n)
    iv_head, blk = forward_block(model, state, t_now, n, m)
    den = np.maximum(np.abs(ref[-blk.size:]), 1e-300)
    gap = float(np.max(np.abs(blk - ref[-blk.size:]) / den))
    if n > m:
        gh = abs(iv_head - ref[:n - m].sum()) / max(ref[:n - m].sum(), 1e-300)
        gap = max(gap, float(gh))
    if gap > tol:
        raise AssertionError("forward_block disagrees with Model.forward_curve by "
                             "%.3g at n=%d, m=%d" % (gap, n, m))
    return gap
