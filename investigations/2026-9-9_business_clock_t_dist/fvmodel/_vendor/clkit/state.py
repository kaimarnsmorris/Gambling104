"""Batch reconstruction of the shipped v2 state (registers, forward curve, IV).

`rvforecast` ships a streaming `Model` that folds one bar at a time. Part B needs the
same quantities at half a million forecast origins, so this module reproduces the *batch*
path the coefficients were fitted with - business clock, hybrid activity factor,
`registers.step_seconds`-second aggregation, `run_bank` - and then evaluates the shipped
ForwardModel vectorised.

`check_against_streaming()` asserts the two agree; it is run by scripts/03_settlement.py
before anything is fitted, because a silent mismatch here would poison every number in
part B.
"""
from __future__ import annotations

import numpy as np

from .common import PARAMS_JSON, load_params  # noqa: F401

from rvforecast.forward import ForwardModel                     # noqa: E402
from rvforecast.kernels import (aggregate_steps, csum_f8, ewma_ratio,  # noqa: E402
                                reverse_ewma, run_bank)
from rvforecast.regression import spline_basis                  # noqa: E402
from rvforecast.seasonality import Seasonality                  # noqa: E402
from rvforecast.streaming import Model                          # noqa: E402

SEC_PER_DAY = 86400.0


class BatchV2:
    """The v2 state over one contiguous 1-second window."""

    def __init__(self, params: dict, ts0: int, n: int, close: np.ndarray,
                 valid: np.ndarray, n_trades: np.ndarray):
        self.p = params
        self.model = Model(params)
        self.ts0 = int(ts0)
        self.n = int(n)
        ts = np.arange(ts0, ts0 + n, dtype=np.int64)

        self.r = np.zeros(n)
        self.r[1:] = np.diff(np.log(close))
        self.r[~valid] = 0.0
        self.valid = valid.copy()

        self.s = self.model.seas.s(ts)                           # per-second variance rate
        c = params["clock"]
        self.gamma = float(c.get("gamma", 0.0))
        self.act_hl_s = float(c.get("act_hl_s", 600.0))
        self.blend_tau_s = float(c.get("blend_tau_s", 0.0)) or self.act_hl_s
        self.act = np.ones(n)
        if self.gamma > 0.0 and params.get("activity_profile"):
            prof = Seasonality.from_dict(params["activity_profile"])
            exp_sec = prof.s(ts) / 60.0
            lam = float(np.exp(-np.log(2.0) / max(self.act_hl_s, 1e-9)))
            self.act = ewma_ratio(np.ascontiguousarray(n_trades.astype(np.float64)),
                                  np.ascontiguousarray(exp_sec.astype(np.float64)),
                                  lam, float(c.get("act_floor", 0.05)), 1.0,
                                  float(c.get("act_cap", 10.0)))
        self.rate = self.s * self.act ** self.gamma
        self.dT = self.rate / SEC_PER_DAY
        self.cum = csum_f8(self.s / SEC_PER_DAY)                 # deterministic profile
        lam_b = float(np.exp(-1.0 / self.blend_tau_s))
        self.B = reverse_ewma(np.ascontiguousarray(self.s / SEC_PER_DAY), lam_b)
        self.lam_b = lam_b

        r_ = params["registers"]
        self.hl = np.asarray(r_["half_life_business"], dtype=np.float64)
        self.v_init = np.asarray(r_["initial_value"], dtype=np.float64)
        self.step = int(r_.get("step_seconds", 1))
        self.forward = ForwardModel.from_dict(params["forward"])
        self.rho = np.asarray(params["acf_kernel"]["last"], dtype=np.float64)

    # ------------------------------------------------------------------ registers
    def registers(self, idx: np.ndarray) -> np.ndarray:
        """Register bank sampled at 1s indices `idx` (state after folding second idx)."""
        idx = np.ascontiguousarray(np.sort(np.unique(idx)).astype(np.int64))
        if self.step == 1:
            r2u, dTu, vu, grid = self.r ** 2, self.dT, self.valid, idx
        else:
            r2u, dTu, vu, end_idx = aggregate_steps(
                np.ascontiguousarray(self.r), np.ascontiguousarray(self.valid),
                np.ascontiguousarray(self.dT), self.step)
            grid = np.clip(np.searchsorted(end_idx, idx, side="right") - 1,
                           0, r2u.size - 1)
        out = np.empty((grid.size, self.hl.size), dtype=np.float64)
        run_bank(np.ascontiguousarray(r2u), np.ascontiguousarray(dTu),
                 np.ascontiguousarray(vu), self.hl, self.v_init,
                 np.ascontiguousarray(grid, dtype=np.int64), out)
        self._reg_idx = idx
        return out

    def logv_at(self, regs: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Rows of `regs` matching `idx` (which need not be sorted or unique)."""
        pos = np.searchsorted(self._reg_idx, idx)
        return np.log(np.maximum(regs[pos], 1e-300))

    # -------------------------------------------------------------- forward curve
    def dT_cum(self, i: np.ndarray, h: int) -> np.ndarray:
        """(m, h) cumulative business time from each origin over the next h seconds.

        Deterministic profile for the future plus the activity factor known at the origin
        decayed back to 1 - the same no-lookahead construction `horizon_dT` uses.
        """
        i = np.asarray(i, dtype=np.int64)
        u = np.arange(1, h + 1, dtype=np.int64)
        base = self.cum[i[:, None] + u[None, :] + 1] - self.cum[i[:, None] + 1]
        if self.gamma == 0.0:
            return base
        kick = (self.act[i] ** self.gamma - 1.0)[:, None]
        lam = self.lam_b
        tail = lam * (self.B[i[:, None] + 1]
                      - (lam ** u[None, :].astype(np.float64))
                      * self.B[i[:, None] + u[None, :] + 1])
        return base + kick * tail

    def forward_curve(self, logv: np.ndarray, dTc: np.ndarray,
                      chunk: int = 4000) -> np.ndarray:
        """xi(t, u), u = 1..h, for many origins at once. Mirrors
        ForwardModel.forward_curve_business row by row."""
        fm = self.forward
        m, h = dTc.shape
        Th = np.ascontiguousarray(spline_basis(fm.z_grid, fm.knots) @ fm.coef)
        ez = np.exp(fm.z_grid)
        z0, dz = fm.z_grid[0], fm.z_grid[1] - fm.z_grid[0]
        M = fm.z_grid.size
        out = np.empty((m, h))
        for a in range(0, m, chunk):
            sl = slice(a, min(a + chunk, m))
            Z = np.column_stack([np.ones(logv[sl].shape[0]),
                                 logv[sl] - fm.logv_center])
            F = np.exp(Th @ Z.T) * ez[:, None]
            C = np.cumsum(F * fm.w_grid[:, None], axis=0)          # (M, mc)
            z = np.log(np.maximum(dTc[sl], 1e-12))
            pos = (z - z0) / dz
            j0 = np.clip(np.floor(pos).astype(np.int64), 0, M - 2)
            fr = np.clip(pos - j0, 0.0, 1.0)
            rows = np.arange(z.shape[0])[:, None]
            lo = C[j0, rows]
            hi = C[j0 + 1, rows]
            g = (lo + fr * (hi - lo)) * np.exp(fm.log_bias(z.ravel()).reshape(z.shape))
            xi = np.diff(np.concatenate([np.zeros((z.shape[0], 1)), g], axis=1), axis=1)
            out[sl] = np.maximum(xi, 0.0)
        return out

    # ------------------------------------------------------------------- checks
    def check_forward(self, i: int, logv_row: np.ndarray, h: int = 300) -> float:
        """Max relative gap between the batch forward curve at origin `i` and the
        shipped streaming `Model.forward_curve` given the same register state."""
        st = self.model.new_state(int(self.ts0 + i), 1.0,
                                  v=np.exp(logv_row))
        st.act = float(self.act[i])
        ref = self.model.forward_curve(st, int(self.ts0 + i), h)
        mine = self.forward_curve(logv_row[None, :], self.dT_cum(np.array([i]), h))[0]
        den = np.maximum(np.abs(ref), 1e-300)
        return float(np.max(np.abs(mine - ref) / den))


def weighted_variance(xi: np.ndarray, rho: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Var(sum_k W_k r_k) for many rows: sum W^2 xi + 2 sum_{l>=1} rho_l W_k W_{k+l} sd sd."""
    sd = np.sqrt(np.maximum(xi, 0.0))
    a = sd * W[None, :]
    var = np.einsum("ij,ij->i", a, a)
    L = min(rho.size - 1, W.size - 1)
    for l in range(1, L + 1):
        if rho[l] == 0.0:
            continue
        var += 2.0 * rho[l] * np.einsum("ij,ij->i", a[:, l:], a[:, :-l])
    return np.maximum(var, 0.0)


def ewma_weights(K: int, lam: float) -> np.ndarray:
    """W_k = 1 - lam^(K-k+1), k = 1..K: an EWMA-settled market's loading on future returns."""
    k = np.arange(1, K + 1, dtype=np.float64)
    return 1.0 - lam ** (K - k + 1.0)


def twap_weights(K: int) -> np.ndarray:
    k = np.arange(1, K + 1, dtype=np.float64)
    return (K - k + 1.0) / K


def single_weights(K: int) -> np.ndarray:
    return np.ones(K)


def qlike(y: np.ndarray, v: np.ndarray) -> float:
    r = np.maximum(y, 1e-30) / np.maximum(v, 1e-30)
    return float(np.mean(r - np.log(r) - 1.0))


def mz(y: np.ndarray, v: np.ndarray) -> tuple:
    """Mincer-Zarnowitz of realised squared residual on predicted variance."""
    A = np.column_stack([np.ones(v.size), v])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    yh = A @ b
    r2 = 1.0 - np.var(y - yh) / np.var(y)
    return float(b[1]), float(b[0]), float(r2)
