"""The Chainlink print model.

    log C_i  =  EWMA_tau( log X )( t_i - delta )  +  b_t  +  eps_i

three pieces, fitted and tracked separately because they live on three timescales:

* **X**, the blended exchange input, and the filter `(tau, delta)` that Chainlink
  applies to it. Timescale: seconds. Fitted once.
* **b_t**, the slow level offset between Chainlink's basket and our X - funding, the
  USDT premium, whatever else separates a perpetual from an index. Timescale: hours.
  Tracked causally at runtime; the preliminary note removed it with a centred +/-15
  minute mean, which a live quote cannot do.
* **eps_i**, the fast residual. Timescale: about a minute of memory. Modelled, not
  removed: at ten seconds from expiry it is a material share of the settlement
  variance, and its conditional mean given the last observed residual is a free
  correction to the settlement's expected level.

On keying the filter to stamps rather than arrivals
---------------------------------------------------
The EWMA is advanced **stamp second to stamp second**, over every intervening second of
X, and each print is matched to the filter value at `stamp - delta`. It is not advanced
on arrival times and it does not jump-decay across gaps.

Both shortcuts break in the same place and in the same direction. Receive latency is not
independent of the market: it lengthens exactly when the feed is busy, which is when the
price is moving. Keying on arrival therefore reads X too late precisely during a
one-directional move, and the filter inherits a bias that has the same sign as the move.
Jump-decaying over a gap - decaying `lam^dt` and mixing in only the newest X - throws
away the path in between, which on a one-way move is the entire signal, and leaves the
filter systematically behind. `fit_filter` measures both alternatives so the size of the
effect is on the record rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import BP

EPS_MODES = ("full", "unconditional", "none")

try:                                                          # pragma: no cover
    from numba import njit
except ImportError:                                           # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not a else a[0]


# ============================================================== the filter, batch side
@njit(cache=True, fastmath=False)
def _ewma_kernel(x, a, out):
    acc = x[0]
    for i in range(x.size):
        acc = a * acc + (1.0 - a) * x[i]
        out[i] = acc


def ewma(x: np.ndarray, tau: float) -> np.ndarray:
    """Clock-time EWMA with time constant `tau` seconds on a complete 1s grid."""
    out = np.empty_like(x)
    _ewma_kernel(x, float(np.exp(-1.0 / tau)), out)
    return out


def shift_frac(a: np.ndarray, d: float) -> np.ndarray:
    """`b[t] = a[t - d]` for a real lag d >= 0, linear between the two bracketing grid
    points. The first ceil(d) entries are meaningless and are returned as NaN."""
    k = int(np.floor(d))
    f = d - k
    out = np.full_like(a, np.nan)
    if f == 0.0:
        if k == 0:
            return a.copy()
        out[k:] = a[:-k]
    else:
        lo = np.empty_like(a)
        lo[:] = np.nan
        hi = np.empty_like(a)
        hi[:] = np.nan
        if k == 0:
            hi[:] = a
        else:
            hi[k:] = a[:-k]
        lo[k + 1:] = a[:-(k + 1)]
        out = (1.0 - f) * hi + f * lo
    return out


def blend_logx(logp: np.ndarray, logs: np.ndarray, w_spot: float) -> np.ndarray:
    """log X = (1 - w) log(perp) + w log(spot), in log space, NaN-propagating."""
    if w_spot == 0.0:
        return logp
    if w_spot == 1.0:
        return logs
    return (1.0 - w_spot) * logp + w_spot * logs


def rolling_mean(x: np.ndarray, half: int) -> np.ndarray:
    n = x.size
    c = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)


# =========================================================================== the fit
def filter_prediction(logx: np.ndarray, tau: float, delta: float,
                      p_late: float = 0.0) -> np.ndarray:
    """The model's prediction of `log C` at every grid second.

    `p_late` is the probability that a *reported* stamp is one second later than the
    print's true payload second. On the 82.9% of the history whose stamp was recovered
    from arrival time rather than carried by the capture, the recovery is exact ~93% of
    the time and the errors are almost all +1 s. Ignoring that biases the fit: a
    seven-per-cent admixture of prints displaced by a whole second is a large
    perturbation next to a 0.5 bp residual, and least squares pays for it by lengthening
    tau and delta. Folding the known error rate into the predictor instead lets the fit
    recover the *underlying* filter from estimated-stamp data.
    """
    F = ewma(logx, tau)
    pred = shift_frac(F, delta)
    if p_late > 0.0:
        pred = (1.0 - p_late) * pred + p_late * shift_frac(F, delta + 1.0)
    return pred


def _gather_shift(a: np.ndarray, idx: np.ndarray, d: float) -> np.ndarray:
    """`a[idx - d]` for a real lag, without materialising a shifted full-length array."""
    k = int(np.floor(d))
    f = d - k
    hi = a[idx - k]
    return hi if f == 0.0 else (1.0 - f) * hi + f * a[idx - k - 1]


def fit_filter(logp: np.ndarray, logs: np.ndarray, dC: np.ndarray,
               idx: np.ndarray, w_grid, tau_grid, delta_grid, p_late_grid=(0.0,),
               detrend_half: int = 900, verbose=None) -> dict:
    """Grid search for `(w_spot, tau, delta, p_late)` on the mapping residual.

    Both sides are detrended by a centred rolling mean so the slow basis `b_t` cannot
    contaminate the identification of a filter whose time constant is a second or two;
    `dC` arrives already detrended and `idx` is the (already subsampled) set of seconds
    to score on. The causal tracker for `b_t` is fitted afterwards, given this answer.
    """
    y = dC[idx]
    best, rows = None, []
    for w in w_grid:
        lx = blend_logx(logp, logs, float(w))
        ok = np.isfinite(lx)
        lxf = _ffill_inplace(np.where(ok, lx, 0.0), ok)
        for tau in tau_grid:
            F = ewma(lxf, float(tau))
            dF = F - rolling_mean(F, detrend_half)
            for d in delta_grid:
                base = _gather_shift(dF, idx, float(d))
                late = None
                for pl in p_late_grid:
                    if pl == 0.0:
                        pred = base
                    else:
                        if late is None:
                            late = _gather_shift(dF, idx, float(d) + 1.0)
                        pred = (1.0 - pl) * base + pl * late
                    r = float(np.sqrt(np.nanmean((y - pred) ** 2)) * BP)
                    rows.append((float(w), float(tau), float(d), float(pl), r))
                    if best is None or r < best[-1]:
                        best = (float(w), float(tau), float(d), float(pl), r)
        if verbose:
            verbose("    w=%.2f  best so far %.4f bp at (w=%.2f, tau=%.3f, delta=%.2f,"
                    " p_late=%.2f)" % (w, best[-1], best[0], best[1], best[2], best[3]))
    return {"w_spot": best[0], "tau_s": best[1], "delta_s": best[2],
            "p_stamp_late": best[3], "rmse_bp": best[4], "surface": rows}


def fit_filter_2stage(logp, logs, dC, usable, sub_coarse=61, sub_fine=11,
                      p_late_grid=(0.0,), verbose=None) -> dict:
    """Coarse sweep over the whole box, then a fine sweep around the winner.

    A single fine grid over (w, tau, delta, p_late) is ~60k evaluations and does not
    finish; a coarse pass on a heavier subsample followed by a local refinement lands on
    the same optimum for a hundredth of the cost, and the coarse surface is what the
    report plots anyway.
    """
    all_idx = np.flatnonzero(usable)
    idx_c = np.ascontiguousarray(all_idx[::sub_coarse])
    c = fit_filter(logp, logs, dC, idx_c,
                   np.round(np.concatenate([[0.0, 0.2, 0.35, 0.5],
                                            np.arange(0.55, 0.91, 0.05), [1.0]]), 3),
                   np.round(np.geomspace(0.4, 6.0, 13), 4),
                   np.round(np.arange(0.0, 1.61, 0.1), 3),
                   p_late_grid, verbose=verbose)
    if verbose:
        verbose("  coarse: w=%.2f tau=%.3f delta=%.2f p_late=%.2f rmse=%.4f"
                % (c["w_spot"], c["tau_s"], c["delta_s"], c["p_stamp_late"],
                   c["rmse_bp"]))
    idx_f = np.ascontiguousarray(all_idx[::sub_fine])
    wg = np.round(np.clip(c["w_spot"] + np.arange(-0.06, 0.061, 0.02), 0.0, 1.0), 3)
    tg = np.round(c["tau_s"] * np.geomspace(0.72, 1.4, 11), 4)
    dg = np.round(np.clip(c["delta_s"] + np.arange(-0.10, 0.101, 0.02), 0.0, 4.0), 3)
    pg = (p_late_grid if len(p_late_grid) == 1
          else np.round(np.clip(c["p_stamp_late"] + np.arange(-0.03, 0.031, 0.01),
                                0.0, 0.4), 3))
    f = fit_filter(logp, logs, dC, idx_f, np.unique(wg), np.unique(tg), np.unique(dg),
                   tuple(np.unique(pg)), verbose=None)
    f["coarse"] = {k: v for k, v in c.items() if k != "surface"}
    f["surface_coarse"] = c["surface"]
    f["n_scored"] = int(idx_f.size)
    return f


def _ffill_inplace(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    idx = np.where(ok, np.arange(x.size), 0)
    np.maximum.accumulate(idx, out=idx)
    return x[idx]


# ------------------------------------------------ the shortcut this design avoids
@njit(cache=True, fastmath=False)
def _jump_ewma(x, dt, lam, out):
    acc = x[0]
    out[0] = acc
    for i in range(1, x.size):
        a = lam ** dt[i]
        acc = a * acc + (1.0 - a) * x[i]
        out[i] = acc


def jump_decay_filter(u: np.ndarray, dt: np.ndarray, tau: float) -> np.ndarray:
    """The EWMA a print-driven implementation would produce.

    Advanced only when a print arrives: decay `lam^dt` and mix in the newest input.
    That throws away the path between prints, and on a one-directional move the path is
    the entire signal - the filter ends up systematically behind the market by more than
    its own time constant. The production filter instead steps second by second over
    every intervening second of X, keyed on payload stamps. `10_print_model.py`
    measures the difference rather than assuming it.
    """
    out = np.empty_like(u)
    _jump_ewma(np.ascontiguousarray(u), np.ascontiguousarray(dt.astype(np.float64)),
               float(np.exp(-1.0 / tau)), out)
    return out


# ==================================================================== the slow basis
def basis_series(d: np.ndarray, have: np.ndarray, hl_s: float,
                 lag_s: int = 2) -> np.ndarray:
    """Causal EWM of the per-print level offset `d_i = log C_i - F(t_i - delta)`.

    `lag_s` is the receive lag in whole seconds: the value returned for second `t` uses
    only prints stamped at or before `t - lag_s`, which is all a live quote can have
    seen. `have` marks the seconds carrying a usable print; on a second with no print
    the tracker decays nothing - it simply holds, because no news is not evidence about
    the level.
    """
    lam = float(np.exp(-np.log(2.0) / max(hl_s, 1e-9)))
    out = np.empty(d.size, dtype=np.float64)
    _basis_kernel(np.ascontiguousarray(d), np.ascontiguousarray(have), lam, out)
    if lag_s > 0:
        sh = np.full_like(out, np.nan)
        sh[lag_s:] = out[:-lag_s]
        return sh
    return out


@njit(cache=True, fastmath=False)
def _basis_kernel(d, have, lam, out):
    acc = np.nan
    n = d.size
    for i in range(n):
        if have[i]:
            if acc != acc:                       # first observation seeds the tracker
                acc = d[i]
            else:
                acc = lam * acc + (1.0 - lam) * d[i]
        out[i] = acc


# ================================================================== the fast residual
@dataclass
class EpsModel:
    """Variance and autocorrelation of the fast Chainlink residual."""
    sigma_bp: float                       # unconditional sd, basis points
    kappa: float                          # sd scales with local vol to this power
    v_ref: float                          # the local-vol level sigma_bp is quoted at
    rho: np.ndarray                       # rho[l], l = 0..max_lag, rho[0] = 1
    reg_index: int = 2                    # which register supplies the local vol
    n: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def sigma(self) -> float:
        return self.sigma_bp / BP

    def sigma_at(self, v_local) -> np.ndarray:
        """sd of eps given the local 1-second volatility `v_local` (same units as
        `v_ref`)."""
        r = np.maximum(np.asarray(v_local, dtype=np.float64), 1e-12) / self.v_ref
        return self.sigma * r ** self.kappa

    def rho_at(self, lag):
        """rho at a (possibly fractional, possibly array) lag; zero past the fitted
        range, where the measured value is already inside its own standard error."""
        lag = np.abs(np.asarray(lag, dtype=np.float64))
        return np.interp(lag, np.arange(self.rho.size, dtype=np.float64), self.rho,
                         left=1.0, right=0.0)

    def to_dict(self) -> dict:
        return {"sigma_bp": self.sigma_bp, "kappa": self.kappa, "v_ref": self.v_ref,
                "reg_index": self.reg_index, "n": self.n, "rho": self.rho.tolist(),
                "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "EpsModel":
        return cls(float(d["sigma_bp"]), float(d["kappa"]), float(d["v_ref"]),
                   np.asarray(d["rho"], dtype=np.float64), int(d.get("reg_index", 2)),
                   int(d.get("n", 0)), d.get("meta", {}))


def eps_acf(eps: np.ndarray, sel: np.ndarray, max_lag: int = 300) -> np.ndarray:
    """rho(l) of the residual, l = 0..max_lag, on a masked 1-second grid.

    Products are formed only where both seconds carry a residual; the normalisation
    corrects for how many pairs each lag actually had, so a lag that loses more pairs to
    blanks is not silently shrunk toward zero.
    """
    e0 = np.where(sel, np.nan_to_num(eps), 0.0)
    w = sel.astype(np.float64)
    den = float(e0 @ e0)
    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for l in range(1, max_lag + 1):
        pairs = float(w[l:] @ w[:-l])
        out[l] = (float(e0[l:] @ e0[:-l]) / den * (w.sum() / pairs)) if pairs > 0 else 0.0
    return out


def fit_eps_model(eps: np.ndarray, sel: np.ndarray, v_local: np.ndarray,
                  reg_index: int = 2, max_lag: int = 300) -> EpsModel:
    """sigma_eps, the vol exponent kappa and the autocorrelation, in one pass."""
    m = sel & np.isfinite(eps) & np.isfinite(v_local) & (v_local > 0)
    e = eps[m]
    v = v_local[m]
    v_ref = float(np.median(v))
    lv = np.log(v / v_ref)
    le = np.log(np.maximum(np.abs(e), 1e-14))
    A = np.column_stack([np.ones(lv.size), lv])
    kappa = float(np.linalg.lstsq(A, le, rcond=None)[0][1])
    # sigma is set so that the *standardised* residual has unit variance at v_ref,
    # rather than by the raw sd, so that the scaling law and the level agree
    z = e / (v / v_ref) ** kappa
    sigma_bp = float(np.sqrt(np.mean(z ** 2)) * BP)
    rho = eps_acf(eps, m, max_lag)
    return EpsModel(sigma_bp, kappa, v_ref, rho, reg_index, int(m.sum()),
                    meta={"sd_raw_bp": float(np.std(e) * BP),
                          "kurtosis": float(((e - e.mean()) ** 4).mean() / e.var() ** 2),
                          "skew": float(((e - e.mean()) ** 3).mean() / e.std() ** 3)})


# ------------------------------------------------- the conditional settlement residual
def eps_conditional(model: EpsModel, ages: np.ndarray, weights: np.ndarray,
                    sigma: float, mode: str = "full") -> tuple:
    """Mean coefficient on `eps_last` and variance of the weighted residual average.

    `ages[i]` is the number of seconds from the last observed residual to component i's
    payload second; `weights` sum to one over the components whose residual is not yet
    known. Returns `(c, var)` with

        E[eps_bar | eps_t] = c * eps_t
        Var(eps_bar | eps_t) = var        (already scaled by sigma^2)

    `mode` selects the ablations: "full" is the model, "unconditional" keeps the
    correlation but forgets `eps_t`, and "none" removes the term.
    """
    if mode not in EPS_MODES:
        raise ValueError("eps mode must be one of %s, got %r" % (EPS_MODES, mode))
    if mode == "none" or weights.size == 0:
        return 0.0, 0.0
    a = np.asarray(ages, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    s2 = float(sigma) ** 2
    R = model.rho_at(np.abs(a[:, None] - a[None, :]))
    if mode == "unconditional":
        return 0.0, s2 * float(w @ R @ w)
    r = model.rho_at(a)
    c = float(w @ r)
    var = s2 * float(w @ R @ w - c * c)
    return c, max(var, 0.0)


# ============================================================ the streaming filter
@dataclass
class FilterParams:
    w_spot: float = 0.0
    tau_s: float = 1.6
    delta_s: float = 0.2
    basis_hl_s: float = 900.0
    # A second, faster tracker carried alongside the pricing one. It is never read by
    # the pricer: 15 s is the constrained optimum of the extended sweep (NOTES E19) and
    # this exists so the shadow period can score the two on lead-k print prediction
    # without a model change. See report section 11.3.
    basis_hl_alt_s: float = 15.0
    lag_s: int = 2
    p_stamp_late: float = 0.0

    @property
    def lam(self) -> float:
        return float(np.exp(-1.0 / self.tau_s)) if self.tau_s > 0 else 0.0

    def to_dict(self) -> dict:
        return {"w_spot": self.w_spot, "tau_s": self.tau_s, "delta_s": self.delta_s,
                "basis_hl_s": self.basis_hl_s, "basis_hl_alt_s": self.basis_hl_alt_s,
                "lag_s": self.lag_s, "p_stamp_late": self.p_stamp_late,
                "lambda_per_s": self.lam}

    @classmethod
    def from_dict(cls, d: dict) -> "FilterParams":
        return cls(w_spot=float(d.get("w_spot", 0.0)),
                   tau_s=float(d.get("tau_s", 1.6)),
                   delta_s=float(d.get("delta_s", 0.2)),
                   basis_hl_s=float(d.get("basis_hl_s", 900.0)),
                   basis_hl_alt_s=float(d.get("basis_hl_alt_s", 15.0)),
                   lag_s=int(d.get("lag_s", 2)),
                   p_stamp_late=float(d.get("p_stamp_late", 0.0)))


HIST = 16          # seconds of filter history kept, enough for any receive lag


class PrintFilter:
    """The runtime half of the print model: advance on X, observe on C.

    Everything it holds is a float or a short ring of floats, so the whole thing
    serialises into the same flat state blob as the register bank.
    """

    def __init__(self, p: FilterParams):
        self.p = p
        self.ts = None
        self.xf = np.nan                 # EWMA of log X at instant `ts`
        self.x = np.nan                  # log X at instant `ts`
        self.hist = np.full(HIST, np.nan)   # xf at ts, ts-1, ...
        self.xhist = np.full(HIST, np.nan)
        self.b = np.nan                  # slow basis, the one the pricer uses
        self.b_alt = np.nan              # faster tracker, carried for scoring only
        self.eps_last = 0.0
        self.eps_last_alt = 0.0
        self.eps_last_ts = None
        self.last_print_ts = None
        self.last_print_logc = np.nan

    # ---- state -----------------------------------------------------------
    def to_flat(self) -> dict:
        d = {"cl_ts": float(self.ts if self.ts is not None else -1),
             "cl_xf": float(self.xf), "cl_x": float(self.x), "cl_b": float(self.b),
             "cl_b_alt": float(self.b_alt),
             "cl_eps_last": float(self.eps_last),
             "cl_eps_last_alt": float(self.eps_last_alt),
             "cl_eps_last_ts": float(self.eps_last_ts
                                     if self.eps_last_ts is not None else -1),
             "cl_last_print_ts": float(self.last_print_ts
                                       if self.last_print_ts is not None else -1),
             "cl_last_print_logc": float(self.last_print_logc)}
        for i in range(HIST):
            d["cl_h%d" % i] = float(self.hist[i])
            d["cl_x%d" % i] = float(self.xhist[i])
        return d

    @classmethod
    def from_flat(cls, d: dict, p: FilterParams) -> "PrintFilter":
        f = cls(p)
        f.ts = None if d["cl_ts"] < 0 else int(d["cl_ts"])
        f.xf, f.x, f.b = d["cl_xf"], d["cl_x"], d["cl_b"]
        f.b_alt = d.get("cl_b_alt", np.nan)
        f.eps_last = d["cl_eps_last"]
        f.eps_last_alt = d.get("cl_eps_last_alt", 0.0)
        f.eps_last_ts = None if d["cl_eps_last_ts"] < 0 else int(d["cl_eps_last_ts"])
        f.last_print_ts = (None if d["cl_last_print_ts"] < 0
                           else int(d["cl_last_print_ts"]))
        f.last_print_logc = d["cl_last_print_logc"]
        f.hist = np.array([d["cl_h%d" % i] for i in range(HIST)])
        f.xhist = np.array([d["cl_x%d" % i] for i in range(HIST)])
        return f

    # ---- advance ---------------------------------------------------------
    def _push(self, xf: float, x: float) -> None:
        self.hist[1:] = self.hist[:-1]
        self.hist[0] = xf
        self.xhist[1:] = self.xhist[:-1]
        self.xhist[0] = x

    def advance(self, ts: int, log_x: float) -> None:
        """Fold one second of the blended input. Call once per second, in order.

        A multi-second jump is stepped through second by second holding X at its newest
        value. That is the least-bad thing to do with no data in between; the honest
        form is `advance_path`, which takes the actual path, and the batch fit uses it.
        """
        ts = int(ts)
        if self.ts is not None and ts <= self.ts:
            raise ValueError("PrintFilter.advance needs strictly increasing seconds")
        lam = self.p.lam
        if self.ts is None or not np.isfinite(self.xf):
            self.xf = log_x                              # cold start seeds the filter
            self._push(self.xf, log_x)
        else:
            for _ in range(min(ts - self.ts, HIST)):
                self.xf = lam * self.xf + (1.0 - lam) * log_x
                self._push(self.xf, log_x)
        self.ts = ts
        self.x = log_x

    def advance_path(self, ts: int, log_x_path) -> None:
        """Fold a whole second-by-second path ending at `ts` (the honest form of a
        multi-second advance, and what the batch path does)."""
        path = np.asarray(log_x_path, dtype=np.float64)
        t0 = ts - path.size + 1
        for i, v in enumerate(path):
            self.advance(t0 + i, float(v))

    # ---- read ------------------------------------------------------------
    def filtered_at(self, ts: float) -> float:
        """The filter value at a real-valued past instant, from the history ring."""
        if self.ts is None:
            return np.nan
        age = self.ts - float(ts)
        if age < 0.0 or age > HIST - 2:
            return np.nan
        k = int(np.floor(age))
        f = age - k
        return (1.0 - f) * self.hist[k] + f * self.hist[k + 1] if f > 0 else self.hist[k]

    def predict_print(self, stamp_ts: float) -> float:
        """The model's expected `log C` for a print with this payload second."""
        return self.filtered_at(float(stamp_ts) - self.p.delta_s) + (
            self.b if np.isfinite(self.b) else 0.0)

    def predict_print_alt(self, stamp_ts: float) -> float:
        """The same prediction from the faster parallel tracker.

        Nothing in the pricer calls this. It exists so a shadow run can score the two
        half-lives against each other on lead-k prediction of the actual print, which
        is the comparison that should decide whether to switch.
        """
        return self.filtered_at(float(stamp_ts) - self.p.delta_s) + (
            self.b_alt if np.isfinite(self.b_alt) else 0.0)

    # ---- observe ---------------------------------------------------------
    def observe(self, stamp_ts: int, log_c: float) -> float:
        """Fold a received print. Returns its residual eps."""
        f = self.filtered_at(float(stamp_ts) - self.p.delta_s)
        if not np.isfinite(f):
            return np.nan
        d = log_c - f
        lam_b = float(np.exp(-np.log(2.0) / max(self.p.basis_hl_s, 1e-9)))
        self.b = d if not np.isfinite(self.b) else lam_b * self.b + (1.0 - lam_b) * d
        eps = d - self.b
        # the parallel tracker: same update, faster half-life, never used for pricing
        lam_a = float(np.exp(-np.log(2.0) / max(self.p.basis_hl_alt_s, 1e-9)))
        self.b_alt = (d if not np.isfinite(self.b_alt)
                      else lam_a * self.b_alt + (1.0 - lam_a) * d)
        self.eps_last_alt = float(d - self.b_alt)
        self.eps_last = float(eps)
        self.eps_last_ts = int(stamp_ts)
        self.last_print_ts = int(stamp_ts)
        self.last_print_logc = float(log_c)
        return float(eps)

    # ---- reconstruction (section 4) --------------------------------------
    def reconstruct(self, stamp_ts: int, eps_model: "EpsModel" = None,
                    sigma: float = None) -> float:
        """`log C` for a print already stamped but not yet received.

        The X path up to `stamp - delta` is in hand, so the only unknown is the
        residual, and the residual has about a minute of memory: carry the last
        observed one forward by rho(age). With no eps model this degrades to the
        filtered level plus the basis, which is the "no reconstruction of the residual"
        ablation.
        """
        f = self.filtered_at(float(stamp_ts) - self.p.delta_s)
        if not np.isfinite(f):
            return np.nan
        out = f + (self.b if np.isfinite(self.b) else 0.0)
        if eps_model is not None and self.eps_last_ts is not None:
            age = float(stamp_ts) - self.eps_last_ts
            scale = 1.0 if sigma is None else sigma / eps_model.sigma
            out += float(eps_model.rho_at(age)) * self.eps_last * scale
        return out
