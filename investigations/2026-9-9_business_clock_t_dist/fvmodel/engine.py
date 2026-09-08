"""Batch evaluation: the same model as `fair_value`, run over half a million quotes.

`fv.fairvalue.fair_value` prices one market from one state and is what the autotrader
calls. Scoring it needs the identical arithmetic at every second of a synthetic market
grid, which a per-quote Python loop cannot deliver. This module is the vectorised twin:
same weights, same variance split, same residual algebra, same tail.

`tests/test_engine_matches_fairvalue.py` prices a sample of the same quotes both ways and
requires them to agree to floating point. That test is the only thing standing between
"the evaluation scores the model" and "the evaluation scores a second implementation of
the model", so it is not optional.

Structure
---------
`Window` holds one contiguous span of 1-second data with the v2.1 batch state and the
print-model series already built. `evaluate` takes a window, a settlement kind, a market
length and a remaining time, and returns one row per synthetic market.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import CL_T0, PERP_T0, load_grid, load_perp
from .chainlink import blend_logx, ewma, shift_frac, basis_series, eps_conditional
from .curve import BatchClock
from .variance import PAD, block_length, settlement_variance
from .weights import settlement_weights

MAX_STEP_AGE = 10          # seconds a Chainlink mark may be carried forward
LAG_S = 2                  # whole-second receive lag used by the batch path


# ==================================================================== the data window
class Window:
    """One contiguous span of seconds with everything the estimator reads off it."""

    def __init__(self, params: dict, fp, t0: int, t1: int, warm: int = 10 * 86400,
                 chainlink: bool = True, log=print):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from clkit.state import BatchV2

        self.params = params
        self.fp = fp
        # the warm-up cannot reach before the data starts; asking for it just fills
        # the prefix with missing bars and drags the register bank through them
        self.base = max(int(t0) - int(warm), PERP_T0 if not chainlink else CL_T0)
        self.t0, self.t1 = int(t0), int(t1)
        self.chainlink = chainlink

        if chainlink:
            df = load_grid(self.base, t1)
            self.ts = df["ts"].to_numpy()
            perp = df["perp"].to_numpy()
            spot = df["spot"].to_numpy()
            cl = df["cl"].to_numpy()
            ntr = df["n_trades"].to_numpy().astype(np.float64)
            gap = df["gap_flag"].to_numpy()
            self.excluded = df["excluded"].to_numpy()
            del df
        else:
            df = load_perp(self.base, t1)
            self.ts = df["ts"].to_numpy()
            perp = df["perp"].to_numpy()
            ntr = df["n_trades"].to_numpy().astype(np.float64)
            gap = df["gap_flag"].to_numpy()
            spot = None
            cl = None
            self.excluded = np.zeros(self.ts.size, dtype=bool)
            del df

        n = self.ts.size
        self.n = n
        self.perp_ok = np.isfinite(perp) & ((gap & 6) == 0)
        self.perp = perp
        self.logp = np.where(self.perp_ok, np.log(np.where(self.perp_ok, perp, 1.0)),
                             np.nan)

        # --- the v2.1 batch state (registers, clock, forward model)
        self.bv = BatchV2(params, int(self.ts[0]), n,
                          np.where(self.perp_ok, perp, np.nan),
                          self.perp_ok & np.roll(self.perp_ok, 1), ntr)
        self.clock = BatchClock(self.bv)
        log("  window %d s, perp ok %.4f" % (n, self.perp_ok.mean()))

        # --- the print model
        if chainlink:
            self.spot_ok = np.isfinite(spot)
            logs = np.where(self.spot_ok, np.log(np.where(self.spot_ok, spot, 1.0)),
                            np.nan)
            lx = blend_logx(self.logp, logs, fp.w_spot)
            ok = np.isfinite(lx)
            idx = np.where(ok, np.arange(n), 0)
            np.maximum.accumulate(idx, out=idx)
            self.lxf = np.where(ok, lx, 0.0)[idx]
            self.F = ewma(self.lxf, fp.tau_s)
            self.Fd = shift_frac(self.F, fp.delta_s)
            self.have_cl = np.isfinite(cl)
            self.cl_step, self.step_age = _step_fill(cl, MAX_STEP_AGE)
            logc = np.where(self.have_cl, np.log(np.where(self.have_cl, cl, 1.0)),
                            np.nan)
            usable = self.have_cl & self.perp_ok & self.spot_ok & ~self.excluded
            d_lvl = logc - self.Fd
            self.b = basis_series(d_lvl, usable, fp.basis_hl_s, lag_s=LAG_S)
            self.eps = np.where(usable & np.isfinite(self.b), d_lvl - self.b, np.nan)
            # the step-filled log series is what a settlement actually averages
            self.logc_step = np.log(np.maximum(self.cl_step, 1e-12))
            self.eps_step = _ffill_arr(self.eps)
            log("  chainlink: %.4f of seconds carry a mark, %.4f step-filled within %d s"
                % (self.have_cl.mean(), np.isfinite(self.cl_step).mean(), MAX_STEP_AGE))

    # ---- index helpers ---------------------------------------------------
    def i(self, ts):
        return np.asarray(ts, dtype=np.int64) - int(self.ts[0])

    def prepare(self, idx: np.ndarray) -> None:
        """Run the register bank once for every quote second the evaluation will use.

        `BatchV2.registers` sweeps the whole window, so calling it per cell per ablation
        would run it several hundred times over twelve million seconds. The quote
        indices are the same across ablations, so they are gathered once.
        """
        self._cache_idx = np.unique(np.asarray(idx, dtype=np.int64))
        regs = self.bv.registers(self._cache_idx)
        self._cache_logv = np.log(np.maximum(regs, 1e-300))

    def logv_at(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        cache = getattr(self, "_cache_idx", None)
        if cache is not None:
            pos = np.searchsorted(cache, idx)
            if pos.size and pos.max() < cache.size and np.all(cache[pos] == idx):
                return self._cache_logv[pos]
        regs = self.bv.registers(idx)
        return self.bv.logv_at(regs, idx)

    def trim(self) -> None:
        """Free what nothing downstream reads, so the whole overlap fits in memory."""
        for a in ("s", "rate"):
            if hasattr(self.bv, a):
                setattr(self.bv, a, None)
        for a in ("logp", "logc_step", "spot_ok"):
            if hasattr(self, a):
                setattr(self, a, None)
        for a in ("b", "eps", "eps_step"):
            if getattr(self, a, None) is not None:
                setattr(self, a, np.asarray(getattr(self, a), dtype=np.float32))


def _ffill_arr(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    idx = np.where(ok, np.arange(x.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[: int(np.argmax(ok)) if ok.any() else x.size] = np.nan
    return out


def _step_fill(cl: np.ndarray, max_age: int):
    """Carry each Chainlink mark forward up to `max_age` seconds.

    A null second in the reconciled file means no capture saw a *new* mark, not that the
    feed went silent - on the recorder's own grid Chainlink republishes an identical
    price in 1.5% of adjacent seconds. A settlement averages the feed's value at each
    second, which is the step function, so that is what gets built here; beyond
    `max_age` the carry is refused and the second is left missing.
    """
    ok = np.isfinite(cl)
    idx = np.where(ok, np.arange(cl.size), -1)
    np.maximum.accumulate(idx, out=idx)
    age = np.arange(cl.size) - idx
    out = np.where((idx >= 0) & (age <= max_age), cl[np.maximum(idx, 0)], np.nan)
    return out, np.where(idx >= 0, age, np.iinfo(np.int32).max)


# ======================================================================= the scoring
@dataclass
class Cell:
    kind: str
    L: int
    n: int
    rows: dict = field(default_factory=dict)

    def __len__(self):
        return len(next(iter(self.rows.values()))) if self.rows else 0


def market_grid(win: Window, L: int, step: int = 300,
                burn_days: int = 14) -> np.ndarray:
    """Expiries on a `step`-second grid, far enough inside the window to be scorable.

    `burn_days` protects the register bank: it starts from the unconditional variance
    and the slowest register has a fourteen-business-day half-life, so quotes are only
    taken once the bank has had that long to forget its seed.
    """
    lo = max(win.t0, int(win.ts[0]) + burn_days * 86400 + L)
    hi = int(win.ts[-1]) - 1
    T = np.arange((lo // step + 1) * step, hi, step, dtype=np.int64)
    return T[(T - L >= win.ts[0]) & (T <= win.ts[-1])]


def evaluate(win: Window, model, kind: str, L: int, n: int, sw,
             expiries: np.ndarray = None, twap_len: int = 60,
             book: dict = None) -> Cell:
    """One row per synthetic market: the model's quote and what actually settled.

    `book`, when given, holds `imbalance` and `px_age_s` as full-window arrays indexed
    by window second, which is what an event-driven feed would fill in directly.
    """
    from .fairvalue import ALPHA_MAX

    T = market_grid(win, L) if expiries is None else np.asarray(expiries)
    iT = win.i(T)
    it = iT - n                                             # the quote second
    iO = iT - L                                             # the market's open
    keep = (it > 0) & (iT < win.n) & (iO >= 0)
    T, iT, it, iO = T[keep], iT[keep], it[keep], iO[keep]
    Lw = int(twap_len)

    # The market-observable baseline is the same model on a counterparty's information
    # set: the received prints and nothing else. Its most recent price is the print it
    # received LAG_S seconds ago, so it quotes from that second - everything since,
    # including the moves the perp feed already shows, is unknown to it. That is the
    # whole of the difference, and it is what makes the gap "the lag edge" rather than
    # an arbitrary handicap.
    mo = bool(sw.market_observable) and kind == "chainlink_twap60"
    it_q = it - LAG_S if mo else it
    n_q = n + LAG_S if mo else n

    # ---- settlement weights, identical for every market in the cell -----------
    if kind == "chainlink_twap60" and not sw.composed_weights:
        s = settlement_weights("perp_twap", n_q, L=Lw)
    else:
        s = settlement_weights(kind, n_q, model.fp.to_dict(), L=Lw)

    # ---- what is realised, and what is known at the quote ---------------------
    if kind == "chainlink_twap60":
        lvl = win.cl_step
        n_recv = int(np.clip(Lw - n - LAG_S, 0, Lw))
        if mo:
            # the counterparty prices off the last print it holds; with no exchange
            # feed the martingale forecast of every later print is that print
            x_t = np.log(np.maximum(lvl[it - LAG_S], 1e-12))
            b_t = np.zeros(it.size)
        else:
            x_t = win.lxf[it]
            b_t = win.b[it] if sw.use_basis else np.zeros(it.size)
            b_t = np.where(np.isfinite(b_t), b_t, 0.0)
        p_ref = np.exp(x_t + b_t)
        strike = lvl[iO]
        settle = _window_mean(lvl, iT, Lw)
    else:
        lvl = win.perp
        n_recv = int(np.clip(Lw - n, 0, Lw)) if kind == "perp_twap" else 0
        p_ref = win.perp[it]
        x_t = win.logp[it]
        b_t = np.zeros(it.size)
        strike = lvl[iO]
        settle = (_window_mean(lvl, iT, Lw) if kind == "perp_twap" else lvl[iT])

    n_comp = Lw if kind != "perp_single" else 1
    n_recv = n_recv if kind != "perp_single" else 0
    omega = (n_comp - n_recv) / n_comp
    # the received prints are those stamped at or before T - n - LAG_S, which is the
    # same window whether or not the baseline is quoting from that second
    A_R = (_window_mean(lvl, iT - n - LAG_S if kind == "chainlink_twap60" else iT - n,
                        n_recv) * n_recv / n_comp / p_ref) if n_recv else np.zeros(it.size)
    W = s.W_raw / omega

    # ---- the deterministic carry ---------------------------------------------
    d_bar = np.zeros(it.size)
    n_transit = 0
    if kind == "chainlink_twap60" and not mo:
        lam = model.fp.lam
        j = np.arange(n_comp - n_recv, dtype=np.float64)     # unknown components
        K = n - j - model.fp.delta_s
        fut = K > 0
        n_transit = int((~fut).sum())
        d_bar = (lam ** K[fut]).sum() * (win.F[it] - x_t)
        if sw.use_reconstruction and n_transit:
            for jj in j[~fut]:
                d_bar = d_bar + (win.Fd[iT - int(jj)] - x_t)
        d_bar = d_bar / n_comp / omega
        if not sw.composed_weights:
            d_bar = np.zeros(it.size)

    # ---- variance -------------------------------------------------------------
    logv = win.logv_at(it_q)
    if model.kappa_vol:
        logv = logv + 2.0 * model.kappa_vol
    m_blk = block_length(kind if sw.composed_weights else "perp_twap", n_q, Lw, PAD)
    iv_head, xi = win.clock.forward_block(logv, it_q, n_q, m_blk,
                                          cap_c=model.xi_cap_c,
                                          cap_i=model.xi_cap_i)
    ratio = model.input_var_ratio if (kind == "chainlink_twap60" and sw.use_blend) else 1.0
    act = win.bv.act[it_q]
    var_ret = np.empty(it.size)
    if sw.rho_mode == "conditional" and "act0" in model.rho:
        grp = np.digitize(act, model.act_cuts)
        for g in range(3):
            m = grp == g
            if m.any():
                var_ret[m] = settlement_variance(iv_head[m], xi[m] * ratio, W,
                                                 model.rho["act%d" % g], n_q)
    else:
        var_ret[:] = settlement_variance(iv_head, xi * ratio, W,
                                         model.rho_for(1.0, sw.rho_mode), n_q)

    # ---- the residual ---------------------------------------------------------
    eps_bar = np.zeros(it.size)
    var_eps = np.zeros(it.size)
    var_basis = 0.0
    if kind == "chainlink_twap60" and sw.eps_mode != "none":
        stamps = T[:, None] - np.arange(n_comp - n_recv)[None, :]
        ages = (stamps[0] - (T[0] - n - LAG_S)).astype(np.float64)
        w = np.full(ages.size, 1.0 / max(ages.size, 1))
        mode = sw.eps_mode if not sw.market_observable else "unconditional"
        c, q_unit = eps_conditional(model.eps, ages, w, 1.0, mode)
        # the same local-volatility definition `fair_value` uses: the business time of
        # the next second, including the activity kick decayed forward
        v_loc = np.sqrt(np.maximum(np.exp(logv[:, model.eps.reg_index]), 0.0)
                        * np.maximum(win.clock.dT_at(it, np.array([1]))[:, 0], 1e-30))
        sig = model.eps.sigma_at(v_loc)
        var_eps = q_unit * sig ** 2
        eps_last = win.eps_step[it - LAG_S]
        eps_bar = c * np.where(np.isfinite(eps_last), eps_last, 0.0)
        if sw.use_basis:
            from .build import basis_drift_window_var
            var_basis = basis_drift_window_var(model, ages, w)

    var_y = np.maximum(var_ret + var_eps + var_basis, 1e-300)

    # ---- the order-book term ---------------------------------------------------
    m_Y = np.zeros(it.size)
    if sw.use_alpha and book is not None:
        # beta saturates, so only the seconds just after the quote origin carry a
        # non-zero increment; `book` holds full-window arrays indexed by window second
        na = min(n_q, ALPHA_MAX)
        dTc = win.clock.dT_at(it_q, np.arange(1, na + 1))
        I = np.nan_to_num(book["imbalance"][it])
        age = book["px_age_s"][it]
        m_Y = model.alpha.m_Y(I, W[:na], dTc, np.where(np.isfinite(age), age, np.nan))

    # ---- the strike and the realised residual ---------------------------------
    if kind == "perp_single":
        y_raw = np.log(strike / p_ref)
        y_act = np.log(settle / p_ref)
    else:
        # the settlement is an arithmetic mean of prices, so the linearised y is
        # g_bar + half the mean square of the component moves; the event
        # {settle > strike} is exact either way, but the *mean* of y is not g_bar's
        y_raw = (strike / p_ref - A_R) / omega - 1.0
        y_act = (settle / p_ref - A_R) / omega - 1.0
        if sw.jensen:
            y_raw = y_raw - 0.5 * var_y
            y_act = y_act - 0.5 * var_y
    y_star = y_raw - d_bar - m_Y - eps_bar
    resid = y_act - d_bar - m_Y - eps_bar

    D = win.clock.dT_at(it_q, np.array([n_q]))[:, 0]
    z = np.log(np.maximum(D, 1e-12))
    tail = model.tail_for(kind, sw.tail_mode)
    p_up = tail.prob_up(y_star, var_y, z)
    up = settle > strike

    ok = (np.isfinite(y_star) & np.isfinite(var_y) & np.isfinite(settle)
          & np.isfinite(strike) & np.isfinite(p_ref) & (var_y > 0))
    if kind == "chainlink_twap60":
        ok &= _window_all_finite(win.cl_step, iT, Lw) & np.isfinite(lvl[iO])
        ok &= ~_window_any(win.excluded, iT, Lw) & ~win.excluded[it]
    return Cell(kind, L, n, {
        "T": T[ok], "t": T[ok] - n, "p_up": p_up[ok], "var_y": var_y[ok],
        "y_star": y_star[ok], "resid": resid[ok], "z": z[ok], "up": up[ok],
        "omega": np.full(int(ok.sum()), omega), "n_known": np.full(int(ok.sum()), n_recv),
        "n_transit": np.full(int(ok.sum()), n_transit),
        "m_Y": m_Y[ok] if np.ndim(m_Y) else np.zeros(int(ok.sum())),
        "eps_bar": eps_bar[ok],
        "var_eps": (var_eps[ok] if np.ndim(var_eps) else
                    np.full(int(ok.sum()), var_eps)),
        "var_basis": np.full(int(ok.sum()), float(var_basis)),
        "carry": d_bar[ok] if
        np.ndim(d_bar) else np.zeros(int(ok.sum())),
        "act": act[ok], "settle": settle[ok], "strike": strike[ok],
        "it": it[ok], "it_q": it_q[ok], "p_ref": p_ref[ok],
    })


# --------------------------------------------------------------------- small helpers
def _window_mean(x: np.ndarray, iend: np.ndarray, L: int) -> np.ndarray:
    """Mean of `x` over the L entries ending at `iend` inclusive."""
    if L <= 0:
        return np.zeros(np.size(iend))
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    return (c[iend + 1] - c[iend + 1 - L]) / L


def _window_all_finite(x: np.ndarray, iend: np.ndarray, L: int) -> np.ndarray:
    c = np.concatenate([[0], np.cumsum(np.isfinite(x).astype(np.int64))])
    return (c[iend + 1] - c[iend + 1 - L]) == L


def _window_any(x: np.ndarray, iend: np.ndarray, L: int) -> np.ndarray:
    c = np.concatenate([[0], np.cumsum(x.astype(np.int64))])
    return (c[iend + 1] - c[iend + 1 - L]) > 0


# ============================================ bridging back to the single-quote path
def state_at(win: "Window", it: int, model) -> "object":
    """The `FVState` a live system would hold at window index `it`.

    This is what makes the batch path checkable: build the state the streaming code
    would have, hand it to `fair_value`, and require the same answer.
    """
    from rvforecast.streaming import State

    from .chainlink import HIST, PrintFilter
    from .fairvalue import FVState

    logv = win.logv_at(np.array([it]))[0]
    st = State(int(win.ts[it]), float(win.perp[it]), np.exp(logv), float(win.bv.act[it]),
               int(it))
    f = PrintFilter(model.fp)
    if win.chainlink:
        f.ts = int(win.ts[it])
        f.xf = float(win.F[it])
        f.x = float(win.lxf[it])
        f.hist = win.F[it - HIST + 1:it + 1][::-1].copy()
        f.xhist = win.lxf[it - HIST + 1:it + 1][::-1].copy()
        f.b = float(win.b[it]) if np.isfinite(win.b[it]) else np.nan
        e = win.eps_step[it - LAG_S]
        f.eps_last = float(e) if np.isfinite(e) else 0.0
        f.eps_last_ts = int(win.ts[it]) - LAG_S
        f.last_print_ts = int(win.ts[it]) - LAG_S
    return FVState(st, f, win_book(win, it))


def win_book(win: "Window", it: int):
    return None


def market_at(win: "Window", kind: str, T: int, n: int, market_len: int,
              twap_len: int = 60):
    """The `Market` a live system would be looking at, at `T - n`.

    `market_len` is the length of the up/down window - it fixes the strike, which is the
    settlement reference price when the market opened. `twap_len` is the settlement
    average's own length, which is a different thing entirely and is what decides how
    many prints have already been received.
    """
    from .fairvalue import Market

    iT = win.i(T)
    prints = ()
    if kind == "chainlink_twap60":
        first_known = n + LAG_S                     # component j >= this is received
        prints = tuple((int(T - j), float(win.cl_step[iT - j]))
                       for j in range(first_known, twap_len)
                       if np.isfinite(win.cl_step[iT - j]))
        strike = float(win.cl_step[win.i(T - market_len)])
    elif kind == "perp_twap":
        prints = tuple((int(T - j), float(win.perp[iT - j]))
                       for j in range(n, twap_len) if np.isfinite(win.perp[iT - j]))
        strike = float(win.perp[win.i(T - market_len)])
    else:
        strike = float(win.perp[win.i(T - market_len)])
    return Market(kind, int(T), strike, prints, None, twap_len)
