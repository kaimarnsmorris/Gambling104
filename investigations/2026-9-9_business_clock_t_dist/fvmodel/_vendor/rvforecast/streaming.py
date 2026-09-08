"""The streaming API the autotrader uses.

    from rvforecast import load_params, State, forecast, warm_start

    model = load_params("output/params.json")       # one file, versioned
    st    = model.new_state(ts0, close0)
    st    = model.update(st, ts, close, n_trades, quote_volume, gap=False)
    f     = forecast(st, ts, 900)                   # 15 minutes ahead
    f.sigma_ann, f.rv, f.prob_above(log(K / S_t))

`State` is a flat dict of floats plus the clock timestamp, so it can be persisted to a
database and resumed without replaying history. Every update is O(K).

The register recursion is literally the same numba kernel the batch fitting code runs, so
a resumed state is bit-identical to one produced by the vectorised path
(tests/test_streaming.py). The horizon integral dT_h is the one place where the two paths
differ numerically: the batch path differences a whole-sample cumulative sum while the
streaming path sums locally. The two agree to ~1e-12 relative, which the test asserts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .config import SEC_PER_DAY, SEC_PER_YEAR
from .distribution import Distribution
from .kernels import register_step
from .regression import Forecaster
from .seasonality import Seasonality

MODEL = None  # module-level default, set by load_params()


# ================================================================= state container
@dataclass
class State:
    ts: int                       # timestamp of the last bar folded in (epoch seconds)
    last_close: float
    v: np.ndarray                 # (K,) register variance rates, business-day units
    act: float = 1.0              # activity EWMA (hybrid clock only)
    n_updates: int = 0
    # Partial register step, only ever non-trivial when registers.step_seconds > 1.
    # The bank is fed one aggregated `step_seconds`-second return, so the 1s bars have
    # to be buffered until a block closes. See Model.update.
    acc_r: float = 0.0            # sum of the 1s log returns in the open block
    acc_dT: float = 0.0           # business time accumulated in the open block
    acc_n: int = 0                # bars in the open block
    acc_valid: bool = True        # every bar in the open block was usable

    def to_flat(self) -> dict:
        d = {"ts": float(self.ts), "last_close": float(self.last_close),
             "act": float(self.act), "n_updates": float(self.n_updates),
             "acc_r": float(self.acc_r), "acc_dT": float(self.acc_dT),
             "acc_n": float(self.acc_n), "acc_valid": float(self.acc_valid)}
        for k, x in enumerate(np.asarray(self.v, dtype=np.float64)):
            d["v%d" % k] = float(x)
        return d

    @classmethod
    def from_flat(cls, d: dict) -> "State":
        ks = sorted(int(k[1:]) for k in d if k.startswith("v") and k[1:].isdigit())
        v = np.array([d["v%d" % k] for k in ks], dtype=np.float64)
        return cls(int(d["ts"]), float(d["last_close"]), v, float(d.get("act", 1.0)),
                   int(d.get("n_updates", 0)), float(d.get("acc_r", 0.0)),
                   float(d.get("acc_dT", 0.0)), int(d.get("acc_n", 0)),
                   bool(d.get("acc_valid", True)))

    def copy(self) -> "State":
        return State(self.ts, self.last_close, np.array(self.v, copy=True), self.act,
                     self.n_updates, self.acc_r, self.acc_dT, self.acc_n,
                     self.acc_valid)


@dataclass
class ForecastResult:
    h: int
    dT: float
    rv: float                     # realised variance over (t, t+h], QLIKE-optimal scale
    rv_mean: float                # the same on the mean-matching scale (literal E[RV])
    sigma_ann: float              # annualised volatility
    nu: float
    loc: float
    scale: float
    dist: Distribution = field(repr=False, default=None)
    x: float = 0.0

    def prob_above(self, log_moneyness) -> float:
        """P(S_{t+h} > K | S_t) for log_moneyness = log(K / S_t)."""
        z = np.asarray(log_moneyness, dtype=np.float64) / np.sqrt(max(self.rv, 1e-300))
        return float(1.0 - self.dist.cdf(z, self.x)) if np.isscalar(log_moneyness) \
            else 1.0 - self.dist.cdf(z, np.full(np.shape(z), self.x))


# ========================================================================== the model
class Model:
    """Everything the streaming path needs, loaded from params.json."""

    def __init__(self, params: dict):
        self.params = params
        self.version = params.get("version", "0")
        self.seas = Seasonality.from_dict(params["seasonality"])
        self.forecaster = Forecaster.from_dict(params["forecaster"])
        # v2 may ship a per-horizon-band loss: a separate regression fitted by direct
        # QLIKE minimisation for horizons below the cut, log-space least squares above.
        # Both are stored; `_fc_for` picks by the clock horizon, which is always known.
        self.forecaster_short = (Forecaster.from_dict(params["forecaster_short"])
                                 if params.get("forecaster_short") else None)
        self.band_cut_s = float(params.get("forecaster_band_cut_s", 0.0))
        self.dist = Distribution.from_dict(params["distribution"])
        r = params["registers"]
        self.hl = np.asarray(r["half_life_business"], dtype=np.float64)
        self.v_init = np.asarray(r["initial_value"], dtype=np.float64)
        # The bank was fitted on `step_seconds`-second aggregated returns. Until
        # 2026-09 the streaming path ignored this field and folded every 1s bar, which
        # on a 5-second bank produced 20-28% lower integrated variance than the batch
        # path the coefficients came from. v2.1 ships step_seconds = 1 so the bug was
        # dormant, but it must not be able to come back: `update` now buffers.
        self.step_seconds = max(int(r.get("step_seconds", 1)), 1)
        c = params["clock"]
        self.clock_kind = c.get("kind", "business")
        self.gamma = float(c.get("gamma", 0.0))
        self.act_hl_s = float(c.get("act_hl_s", 120.0))
        self.blend_tau_s = float(c.get("blend_tau_s", 0.0)) or self.act_hl_s
        self.act_lam = float(np.exp(-np.log(2.0) / max(self.act_hl_s, 1e-9)))
        self.act_floor = float(c.get("act_floor", 0.05))
        self.act_cap = float(c.get("act_cap", 10.0))
        self.act_prof = (Seasonality.from_dict(params["activity_profile"])
                         if params.get("activity_profile") else None)
        self.max_stale_s = int(params.get("data", {}).get("max_stale_s", 5))
        # --- v2: forward-variance density, the return ACF kernel and the TWAP tails.
        # All optional, so a v1 params.json still loads and every v1 call path is
        # unchanged; the TWAP methods raise rather than guess if they are absent.
        from .forward import ForwardModel
        self.forward = (ForwardModel.from_dict(params["forward"])
                        if params.get("forward") else None)
        self.acf = {k: np.asarray(v, dtype=np.float64)
                    for k, v in (params.get("acf_kernel") or {}).items()}
        self.twap_tails = {int(k): v
                           for k, v in (params.get("twap_tail") or {}).items()}
        # rolling 1s cache of s(t)
        self._c0 = None
        self._s = None

    # ---- seasonal cache ---------------------------------------------------
    def _cache(self, t0: int, t1: int):
        if self._c0 is not None and t0 >= self._c0 and t1 < self._c0 + self._s.size:
            return
        lo = int(t0) - 7200
        hi = int(t1) + 3 * 86400
        self._c0 = lo
        self._s = self.seas.s(np.arange(lo, hi, dtype=np.int64))

    def s_at(self, ts: int) -> float:
        self._cache(int(ts), int(ts) + 1)
        return float(self._s[int(ts) - self._c0])

    def dT_ahead(self, t: int, h: int, act: float = 1.0) -> float:
        """Business time over (t, t+h], deterministic profile + decayed activity kick."""
        if self.clock_kind == "ratio":
            return h / SEC_PER_DAY
        t = int(t); h = int(h)
        self._cache(t, t + h + 1)
        i = t - self._c0
        seg = self._s[i + 1:i + h + 1]
        base = float(seg.sum()) / SEC_PER_DAY
        if self.gamma == 0.0 or act == 1.0:
            return base
        lam = float(np.exp(-1.0 / self.blend_tau_s))
        w = lam ** np.arange(1, h + 1, dtype=np.float64)
        kick = act ** self.gamma - 1.0
        return base + kick * float(np.dot(seg, w)) / SEC_PER_DAY

    # ---- state ------------------------------------------------------------
    def new_state(self, ts: int, close: float, v=None) -> State:
        st = State(int(ts), float(close),
                   np.array(self.v_init if v is None else v, dtype=np.float64), 1.0, 0)
        if self.step_seconds > 1:
            # Bar 0 seeds the state and is a no-op in the batch path (r = 0, dT = 0,
            # invalid). `kernels.aggregate_steps` blocks from the first bar of the
            # sample, so the seed has to occupy the first slot of block 0 or every
            # block boundary downstream would be off by one.
            st.acc_n, st.acc_valid = 1, False
        return st

    def update(self, state: State, ts: int, close: float, n_trades: float = 0.0,
               quote_volume: float = 0.0, gap: bool = False) -> State:
        """Fold one 1s bar into the state. O(K). Returns the same (mutated) State."""
        ts = int(ts)
        dt = ts - state.ts
        if dt <= 0:
            raise ValueError("bars must arrive in strictly increasing time order")
        stale = dt > self.max_stale_s
        r = 0.0
        if state.last_close > 0 and close > 0 and not gap and not stale:
            r = float(np.log(close / state.last_close))
        valid = (not gap) and (not stale) and np.isfinite(r)

        s_now = self.s_at(ts)
        if self.gamma > 0.0 and self.act_prof is not None:
            exp_sec = float(self.act_prof.s(np.array([ts]))[0]) / 60.0
            x = n_trades if self.params["clock"].get("activity", "trades") == "trades" \
                else quote_volume
            ratio = x / exp_sec if exp_sec > 1e-12 else 1.0
            a = self.act_lam * state.act + (1.0 - self.act_lam) * ratio
            state.act = min(max(a, self.act_floor), self.act_cap)
            rate = s_now * state.act ** self.gamma
        else:
            rate = s_now
        if self.clock_kind == "ratio":
            if self.step_seconds > 1:
                raise NotImplementedError(
                    "registers.step_seconds > 1 is not defined for the ratio clock: "
                    "r^2/s is not additive across bars")
            dT = dt / SEC_PER_DAY
            r2 = (r * r) / max(s_now, 1e-12)
        else:
            dT = rate * dt / SEC_PER_DAY
            r2 = r * r
        if self.step_seconds == 1:
            register_step(state.v, r2, dT, bool(valid), self.hl)
        else:
            # Mirror kernels.aggregate_steps exactly: sum the returns (invalid bars
            # contribute zero), sum the business time, AND the validity flags, and fold
            # the squared block return once the block closes.
            state.acc_r += r if valid else 0.0
            state.acc_dT += dT
            state.acc_n += 1
            if not valid:
                state.acc_valid = False
            if state.acc_n >= self.step_seconds:
                register_step(state.v, state.acc_r * state.acc_r, state.acc_dT,
                              bool(state.acc_valid), self.hl)
                state.acc_r = 0.0
                state.acc_dT = 0.0
                state.acc_n = 0
                state.acc_valid = True
        state.ts = ts
        state.last_close = float(close)
        state.n_updates += 1
        return state

    # ---- forecast ---------------------------------------------------------
    def forecast(self, state: State, t_now: int, h: int) -> ForecastResult:
        t_now = int(t_now)
        v = state.v
        if t_now > state.ts:
            # no observations since the last update: decay the bank forward
            v = np.array(v, copy=True)
            dT = self.dT_ahead(state.ts, t_now - state.ts, state.act)
            register_step(v, 0.0, max(dT, 1e-15), False, self.hl)
        dTh = self.dT_ahead(t_now, h, state.act)
        x = float(np.log(max(dTh, 1e-12)))
        logv = np.log(np.maximum(v, 1e-300))
        fc = self._fc_for(h)
        rv = float(fc.predict_rv(np.array([x]), logv[None, :])[0])
        rv_mean = float(fc.predict_rv_mean(np.array([x]), logv[None, :])[0])
        nu, loc, sc = self.dist.params(np.array([x]))
        return ForecastResult(h=int(h), dT=dTh, rv=rv, rv_mean=rv_mean,
                              sigma_ann=float(np.sqrt(rv / h * SEC_PER_YEAR)),
                              nu=float(nu[0]), loc=float(loc[0]), scale=float(sc[0]),
                              dist=self.dist, x=x)

    def _fc_for(self, h):
        if self.forecaster_short is not None and self.band_cut_s > 0                 and float(h) < self.band_cut_s:
            return self.forecaster_short
        return self.forecaster

    def term_structure(self, state: State, t_now: int, horizons) -> list:
        return [self.forecast(state, t_now, h) for h in horizons]

    # ---- v2: forward variance and TWAP -----------------------------------
    def forward_curve(self, state: State, t_now: int, n_seconds: int) -> np.ndarray:
        """xi(t, u) for u = 1 .. n_seconds: the variance of each future 1-second return.

        Requires a v2 params.json (one carrying `forward`). The values are increments of
        the same integral `forecast()` reads off, so `forward_curve(...).sum()` equals
        `forecast(state, t_now, n_seconds).rv` up to the bias correction, exactly.
        """
        if self.forward is None:
            raise RuntimeError("this params.json has no forward-variance model; "
                               "rebuild with scripts/08_v2_params.py")
        t_now = int(t_now)
        n = int(n_seconds)
        self._cache(t_now, t_now + n + 1)
        i = t_now - self._c0
        seg = self._s[i + 1:i + n + 1] / SEC_PER_DAY
        if self.gamma > 0.0 and state.act != 1.0:
            lam = float(np.exp(-1.0 / self.blend_tau_s))
            seg = seg * (1.0 + (state.act ** self.gamma - 1.0)
                         * lam ** np.arange(1, n + 1, dtype=np.float64))
        dT_cum = np.cumsum(seg)
        logv = np.log(np.maximum(state.v, 1e-300))
        return self.forward.forward_curve_business(dT_cum, logv)

    def twap_variance(self, state: State, t_now: int, window_start: int,
                      window_end: int, observed_prices=None,
                      settlement_price: str = "last") -> dict:
        """Variance of the TWAP of 1-second prices over [window_start, window_end].

        `window_start`/`window_end` are epoch seconds; the TWAP is over the N = end -
        start second-end prices. `observed_prices` is whatever part of the window has
        already happened (its length m is what matters; the values are used only for the
        realised part of the average). Returns the variance of the *unknown* part, the
        per-second contributions, and a `prob_above` callable.

        `settlement_price` selects the autocorrelation kernel. Only "last" is fitted
        here, from Binance last-trade 1-second bars. An index, mark or oracle TWAP is a
        smoother series with its own - materially different - kernel, and reusing this
        one for it would be wrong; the report says so and the code refuses rather than
        silently substituting.
        """
        if self.forward is None or self.acf is None:
            raise RuntimeError("this params.json has no forward/ACF model")
        if settlement_price not in self.acf:
            raise ValueError(
                "no autocorrelation kernel calibrated for settlement_price=%r; "
                "available: %s. An index/mark/oracle TWAP needs its own series to "
                "calibrate against." % (settlement_price, sorted(self.acf)))
        N = int(window_end) - int(window_start)
        m = 0 if observed_prices is None else len(observed_prices)
        m = max(m, int(t_now) - int(window_start))
        n_rem = N - m
        if n_rem <= 0:
            return {"variance": 0.0, "contributions": np.zeros(0), "N": N, "m": m}
        xi = self.forward_curve(state, int(t_now), n_rem)
        rho = np.asarray(self.acf[settlement_price], dtype=np.float64)
        from .forward import twap_variance_from_xi
        var, contrib = twap_variance_from_xi(xi, rho, N, m)
        nu, loc, sc = self.twap_tail(N)

        def prob_above(log_moneyness, observed_mean_log=0.0):
            """P(TWAP > K), with log_moneyness = log(K / S_t).

            `observed_mean_log` is the contribution of the already-observed part of the
            window to log(TWAP/S_t), so a mid-window quote prices the remaining part
            against what is left of the distance to the strike.
            """
            z = (float(log_moneyness) - float(observed_mean_log)) / max(
                np.sqrt(var), 1e-300)
            return float(1.0 - stats_t_cdf(z, nu, loc, sc))

        return {"variance": float(var), "contributions": contrib, "N": N, "m": m,
                "sd": float(np.sqrt(var)), "nu": nu, "loc": loc, "scale": sc,
                "prob_above": prob_above, "settlement_price": settlement_price}

    def twap_tail(self, N: int):
        """(nu, loc, scale) of the standardised TWAP residual for a window of N seconds."""
        if not self.twap_tails:
            return 1e6, 0.0, 1.0
        ns = np.array(sorted(self.twap_tails), dtype=np.float64)
        g = np.log(np.maximum(float(N), 1.0))
        def _i(key):
            v = np.array([self.twap_tails[int(n)][key] for n in ns])
            return float(np.interp(g, np.log(ns), v))
        return _i("nu"), _i("loc"), _i("scale")


# ============================================================== module-level helpers
def load_params(path="output/params.json", set_default: bool = True) -> Model:
    global MODEL
    with open(path) as fh:
        m = Model(json.load(fh))
    if set_default:
        MODEL = m
    return m


def forecast(state: State, t_now: int, h: int, model: Model = None) -> ForecastResult:
    """Forecast the distribution of log(S_{t+h}/S_t) from a persisted state."""
    m = model or MODEL
    if m is None:
        raise RuntimeError("no model loaded; call rvforecast.load_params(path) first")
    return m.forecast(state, t_now, h)


def stats_t_cdf(z, nu, loc, scale):
    from scipy import stats
    return stats.t.cdf(z, df=nu, loc=loc, scale=scale)


def forward_curve(state: State, t_now: int, n_seconds: int,
                  model: Model = None) -> np.ndarray:
    """Per-second forward variance over the next `n_seconds` (brief section 11.3)."""
    m = model or MODEL
    if m is None:
        raise RuntimeError("no model loaded; call rvforecast.load_params(path) first")
    return m.forward_curve(state, t_now, n_seconds)


def twap_variance(state: State, t_now: int, window_start: int, window_end: int,
                  observed_prices=None, settlement_price: str = "last",
                  model: Model = None) -> dict:
    """Variance of the TWAP over a window, mid-window supported (section 11.3)."""
    m = model or MODEL
    if m is None:
        raise RuntimeError("no model loaded; call rvforecast.load_params(path) first")
    return m.twap_variance(state, t_now, window_start, window_end, observed_prices,
                           settlement_price)


def warm_start(state: State, bars, model: Model = None) -> State:
    """Replay an iterable of (ts, close, n_trades, quote_volume, gap) into a state."""
    m = model or MODEL
    if m is None:
        raise RuntimeError("no model loaded; call rvforecast.load_params(path) first")
    for b in bars:
        ts, close = b[0], b[1]
        nt = b[2] if len(b) > 2 else 0.0
        qv = b[3] if len(b) > 3 else 0.0
        gp = bool(b[4]) if len(b) > 4 else False
        state = m.update(state, ts, close, nt, qv, gp)
    return state
