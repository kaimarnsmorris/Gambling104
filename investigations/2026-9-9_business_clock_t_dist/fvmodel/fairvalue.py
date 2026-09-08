"""`fair_value(state, market)` - one call that prices every settlement type.

    market = {kind, expiry_ts, strike, received_prints, book_snapshot}

and the answer comes back with everything that went into it: the probability, the
variance, the standardised distance to the strike, the known/unknown split of the
settlement, the order-book location term, the conditional residual mean, and the tail
parameters used. Nothing is hidden inside, because every one of those is a thing the
autotrader will eventually want to see disagree with the market.

The algebra, once
-----------------
Write `P_ref = exp(x_t + b_t)`: the blended input translated to Chainlink's level, which
is the model's view of where a print landing right now would be. Each settlement
component `j` (a print, or a second of a perp TWAP) has

    g_j = log(C_j / P_ref) = d_j  +  sum_k W_k^(j) r_k  +  eps_j

with `d_j` the deterministic carry - for a future Chainlink print, `lam^K_j (F_t - x_t)`,
the amount by which the filter still trails the price; for a print already stamped but
not yet received, the filter value we can compute for it directly.

Splitting the components into received `R`, in transit `T` and future `F`,

    settlement / P_ref  =  (1/N) sum_R C_j/P_ref  +  omega * (1 + g_bar)

exactly for `R` and to first order for the rest, with `omega = (N - |R|)/N`. Setting that
equal to `K / P_ref` and moving every known piece to the right gives

    y*  =  (K/P_ref - A_R)/omega - 1 - d_bar - m_Y - c * eps_last

and `P(up) = 1 - F_tail(y* / sqrt(Var_Y))`.

For a single perp print the whole thing collapses: `N = 1`, nothing is known, `d = 0`,
`eps = 0`, and the linearisation is unnecessary, so that case is done in logs exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .alpha import AlphaModel
from .chainlink import EpsModel, FilterParams, PrintFilter, eps_conditional
from .curve import dT_at, forward_block, unconditional_xi
from .overrides import Overrides, cap_m_Y, quoted_prob, xi_adjust
from .tails import SettlementTail
from .variance import PAD, block_length, settlement_variance
from .weights import settlement_weights

BP = 1e4
ALPHA_MAX = 300      # clock seconds past which beta(dT) is certainly saturated


# ==================================================================== configuration
@dataclass
class Switches:
    """Every ablation the evaluation runs, in one place, defaulting to the full model."""
    eps_mode: str = "full"            # full | unconditional | iid | none
    use_basis: bool = True
    use_blend: bool = True            # False: no input-variance ratio (see 13_eval)
    use_reconstruction: bool = True   # reconstruct prints stamped but not received
    use_alpha: bool = True
    rho_mode: str = "conditional"     # conditional | unconditional | zero
    tail_mode: str = "fitted"         # fitted | normal | v2_twap
    composed_weights: bool = True     # False: the un-composed triangular ramp
    market_observable: bool = False   # no perp feed: prints only, no reconstruction
    jensen: bool = True               # second-order term for an arithmetic average

    def label(self) -> str:
        d = {"eps_mode": "full", "use_basis": True, "use_blend": True,
             "use_reconstruction": True, "use_alpha": True, "rho_mode": "conditional",
             "tail_mode": "fitted", "composed_weights": True,
             "market_observable": False, "jensen": True}
        off = [k for k, v in d.items() if getattr(self, k) != v]
        return "full" if not off else "+".join(
            "%s=%s" % (k, getattr(self, k)) for k in off)


@dataclass
class FairValueModel:
    """The v2.1 forecaster plus everything this project adds on top of it."""
    v2: object                                   # rvforecast.streaming.Model
    fp: FilterParams
    eps: EpsModel
    alpha: AlphaModel = field(default_factory=AlphaModel.zero)
    rho: dict = field(default_factory=dict)      # 'all' plus 'act0'/'act1'/'act2'
    act_cuts: tuple = (0.75, 1.25)               # activity-factor tercile cuts
    tails: dict = field(default_factory=dict)    # kind -> SettlementTail
    input_var_ratio: float = 1.0                 # Var(dlog X) / Var(dlog perp)
    # The short-business-time cap on the forward curve (report 11.2). Off at 0.0.
    # Turning it on is a model change: it invalidates every fitted table downstream.
    xi_cap_c: float = 0.0
    xi_cap_i: int = None
    ov: object = None                            # fvmodel.overrides.Overrides

    def rho_for(self, act: float, mode: str = None) -> np.ndarray:
        mode = mode or (self.ov.rho_kernel if self.ov else "conditional")
        if mode == "off":
            z = np.zeros(2)
            z[0] = 1.0
            return z
        if mode != "conditional" or "act0" not in self.rho:
            return self.rho.get("all", np.array([1.0]))
        k = 0 if act < self.act_cuts[0] else (1 if act < self.act_cuts[1] else 2)
        return self.rho["act%d" % k]

    def tail_for(self, kind: str) -> SettlementTail:
        return self.tails.get(kind) or SettlementTail.normal(kind)


@dataclass
class Market:
    kind: str
    expiry_ts: int
    strike: float
    received_prints: tuple = ()       # ((stamp_ts, value), ...) inside the window
    book_snapshot: dict = None        # {"imbalance": I, "px_age_s": a}
    L: int = 60                       # settlement window, seconds


@dataclass
class FairValue:
    p_up: float
    var_y: float
    y_star: float
    sd_y: float
    n_remaining: int
    n_components: int
    n_known: int
    n_in_transit: int
    omega: float
    known_value: float                # A_R, the settled share of settlement / P_ref
    carry: float                      # d_bar
    m_Y: float
    eps_bar: float
    var_eps: float
    var_basis: float
    z: float                          # log of the business time left
    tail: tuple                       # (nu, mu, sigma)
    p_ref: float
    # Diagnostics for the shadow period, not inputs to anything priced here: the two
    # basis trackers side by side, so the 60 s / 15 s question (report 11.3) can be
    # settled on live lead-k prediction instead of on a fitted sweep.
    basis: float = float("nan")
    basis_alt: float = float("nan")
    switches: str = "full"
    p_model: float = float("nan")
    p_quoted: float = float("nan")

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["tail"] = list(self.tail)
        return d


# ========================================================================== the call
def fair_value(state, market: Market, model: FairValueModel,
               sw: Switches = None, t_now: int = None) -> FairValue:
    """Price one market from one state. See the module docstring for the algebra."""
    sw = sw or Switches()
    t = int(t_now if t_now is not None else state.v2.ts)
    n = int(market.expiry_ts) - t
    kind = market.kind
    Lw = int(market.L)

    if n <= 0:
        raise ValueError("the market has already expired at this quote time")

    # ---- the weights ---------------------------------------------------------
    if kind == "chainlink_twap60" and not sw.composed_weights:
        s = settlement_weights("perp_twap", n, L=Lw)
        s.kind = "chainlink_twap60"
        s.carry = 0.0
    else:
        s = settlement_weights(kind, n, model.fp.to_dict(), L=Lw)

    # ---- what is already known ----------------------------------------------
    x_t = state.filt.x if kind == "chainlink_twap60" else np.log(state.v2.last_close)
    b_t = (state.filt.b if (kind == "chainlink_twap60" and sw.use_basis
                            and np.isfinite(state.filt.b)) else 0.0)
    p_ref = float(np.exp(x_t + b_t))

    n_known = 0
    A_R = 0.0
    d_sum = 0.0
    unknown_ages = []
    n_transit = 0
    if kind == "chainlink_twap60":
        T = int(market.expiry_ts)
        got = {int(k): float(v) for k, v in market.received_prints}
        lam = model.fp.lam
        for j in range(s.n_components):
            stamp = T - j
            if stamp in got:
                n_known += 1
                A_R += got[stamp] / p_ref
                continue
            K = n - j - model.fp.delta_s
            if K > 0.0:                                  # future print
                d_sum += (lam ** K) * (state.filt.xf - x_t) if sw.composed_weights else 0.0
            elif sw.use_reconstruction and not sw.market_observable:
                n_transit += 1
                f = state.filt.filtered_at(stamp - model.fp.delta_s)
                d_sum += (f - x_t) if np.isfinite(f) else 0.0
            else:
                n_transit += 1                           # unknown and not reconstructed
            unknown_ages.append(stamp)
        A_R /= s.n_components
    elif kind == "perp_twap":
        n_known = s.n_known
        # the observed part of a perp TWAP is exact and comes in with the market
        got = {int(k): float(v) for k, v in market.received_prints}
        A_R = sum(v / p_ref for v in got.values()) / s.n_components
        n_known = len(got)

    n_unknown = s.n_components - n_known
    omega = n_unknown / s.n_components if s.n_components else 0.0
    if omega <= 0:
        raise ValueError("nothing left unknown; the market has settled")
    W = s.W_raw / omega if hasattr(s, "W_raw") else s.W * (s.omega / omega)
    d_bar = d_sum / s.n_components / omega

    # ---- the variance of the return part ------------------------------------
    # The register bank is NOT touched: kappa_vol moved to the forward curve
    # (brief section 4.2), which is what makes the bank variant-invariant and so
    # cacheable across every variant run. See the spec, section 4.3.
    v_state = state.v2
    m_blk = block_length(kind if sw.composed_weights else "perp_twap", n, Lw, PAD)
    iv_head, xi = forward_block(model.v2, v_state, t, n, m_blk,
                                cap_c=model.xi_cap_c, cap_i=model.xi_cap_i)
    ov = model.ov or Overrides()
    if xi.size:
        u_blk = np.arange(n - xi.size + 1, n + 1, dtype=np.int64)
        dT_blk = dT_at(model.v2, v_state, t, u_blk) * 86400.0     # business seconds
        xi_bar = (unconditional_xi(model.v2, dT_blk / 86400.0)
                  if ov.shrink_w != 0.0 else None)
        xi = xi_adjust(ov, xi, dT_blk, xi_bar)
        # the head integral carries the same uniform scaling; the short-end and
        # shrink knobs are defined on the block, where the settlement weights live
        if ov.kappa_vol != 0.0:
            iv_head = iv_head * np.exp(2.0 * ov.kappa_vol)
    rho = model.rho_for(state.v2.act, sw.rho_mode)
    ratio = model.input_var_ratio if (kind == "chainlink_twap60" and sw.use_blend) else 1.0
    var_ret = float(settlement_variance(np.array([iv_head]), xi[None, :] * ratio,
                                        W, rho, n)[0])

    # ---- the residual --------------------------------------------------------
    eps_bar, var_eps, var_basis = 0.0, 0.0, 0.0
    if kind == "chainlink_twap60" and sw.eps_mode != "none" and unknown_ages:
        v_loc = float(np.sqrt(max(state.v2.v[model.eps.reg_index], 0.0)
                              * max(dT_at(model.v2, v_state, t, np.array([1]))[0], 1e-12)))
        sig = float(model.eps.sigma_at(v_loc))
        ref_ts = state.filt.eps_last_ts if state.filt.eps_last_ts is not None else t
        ages = np.array([a - ref_ts for a in unknown_ages], dtype=np.float64)
        w = np.full(ages.size, 1.0 / max(n_unknown, 1))
        mode = sw.eps_mode if not sw.market_observable else "unconditional"
        c, var_eps = eps_conditional(model.eps, ages, w, sig, mode)
        eps_bar = c * state.filt.eps_last
        if sw.use_basis:
            from .build import basis_drift_window_var
            var_basis = basis_drift_window_var(model, ages, w)

    var_y = max(var_ret + var_eps + var_basis, 1e-300)

    # ---- the order-book location term ---------------------------------------
    m_Y = 0.0
    if sw.use_alpha and market.book_snapshot:
        # beta saturates, so only the seconds *just after* the quote origin carry a
        # non-zero increment; past ALPHA_MAX the curve is flat and adds nothing however
        # long the market runs
        na = min(n, ALPHA_MAX)
        u = np.arange(1, na + 1, dtype=np.int64)
        dTc = dT_at(model.v2, v_state, t, u)
        m_Y = float(model.alpha.m_Y(market.book_snapshot.get("imbalance", 0.0),
                                    W[:na], dTc[None, :],
                                    market.book_snapshot.get("px_age_s")))
    m_Y = cap_m_Y(ov, m_Y, var_y)

    # ---- the strike ----------------------------------------------------------
    if kind == "perp_single":
        y_raw = float(np.log(market.strike / p_ref))
    else:
        # an arithmetic mean of prices: the linearised y carries half the mean square
        # of the component moves in its own mean, so the strike side takes it back out
        y_raw = (market.strike / p_ref - A_R) / omega - 1.0
        if sw.jensen:
            y_raw -= 0.5 * var_y
    y_star = y_raw - d_bar - m_Y - eps_bar

    # ---- the tail ------------------------------------------------------------
    D = float(dT_at(model.v2, v_state, t, np.array([n]))[0])
    z = float(np.log(max(D, 1e-12)))
    tail = model.tail_for(kind)
    nu, mu, sg = tail.params(z)
    p_model = float(tail.prob_up(y_star, var_y, z))
    p_quoted = float(quoted_prob(ov, p_model))

    return FairValue(p_up=p_model, var_y=var_y, y_star=y_star, sd_y=float(np.sqrt(var_y)),
                     n_remaining=n, n_components=s.n_components, n_known=n_known,
                     n_in_transit=n_transit, omega=omega, known_value=A_R,
                     carry=d_bar, m_Y=m_Y, eps_bar=eps_bar, var_eps=var_eps,
                     var_basis=var_basis, z=z,
                     tail=(float(nu), float(mu), float(sg)), p_ref=p_ref,
                     basis=float(getattr(state.filt, "b", float("nan"))),
                     basis_alt=float(getattr(state.filt, "b_alt", float("nan"))),
                     switches=sw.label(), p_model=p_model, p_quoted=p_quoted)


# ============================================================== the composite state
@dataclass
class FVState:
    """v2.1's register state plus the print filter and the last book snapshot."""
    v2: object
    filt: PrintFilter
    book: dict = None

    def to_flat(self) -> dict:
        d = self.v2.to_flat()
        d.update(self.filt.to_flat())
        if self.book:
            d["bk_imb"] = float(self.book.get("imbalance", 0.0))
            d["bk_age"] = float(self.book.get("px_age_s", np.nan))
        return d

    @classmethod
    def from_flat(cls, d: dict, model: FairValueModel) -> "FVState":
        from rvforecast.streaming import State
        book = None
        if "bk_imb" in d:
            book = {"imbalance": d["bk_imb"], "px_age_s": d.get("bk_age", np.nan)}
        return cls(State.from_flat(d), PrintFilter.from_flat(d, model.fp), book)
