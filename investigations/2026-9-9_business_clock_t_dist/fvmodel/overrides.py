"""Every knob the model exposes, and the one place their semantics live.

Two layers, because six application points cannot honestly be one call site.

`apply_overrides` is the build-time layer: everything statically resolvable is
folded into a new model object. The per-tick transforms below it - `xi_adjust`,
`cap_m_Y`, `quoted_prob` - are pure functions called by BOTH the scalar and the
vectorised path, which is what stops those two implementations drifting apart.

Nothing here mutates the model it is given.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields

import numpy as np

RHO_KERNELS = ("conditional", "unconditional", "off")
BASIS_TRACKERS = ("main", "alt", "off")
INFORMATION_SETS = ("full", "prints_only")
TAIL_FAMILIES = ("t", "normal")
NU_FLOOR = 2.5


@dataclass(frozen=True)
class Overrides:
    """All defaults are the off value. See variants/README.md for sign conventions."""
    kappa_vol: float = 0.0
    kappa_vol_short: float = 0.0
    shrink_w: float = 0.0
    shrink_decay_s: float = 60.0
    rho_kernel: str = "conditional"
    eps_scale: float = 1.0
    eps_condition: bool = True
    basis_tracker: str = "main"
    information_set: str = "full"
    reconstruct_in_transit: bool = True
    kappa_tail: float = 0.0
    tail_scale: float = 1.0
    tail_family: str = "t"
    alpha_scale: float = 1.0
    alpha_scale_age: float = 1.0
    alpha_cap_sd: float | None = None
    w_spot: float | None = None
    xi_cap_c: float | None = None
    temperature: float = 1.0

    def __post_init__(self):
        for name, allowed in (("rho_kernel", RHO_KERNELS),
                              ("basis_tracker", BASIS_TRACKERS),
                              ("information_set", INFORMATION_SETS),
                              ("tail_family", TAIL_FAMILIES)):
            v = getattr(self, name)
            if v not in allowed:
                raise ValueError("%s must be one of %s, got %r" % (name, allowed, v))
        if self.temperature <= 0:
            raise ValueError("temperature must be positive, got %r" % self.temperature)
        if self.eps_scale < 0 or self.tail_scale <= 0:
            raise ValueError("eps_scale must be >= 0 and tail_scale > 0")
        if self.w_spot is not None:
            from .config import filter_by_w
            grid = [r[0] for r in filter_by_w()]
            if not any(abs(self.w_spot - g) < 1e-9 for g in grid):
                raise ValueError(
                    "w_spot must be a value in tables/filter_by_w.json %s, got %r; "
                    "the filter is re-read from that table, never refitted"
                    % (grid, self.w_spot))

    # ---- io ----------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Overrides":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known - {"version", "source"}
        if unknown:
            raise ValueError("unknown override key(s): %s; known keys are %s"
                             % (sorted(unknown), sorted(known)))
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    def non_default(self) -> dict:
        d = Overrides()
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) != getattr(d, f.name)}

    def label(self) -> str:
        nd = self.non_default()
        return "baseline" if not nd else "+".join(
            "%s=%s" % (k, v) for k, v in sorted(nd.items()))


# ==================================================================== build time
def _scaled_tail(tail, ov: "Overrides"):
    """kappa_tail, tail_scale and tail_family, applied to a stored (nu, mu, sigma).

    kappa_tail moves the shape and holds Var(z) = nu/(nu-2) * sigma^2 fixed, so it is
    a pure tail-thickness knob and does not smuggle in a width change. tail_scale is
    the width knob, and scales mu with sigma (spec ruling R2) so that it rescales the
    standardised variable as a whole - which is what makes it exactly equivalent to
    kappa_vol at the other end of the pipeline.
    """
    t = copy.deepcopy(tail)
    if ov.tail_family == "normal":
        t.family = "normal"
        return t
    if ov.kappa_tail != 0.0:
        nu0 = np.asarray(t.nu, dtype=np.float64)
        nu1 = np.maximum(2.0 + (nu0 - 2.0) * np.exp(ov.kappa_tail), NU_FLOOR)
        t.sigma = t.sigma * np.sqrt((nu0 / (nu0 - 2.0)) / (nu1 / (nu1 - 2.0)))
        t.nu = nu1
    if ov.tail_scale != 1.0:
        t.sigma = t.sigma * ov.tail_scale
        t.mu = t.mu * ov.tail_scale
    return t


def apply_overrides(model, ov: "Overrides"):
    """A new model with every statically resolvable override folded in."""
    from .chainlink import FilterParams
    from .curve import slow_index

    m = copy.copy(model)                       # shallow: v2 and rho are read-only
    m.ov = ov

    # ---- the input blend, and the filter that goes with it -------------------
    if ov.w_spot is not None:
        from .config import filter_by_w
        row = min(filter_by_w(), key=lambda r: abs(r[0] - ov.w_spot))
        fp = FilterParams.from_dict(dict(model.fp.to_dict(),
                                         w_spot=row[0], tau_s=row[1],
                                         delta_s=row[2], p_stamp_late=row[3]))
        fp.basis_hl_s = model.fp.basis_hl_s
        fp.basis_hl_alt_s = model.fp.basis_hl_alt_s
        fp.lag_s = model.fp.lag_s
        m.fp = fp
    else:
        m.fp = copy.copy(model.fp)

    # ---- the eps process (R1: scale the process, not only its sd) ------------
    m.eps = copy.deepcopy(model.eps)
    if ov.eps_scale != 1.0:
        m.eps.sigma_bp = model.eps.sigma_bp * ov.eps_scale

    # ---- the location term ---------------------------------------------------
    m.alpha = copy.deepcopy(model.alpha)
    m.alpha.beta0_bp = model.alpha.beta0_bp * ov.alpha_scale
    m.alpha.beta1_bp = model.alpha.beta1_bp * ov.alpha_scale * ov.alpha_scale_age

    # ---- the tails -----------------------------------------------------------
    m.tails = {k: _scaled_tail(v, ov) for k, v in model.tails.items()}

    # ---- the forward-curve cap ----------------------------------------------
    if ov.xi_cap_c is not None:
        m.xi_cap_c = float(ov.xi_cap_c)
        m.xi_cap_i = slow_index(np.asarray(
            model.v2.params["registers"]["half_life_business"], dtype=np.float64))
    else:
        m.xi_cap_c = 0.0
        m.xi_cap_i = None
    return m


# ===================================================================== per tick
def xi_adjust(ov: "Overrides", xi, dT_business_s, xi_bar=None):
    """The vol overrides, applied to the forward curve.

    `xi` is the per-second forward variance over the exact block, `dT_business_s` the
    business age in seconds of each of those seconds measured from the quote origin
    (spec R7: business days times 86400), broadcastable against `xi`.

    Order matters and is the brief's: kappa_vol and kappa_vol_short move log xi, then
    the shrink pulls the MOVED curve toward the seasonal-unconditional level `xi_bar`
    (spec R6). Applying the shrink first would let a vol override escape it.

    Returns `xi` itself when nothing is on, so the default path allocates nothing and
    is bit-identical.
    """
    if ov.kappa_vol == 0.0 and ov.kappa_vol_short == 0.0 and ov.shrink_w == 0.0:
        return xi
    logxi = np.log(np.maximum(np.asarray(xi, dtype=np.float64), 1e-300))
    D = np.maximum(np.asarray(dT_business_s, dtype=np.float64), 1e-12)
    if ov.kappa_vol != 0.0:
        logxi = logxi + 2.0 * ov.kappa_vol
    if ov.kappa_vol_short != 0.0:
        logxi = logxi + 2.0 * ov.kappa_vol_short * np.minimum(1.0, 60.0 / D)
    if ov.shrink_w != 0.0:
        if xi_bar is None:
            raise ValueError("shrink_w needs the unconditional curve xi_bar")
        w = ov.shrink_w * np.exp(-D / max(ov.shrink_decay_s, 1e-9))
        logbar = np.log(np.maximum(np.asarray(xi_bar, dtype=np.float64), 1e-300))
        logxi = (1.0 - w) * logxi + w * logbar
    return np.exp(logxi)


def cap_m_Y(ov: "Overrides", m_Y, var_y):
    """Clip the location term at `alpha_cap_sd` settlement standard deviations."""
    if ov.alpha_cap_sd is None:
        return m_Y
    lim = float(ov.alpha_cap_sd) * np.sqrt(np.maximum(var_y, 0.0))
    return np.clip(m_Y, -lim, lim)


def quoted_prob(ov: "Overrides", p):
    """The quoting-layer temperature. Never touches `p_model`."""
    if ov.temperature == 1.0:
        return p
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    lg = np.log(q / (1.0 - q)) / float(ov.temperature)
    out = 1.0 / (1.0 + np.exp(-lg))
    return float(out) if np.ndim(p) == 0 else out
