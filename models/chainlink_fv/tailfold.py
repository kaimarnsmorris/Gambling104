"""Fold a per-tick Student-t tail into the two numbers the harness carries.

The harness hands the quote algebra `s[i]` and `sigma[i]` per tick, plus a
`link(z)` that is a pure function with no tick index and no episode. Our tail is
`(nu, mu, sigma_t)` indexed by business time, so it is a third per-tick degree of
freedom with nowhere of its own to go.

It goes here. Writing `u = level - strike`, the model's price curve is

    p(level) = F_t( (u/sigma + mu) / sigma_t ; nu )

and the harness computes `F_t(u / S ; NU0)`. Matching the two at their quartiles
- which for a symmetric t also matches their medians - gives

    S       = sigma * sigma_t * q75(nu) / q75(NU0)
    s'      = s + mu * sigma

The scale carries the shape mismatch and the shifted fair value carries `mu`,
because `f` measures from the strike and cannot be told to centre anywhere else.

Why quartile-matching rather than copying the model's own scale: copying it
(`S = sigma * sigma_t`) forces the whole nu discrepancy into the shape and
measures 1.8x worse at the 95th percentile. Letting the scale absorb part of the
mismatch is what makes a fixed shape affordable at all.

WHAT THIS COSTS. Exact wherever the fitted `nu` equals `NU0`, and elsewhere a
median 0.07c and 95th-percentile 0.42c of probability in the band where quoting
happens, against a taker fee that peaks at 1.75c and a maker rebate averaging
0.234c. `NU0` is the minimax choice over the nine fitted business-time bins: it
is not the lowest typical error available (nu0 = 4.35 is) but it has the best
worst case, 0.82c against 1.26c, and on a venue where one bad fill costs more
than several good ones that is the trade worth making.

The fold is the one deliberate loss of fidelity in the chain. If the harness ever
gives `link` a per-tick binding, delete this module and hand the tail over whole.
"""
from __future__ import annotations

from scipy import stats

#: Shape of the fixed link. Minimax over the fitted tails; see the module note.
NU0 = 3.483

_Q75_NU0 = float(stats.t.ppf(0.75, df=NU0))
_Q75_NORMAL = float(stats.norm.ppf(0.75))


def harness_columns(*, s, sigma, nu, mu, sigma_t, family="t"):
    """`(s', S)`: the fair value and scale that make a fixed-`NU0` link correct.

    Scalars or arrays; arrays broadcast. `sigma` is the settlement standard
    deviation in USD, `(nu, mu, sigma_t)` the fitted tail at this tick.

    `family` must be what the model itself would use. `SettlementTail.prob_up`
    ignores `nu`, `mu` and `sigma_t` completely when the family is normal -- it is
    `1 - norm.cdf(y*/sd)` and nothing more -- while those three columns stay in the
    export carrying their fitted Student-t values. Reading them anyway would fold a
    deliberately-normal variant as a t, which is the one thing that variant exists
    to rule out.
    """
    if family == "normal":
        # link.py serves a normal for this variant, so the fold is the identity.
        # Quartile-matching a normal onto the t link instead costs 1.64c at the
        # 95th percentile of the quoting band -- four times the t fold, and enough
        # that `normal_tail` would be measuring the fold rather than the model.
        return s, sigma
    if family != "t":
        raise ValueError("unknown tail family %r; expected 't' or 'normal'" % family)
    scale = sigma * sigma_t * (stats.t.ppf(0.75, df=nu) / _Q75_NU0)
    return s + mu * sigma, scale
