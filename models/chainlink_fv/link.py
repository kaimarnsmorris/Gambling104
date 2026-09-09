"""z -> p. The settlement tail, as one fixed shape for the whole run.

A Student-t at `tailfold.NU0`, or a normal when the variant's tail is one.

The shape is fixed per TICK -- the harness holds this as a bare function and calls
it `link(z)` with no tick index -- but it is NOT fixed per run, and the family is a
per-variant constant. So the variant chooses it, once, here. That is what lets the
`normal_tail` variant be exact instead of being quartile-matched onto a t, which
costs 1.64c at the 95th percentile of the quoting band.

What cannot be chosen per tick is the Student-t's `nu`, which the fitted tails vary
with business time. That variation is folded into the scale by `tailfold`; see its
module note for what the fold costs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy import stats

from _chainlink_fv_export import tail_family, temperature
from tailfold import NU0

_FAMILY = tail_family()
_TEMPERATURE = temperature()


def link(z: float) -> float:
    """P(up) given the standardised distance from the strike.

    The temperature is applied here for the same reason the family is chosen here:
    `p -> sigmoid(logit(p)/T)` is neither a location nor a scale change, so the fold
    cannot carry it, but it is constant across the run.
    """
    p = stats.norm.cdf(z) if _FAMILY == "normal" else stats.t.cdf(z, df=NU0)
    if _TEMPERATURE == 1.0:
        return float(p)
    from fvmodel.overrides import Overrides, quoted_prob
    return float(quoted_prob(Overrides(temperature=_TEMPERATURE), p))
