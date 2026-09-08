"""link block: the fitted settlement tail.

    P(up) = P(Y > y*) = 1 - F_t(-z; mu, sigma_t) = F_t((z + mu) / sigma_t)

by the symmetry of the Student-t, with (nu, mu, sigma_t) read from the export at
the tick's own business time.

The export's own `p_quoted` column is not usable here: `f` perturbs the level
(`s +/- e_s`), so the quote's `z` differs from the export's own row, and `p_model`
has to be recomputed from `(z, nu, mu, sigma_t)`. But a recomputed `p_model` does
not carry the quoting-layer temperature knob (`fvmodel.overrides.temperature`),
while the spec's quoting layer wants `p_quoted`. `_fvexport.temperature()` reads
the variant's knob back out of the export's own sidecar, and it is applied here
through `fvmodel.overrides.quoted_prob` - not reimplemented - so there is exactly
one definition of the sigmoid/logit transform in the codebase. `quoted_prob` wants
an object with a `.temperature` attribute; a full `Overrides(temperature=...)` is
used rather than a bespoke shim class because `Overrides` already is that object,
every other field defaults, and constructing a second "temperature holder" type
would just be a second definition of the same idea.

NOTE FOR THE HARNESS: the slot contract is a pure `link(z) -> p`, but this tail's
parameters are a function of business time left and are therefore per-tick, the
same way `f` is bound to `strike` and `sigma[i]`. `bind(ep)` returns the per-tick
callable; if the harness cannot bind it, `link(z, i)` takes the index directly.
"""
import numpy as np
from scipy import stats

from _fvexport import for_episode, temperature as _read_temperature
from fvmodel.overrides import Overrides, quoted_prob

_BOUND = {}


def _prob(z, nu, mu, sigma_t):
    z = np.asarray(z, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    sg = np.maximum(np.asarray(sigma_t, dtype=np.float64), 1e-12)
    return stats.t.cdf((z + np.asarray(mu, dtype=np.float64)) / sg, df=nu)


def _quote(z, nu, mu, sigma_t, temp: float = 1.0):
    """`_prob` then the quoting-layer temperature - never touches `p_model`."""
    p = _prob(z, nu, mu, sigma_t)
    return quoted_prob(Overrides(temperature=temp), p)


def bind(ep):
    c = for_episode(ep)
    temp = _read_temperature()

    def link(z, i):
        return float(_quote(z, c["nu"][i], c["mu"][i], c["sigma_t"][i], temp))

    _BOUND[id(ep)] = link
    return link


def link(z, i, ep=None):
    if ep is not None:
        return bind(ep)(z, i)
    if not _BOUND:
        raise RuntimeError("link.bind(ep) must be called once per episode")
    return next(reversed(list(_BOUND.values())))(z, i)
