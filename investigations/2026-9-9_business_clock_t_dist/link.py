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

NOTE FOR THE HARNESS - THE ONE OPEN CONTRACT ITEM
-------------------------------------------------
The slot contract is a pure `link(z) -> p`, but this tail's parameters are a
function of the business time left and are therefore PER TICK, the same way `f`
is bound to `strike` and `sigma[i]`. As shipped, `harness/core/run.py` puts
`modules["link"].link` straight into the blocks dict and `harness/core/loop.py`
calls it as `link(z_i)`, with no index and no `bind` - which this file cannot
serve, and does not pretend to: `link(z)` with no index raises a `TypeError`
naming the fix rather than guessing a tail. Guessing is not available. Within one
5 m episode `sigma_t` moves by more than an order of magnitude, so any episode-
level aggregate would be wrong on nearly every tick and wrong invisibly.

`at(ep, i)` below IS the per-tick hook, and it is attached to the `link` function
object as well as to the module, because the blocks dict holds the function.
The harness change is ONE added line in `loop.py`'s tick loop, right after
`sigma_i = float(sigma_arr[i])`:

    link_i = getattr(link, "at", lambda _e, _i: link)(ep, i)

and then `link_i` in place of `link` in the two calls two lines below (the
`quotes(...)` argument and `fair_p = link(z_i)`). `link_i` is itself a plain
`link(z) -> p`, so `quote.py` and every other block are untouched, and the
`getattr` default leaves the stock logistic `blocks/defaults/link.py` - which has
no `.at` - behaving exactly as it does today. See
`backtesting_5m/docs/fair-export-handoff.md`.
"""
import numpy as np
from scipy import stats

from _fvexport import (for_episode, tail_family as _read_tail_family,
                       temperature as _read_temperature)
from fvmodel.overrides import Overrides, quoted_prob

#: market_id -> the per-tick `link(z, i)` for the episode currently in flight.
#: One entry: the loop runs one episode at a time and an episode's expanded
#: export columns are already cached in `_fvexport._for_market`.
_BOUND = {}

_NO_INDEX = """\
link(z) was called with no tick index. This variant's settlement tail
(nu, mu, sigma_t) is a function of the business time left, so it is PER TICK -
the same way `f` is bound to `strike` and `sigma[i]`. There is no tail to guess:
within one 5 m episode sigma_t moves by more than a factor of ten, so an
episode-level aggregate would be silently wrong on nearly every tick.

THE HARNESS CHANGE - one added line, in harness/core/loop.py's tick loop, right
after `sigma_i = float(sigma_arr[i])`:

    link_i = getattr(link, "at", lambda _e, _i: link)(ep, i)

then pass `link_i` instead of `link` to `quotes(...)` and call `link_i(z_i)`
instead of `link(z_i)` two lines below. `link_i` is still a plain
`link(z) -> p`, so quote.py and every other block are untouched, and the getattr
default leaves the stock logistic blocks/defaults/link.py (which has no `.at`)
behaving exactly as it does now.

See backtesting_5m/docs/fair-export-handoff.md."""


def _prob(z, nu, mu, sigma_t, family: str = "t"):
    """P(up) at the block's own `z`, in the family the variant actually fits.

    Derivation of the `family == "normal"` branch, from `fvmodel/tails.py`'s
    `SettlementTail.prob_up`: that method computes `q = y_star / sqrt(var_y)` and,
    for the normal family, returns `1 - norm.cdf(q)`, ignoring `(mu, sigma)`
    entirely (`SettlementTail.normal()` ships `mu=0, sigma=1` precisely so those
    columns would be no-ops even if consulted). Relating `q` to this block's `z`:
    the export sets `y_star = (K - s) / (p_ref * omega)` and `sigma = sqrt(var_y) *
    p_ref * omega`, so `q = y_star / sqrt(var_y) = (K - s) / sigma = -z` (with
    `z = (s - K) / sigma`, this block's convention). So:

        P(up) = 1 - norm.cdf(-z) = norm.cdf(z)      (standard normal symmetry)

    with no dependence on `(mu, sigma_t)` at all - matching `prob_up` exactly,
    including its indifference to those two columns under this family. Do not
    reach for a mu/sigma-shifted normal here; that would not match `tails.py`.
    """
    z = np.asarray(z, dtype=np.float64)
    if family == "normal":
        return stats.norm.cdf(z)
    nu = np.asarray(nu, dtype=np.float64)
    sg = np.maximum(np.asarray(sigma_t, dtype=np.float64), 1e-12)
    return stats.t.cdf((z + np.asarray(mu, dtype=np.float64)) / sg, df=nu)


def _quote(z, nu, mu, sigma_t, temp: float = 1.0, family: str = "t"):
    """`_prob` then the quoting-layer temperature - never touches `p_model`."""
    p = _prob(z, nu, mu, sigma_t, family)
    return quoted_prob(Overrides(temperature=temp), p)


def bind(ep):
    """The per-episode callable `link(z, i)`, reading `ep`'s own export rows."""
    c = for_episode(ep)
    temp = _read_temperature()
    family = _read_tail_family()

    def link(z, i):
        return float(_quote(z, c["nu"][i], c["mu"][i], c["sigma_t"][i], temp, family))

    _BOUND.clear()
    _BOUND[ep.market_id] = link
    return link


def at(ep, i):
    """THE PER-TICK HOOK: a plain `link(z) -> p` for index `i` of episode `ep`.

    Binds lazily and caches per market, so the harness needs no separate `bind`
    call in `run.py` - one line in the tick loop is the whole contract extension
    (see this module's header). The returned callable takes `z` and nothing else,
    which is exactly the slot contract every other block already assumes.
    """
    fn = _BOUND.get(getattr(ep, "market_id", None)) or bind(ep)
    return lambda z: fn(z, i)


def link(z, i=None, ep=None):
    """`P(up)` at standardised distance `z`, at tick `i` of the bound episode.

    `i` is optional only so that the missing-index case fails with a message
    that says what to do about it instead of an argument-count `TypeError` from
    the interpreter. It is not optional in substance.
    """
    if ep is not None:
        if i is None:
            raise TypeError(_NO_INDEX)
        return bind(ep)(z, i)
    if i is None:
        raise TypeError(_NO_INDEX)
    if not _BOUND:
        raise RuntimeError(
            "link(z, i) needs an episode bound first: call link.bind(ep) once per "
            "episode, or use link.at(ep, i). " + _NO_INDEX)
    return next(reversed(list(_BOUND.values())))(z, i)


#: `harness/core/run.py` puts the FUNCTION (not the module) into the blocks dict,
#: so the per-tick hook has to be reachable from the function object too.
link.at = at
link.bind = bind
