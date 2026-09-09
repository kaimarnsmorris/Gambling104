"""The take-only execution arm.

WHY IT EXISTS. The harness's default policy is deliberately unified -- "we rest on
both sides AND lift the book whenever it is through our threshold" -- and it has no
mode, by design. That is the right default. It is not the right policy for THIS
model, and the measurement says so plainly: over 400 markets the maker arm earns a
markout of -3.70 c/share while crossing on a large disagreement with the book earns
+0.55 c/share at |p - mid| >= 0.05 and +7.84 c at >= 0.12, both already net of the
taker fee. The passive fills arrive precisely when the book is about to run through
us; the crossing trades are the ones we are right about.

So this block keeps the default's crossing decision and drops its resting one. It
does that by WRAPPING the default rather than reimplementing it: every gate the
default enforces -- book staleness, the tte window, the requote cadence, the
per-side position cap, the in-flight cap, the one-live-order rule, the
non-cancellable in-flight cross -- is logic we want unchanged and would otherwise
have to copy correctly. Copying it correctly is not the interesting part of this
investigation, and a copy is a thing that drifts.

Cancels pass through untouched. An order already resting must still be pulled when
the book goes stale or the market ends, and suppressing a cancel because we no
longer intend to rest would strand it.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

MODEL_DIR = os.path.join(
    pathlib.Path(__file__).resolve().parents[3], "models", "chainlink_fv")
INV = str(pathlib.Path(__file__).resolve().parents[1])


class FakeEp:
    """Exactly the surface `decide` reads: a book, its age, and time to expiry."""

    def __init__(self, bid, ask, n=3000):
        self.bid = np.full(n, bid)
        self.ask = np.full(n, ask)
        self.has_book = np.ones(n, dtype=bool)
        self.book_age_ms = np.zeros(n)
        self._n = n

    def __len__(self):
        return self._n

    def tte_s(self, i):
        return 300.0 - i * 0.1


def _blocks():
    from harness.core import provenance
    resolved = provenance.resolve_slots(INV, model_dir=MODEL_DIR)
    default = provenance.load_slot(
        provenance.resolve_slots("no_such_investigation")["execution"], "execution")
    ours = provenance.load_slot(resolved["execution"], "execution")
    return default, ours


def _args(eff_bid, eff_ask, bid, ask, q=0.0, i=0):
    from harness.core.config import ExecConfig, QuoteParams
    return dict(i=i, eff_bid=eff_bid, eff_ask=eff_ask, q=q,
                ep=FakeEp(bid, ask), live_orders=[], execn=ExecConfig(),
                params=QuoteParams(e_p=0.0, max_pos=100.0, shares=10.0))


def test_our_execution_block_shadows_the_default():
    from harness.core import provenance
    resolved = provenance.resolve_slots(INV, model_dir=MODEL_DIR)
    assert os.path.dirname(resolved["execution"]) == INV, resolved["execution"]


def test_it_never_rests():
    """The whole point. A book far from our fair is where the default would post."""
    from harness.blocks.defaults.fees import Liquidity

    _, ours = _blocks()
    place, _ = ours.decide(**_args(eff_bid=0.50, eff_ask=0.52, bid=0.20, ask=0.80))
    assert all(o.liquidity == Liquidity.TAKER for o in place), [
        (o.side, o.liquidity, o.reason) for o in place]


def test_it_crosses_exactly_when_the_default_would():
    """A book through our threshold on the buy side: the taker order must survive
    the wrapper unchanged, price, size and reason."""
    from harness.blocks.defaults.fees import Liquidity

    default, ours = _blocks()
    a = _args(eff_bid=0.60, eff_ask=0.62, bid=0.30, ask=0.31)
    d_place, _ = default.decide(**a)
    o_place, _ = ours.decide(**_args(eff_bid=0.60, eff_ask=0.62, bid=0.30, ask=0.31))

    d_take = [o for o in d_place if o.liquidity == Liquidity.TAKER]
    assert d_take, "the default did not cross, so this test proves nothing"
    assert len(o_place) == len(d_take)
    for x, y in zip(o_place, d_take):
        assert (x.side, x.price, x.shares, x.reason) == (
            y.side, y.price, y.shares, y.reason)


def test_the_default_would_have_rested_here():
    """Guards the test above: if the default stopped resting, `test_it_never_rests`
    would pass vacuously and we would not notice."""
    from harness.blocks.defaults.fees import Liquidity

    default, _ = _blocks()
    place, _ = default.decide(**_args(eff_bid=0.50, eff_ask=0.52, bid=0.20, ask=0.80))
    assert any(o.liquidity == Liquidity.MAKER for o in place)


def test_cancels_pass_through_untouched():
    """A stale book must still pull resting orders, whatever we intend to place."""
    default, ours = _blocks()
    a = _args(eff_bid=0.50, eff_ask=0.52, bid=0.49, ask=0.51)
    a["ep"].book_age_ms[:] = 99_999.0            # far past max_book_age_ms
    d_place, d_cancel = default.decide(**a)
    a2 = _args(eff_bid=0.50, eff_ask=0.52, bid=0.49, ask=0.51)
    a2["ep"].book_age_ms[:] = 99_999.0
    o_place, o_cancel = ours.decide(**a2)
    assert o_cancel == d_cancel
    assert o_place == []


# ------------------------------------------------------ the vol scale knob
def test_vol_scale_multiplies_the_scale_and_defaults_to_one():
    """The manual adjustment this investigation exists to sweep. Scaling the
    settlement scale up pulls `p` toward a half, which is what undoes the
    overconfidence measured in the tails -- and it is a signal param, so the
    harness can sweep it without rebuilding a single export."""
    from harness.core import provenance

    vol = provenance.load_slot(os.path.join(MODEL_DIR, "vol.py"), "vol")
    import polars as pl

    from harness_paths import FAIR_DIR
    mid = pl.read_parquet(FAIR_DIR / "baseline.parquet",
                          columns=["market_id"])["market_id"][0]

    class Ep:
        market_id = mid

        def __len__(self):
            return 3000

    base = vol.precompute(Ep())
    twice = vol.precompute(Ep(), scale=2.0)
    ok = np.isfinite(base)
    assert np.allclose(twice[ok], 2.0 * base[ok], rtol=1e-12)
    assert np.allclose(vol.precompute(Ep(), scale=1.0)[ok], base[ok], rtol=1e-12)
