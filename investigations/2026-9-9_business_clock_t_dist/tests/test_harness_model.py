"""The tail fold: how a per-tick Student-t reaches a harness that has room for two.

The harness passes exactly two numbers per tick into the quote algebra -- `s[i]`
and `sigma[i]` -- and a `link(z)` that is a pure function with no tick index. Our
tail is `(nu, mu, sigma_t)` varying with business time, which is a third per-tick
degree of freedom. `harness_columns` is where that third one goes: it folds the
tail into a location shift and a scale, against a fixed-shape link, by matching
the true distribution at its quartiles.

The first two tests below are the exactness claim -- when the fitted `nu` happens
to equal the link's own `NU0` the fold costs nothing at all, at every level, not
just at the money. The third is the price of the concession where `nu` differs,
measured on the real fitted tails rather than asserted.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest
from scipy import stats


def _p_true(level, strike, s, sigma, nu, mu, sigma_t):
    """The model's own probability, straight from `SettlementTail.prob_up`'s form."""
    return stats.t.cdf(((level - strike) / sigma + mu) / sigma_t, df=nu)


def _p_chain(level, strike, s_prime, scale, nu0):
    """What the harness computes: f(level) then link(z)."""
    return stats.t.cdf((level - strike) / scale, df=nu0)


def test_the_fold_is_exact_at_every_level_when_nu_matches_the_link():
    """With nu == NU0 the fixed shape IS the true shape, so the fold must be lossless
    across the whole curve -- including the perturbed levels the quote algebra visits,
    which is why this checks a range and not just the fair point."""
    from tailfold import NU0, harness_columns

    strike, s, sigma, mu, sigma_t = 100_000.0, 100_040.0, 65.0, -0.0124, 0.4645
    s_prime, scale = harness_columns(s=s, sigma=sigma, nu=NU0, mu=mu, sigma_t=sigma_t)

    for level in (s - 300.0, s - 50.0, s, s + 50.0, s + 300.0):
        want = _p_true(level, strike, s, sigma, NU0, mu, sigma_t)
        got = _p_chain(level + (s_prime - s), strike, s_prime, scale, NU0)
        assert got == pytest.approx(want, abs=1e-12), (
            "level %.1f: chain %.15f vs model %.15f" % (level, got, want))


def test_the_fold_puts_the_median_where_the_model_puts_it():
    """mu shifts the model's 50/50 level off the strike. The harness centres `link`
    on the strike and cannot be told otherwise, so the shift has to ride on `s`."""
    from tailfold import NU0, harness_columns

    strike, s, sigma, mu, sigma_t = 100_000.0, 100_000.0, 65.0, -0.02, 0.5
    s_prime, scale = harness_columns(s=s, sigma=sigma, nu=NU0, mu=mu, sigma_t=sigma_t)

    # the model is 50/50 at the level where (level-strike)/sigma + mu == 0
    true_median_level = strike - mu * sigma
    # the chain is 50/50 where its own argument is zero, i.e. level == strike,
    # reached from a fair value shifted by exactly (s_prime - s).
    # abs, not rel: recovering a $1.30 shift by subtracting two $100,000 numbers
    # cancels away all but ~1e5 * 2.2e-16 ~= 2e-11 of it, so a relative tolerance
    # on 1.3 is below the noise floor of the arithmetic itself.
    assert s_prime - s == pytest.approx(mu * sigma, abs=1e-9)
    assert _p_chain(true_median_level + (s_prime - s), strike, s_prime, scale,
                    NU0) == pytest.approx(0.5, abs=1e-12)


def test_a_fatter_fitted_tail_widens_the_scale():
    """nu below NU0 means a fatter true tail than the link can express, so the
    quartile match must widen the scale to compensate. If this ever inverts, the
    fold is fighting the shape instead of matching it."""
    from tailfold import NU0, harness_columns

    common = dict(s=100_000.0, sigma=65.0, mu=0.0, sigma_t=0.5)
    _, wide = harness_columns(nu=2.5, **common)
    _, base = harness_columns(nu=NU0, **common)
    _, thin = harness_columns(nu=8.0, **common)
    # q75 falls as nu rises (0.7850 at 2.5, 0.7064 at 8.0), and the quartile match
    # sets S proportional to it, so the fatter tail gets the wider scale
    assert thin < base < wide, (thin, base, wide)


def test_scale_is_the_plain_settlement_sd_when_the_tail_is_the_link():
    """The degenerate case worth pinning: a unit-scale tail at the link's own nu
    must hand the harness the settlement sd untouched."""
    from tailfold import NU0, harness_columns

    s_prime, scale = harness_columns(s=100_000.0, sigma=65.0, nu=NU0, mu=0.0,
                                     sigma_t=1.0)
    assert scale == pytest.approx(65.0, rel=1e-12)
    assert s_prime == pytest.approx(100_000.0, rel=1e-12)


def test_harness_columns_is_vectorised():
    """The export builder applies this to a million rows at a time."""
    from tailfold import NU0, harness_columns

    n = 5
    s_prime, scale = harness_columns(
        s=np.full(n, 100_000.0), sigma=np.full(n, 65.0),
        nu=np.linspace(2.5, 6.0, n), mu=np.full(n, -0.01),
        sigma_t=np.full(n, 0.5))
    assert s_prime.shape == (n,) and scale.shape == (n,)
    assert np.all(np.diff(scale) < 0), "scale must fall as nu rises"


@pytest.mark.slow
def test_the_concession_stays_inside_the_budget_it_was_approved_at():
    """The fold is a deliberate loss of fidelity, approved at a measured size. This
    is the guardrail on it: median 0.07c and p95 0.42c in the band where quoting
    happens. If a retune of the tails pushes it past half a cent, that is a decision
    to re-take, not a number to widen."""
    import polars as pl

    from harness_paths import FAIR_DIR, STRIKES
    from tailfold import NU0, harness_columns

    df = pl.read_parquet(FAIR_DIR / "baseline.parquet").filter(pl.col("ok"))
    k = pl.read_parquet(STRIKES).select(["market_id", "strike"])
    d = df.join(k, on="market_id").sample(100_000, seed=0)
    s, sigma = d["s"].to_numpy(), d["sigma"].to_numpy()
    strike = d["strike"].to_numpy()
    nu, mu, sigma_t = (d[c].to_numpy() for c in ("nu", "mu", "sigma_t"))

    p_true = _p_true(s, strike, s, sigma, nu, mu, sigma_t)
    s_prime, scale = harness_columns(s=s, sigma=sigma, nu=nu, mu=mu, sigma_t=sigma_t)
    p_chain = _p_chain(s_prime, strike, s_prime, scale, NU0)

    err = np.abs(p_chain - p_true)
    band = np.abs(p_true - 0.5) < 0.40
    assert np.median(err[band]) < 0.0015, np.median(err[band])
    assert np.percentile(err[band], 95) < 0.005, np.percentile(err[band], 95)
    assert err[band].max() < 0.010, err[band].max()


def test_a_normal_family_tail_folds_to_an_untouched_settlement_sd():
    """`SettlementTail.prob_up` ignores nu/mu/sigma_t completely when the family is
    normal -- it is `1 - norm.cdf(y*/sd)` and nothing more -- while those three
    columns stay in the export carrying their fitted Student-t values. A fold that
    read them anyway would build a t-shaped curve for a deliberately-normal variant.

    Quartile-matching a normal onto a t(NU0) link was the first thing tried and it
    costs 1.64c at the 95th percentile in the quoting band -- four times the t fold
    and comparable to the whole taker fee, which would make `normal_tail` measure
    the fold rather than the model. So the link takes the family instead: it is
    fixed per TICK, not per run, and the family is a per-variant constant. With a
    normal link the fold is the identity and the variant is exact.
    """
    from tailfold import harness_columns

    s, sigma = 100_040.0, 65.0
    s_prime, scale = harness_columns(s=s, sigma=sigma, nu=2.5, mu=-0.02,
                                     sigma_t=0.46, family="normal")
    assert s_prime == pytest.approx(s, rel=1e-15)
    assert scale == pytest.approx(sigma, rel=1e-15)


def test_the_normal_fold_is_exact_against_a_normal_link():
    """The claim the previous test rests on, stated at the level of probabilities."""
    from tailfold import harness_columns

    strike, s, sigma = 100_000.0, 100_040.0, 65.0
    s_prime, scale = harness_columns(s=s, sigma=sigma, nu=2.5, mu=-0.02,
                                     sigma_t=0.46, family="normal")
    for level in (s - 300.0, s, s + 300.0):
        got = stats.norm.cdf((level - strike) / scale)
        want = stats.norm.cdf((level - strike) / sigma)
        assert got == pytest.approx(want, abs=1e-15), level
