"""The blocks must reproduce the model's own p_up from s, sigma and the tail."""
from __future__ import annotations

import numpy as np
import pytest


class FakeEpisode:
    """The minimum of the harness's Episode that the blocks read."""

    def __init__(self, market_id, open_ts, strike, n=3000):
        self.market_id = market_id
        self.open_ts = open_ts
        self.strike = strike
        self.t_ms = np.arange(n, dtype=np.int64) * 100


def test_standardise_is_the_signed_moneyness():
    import f

    assert f.standardise(101.0, 100.0, 2.0) == pytest.approx(0.5)
    assert f.standardise(99.0, 100.0, 2.0) == pytest.approx(-0.5)
    assert np.isnan(f.standardise(101.0, 100.0, 0.0))


def test_link_is_increasing_and_centres_on_the_tail_location():
    import link

    p = [link._prob(z, nu=6.0, mu=0.0, sigma_t=1.0) for z in (-2.0, 0.0, 2.0)]
    assert p[0] < p[1] < p[2]
    assert p[1] == pytest.approx(0.5)
    # a non-zero tail location shifts where p crosses a half
    assert link._prob(0.0, nu=6.0, mu=0.3, sigma_t=1.0) > 0.5


def test_quote_applies_temperature_but_prob_does_not():
    """`link._quote` must carry the quoting-layer knob; `link._prob` (p_model) must
    not - it is the recomputed model probability the spec forbids the knob from
    touching. T > 1 pulls a probability away from the extremes toward 0.5; T == 1
    must leave `_prob`'s output unchanged."""
    import link

    z, nu, mu, sigma_t = 1.5, 6.0, 0.2, 1.0
    p_model = link._prob(z, nu, mu, sigma_t)
    assert p_model > 0.5

    at_one = link._quote(z, nu, mu, sigma_t, temp=1.0)
    assert at_one == pytest.approx(p_model)

    warmed = link._quote(z, nu, mu, sigma_t, temp=2.5)
    assert warmed != pytest.approx(p_model)
    # T > 1 pulls toward 0.5
    assert abs(warmed - 0.5) < abs(p_model - 0.5)

    # symmetric on the other side of a half
    p_model_neg = link._prob(-z, nu, mu, sigma_t)
    warmed_neg = link._quote(-z, nu, mu, sigma_t, temp=2.5)
    assert abs(warmed_neg - 0.5) < abs(p_model_neg - 0.5)


@pytest.mark.slow
def test_blocks_reproduce_the_export_probability():
    """f() then link() must give back p_model for the market's own strike.

    NOTE ON THE STRIKE USED HERE - a deviation from the task-8 brief's literal
    test, with the evidence for it below:

    The brief's version of this test joined `backtesting_5m/data/strikes_5m.parquet`
    (Gamma's REAL published strike) onto the export and expected an atol=1e-9 match
    against `p_quoted`. That assumption is false for this dataset, and it is a data
    fact, not a sign bug: `evaluate()`'s own `strike` (`lvl[iO]`, fvmodel's ATM
    linearisation anchor - `win.cl_step[iO]`, its own step-filled reconstruction of
    the Chainlink print series) and Gamma's `priceToBeat` are two independently
    reconstructed numbers for "the price at market open," and they disagree by real
    dollars (median ~$7, up to ~$245 in a 2,000-row sample of baseline.parquet) -
    almost certainly print/feed staleness between fvmodel's own second-grid
    reconstruction and whatever oracle round Gamma read. `s`/`sigma`/`nu`/`mu`/
    `sigma_t` are all exactly strike-free (verified: `tests/test_export.py::
    test_s_and_sigma_reproduce_y_star`, and independently here by re-deriving `s`
    algebraically from `fvmodel.engine.evaluate`'s own formulas - the strike cancels
    completely), so plugging the REAL Gamma strike into `f.standardise` is a
    perfectly legitimate re-pricing at a different strike - it is simply not the
    strike `evaluate` used to produce the `p_model`/`p_quoted` columns already in the
    export, so nothing requires the two to agree, and empirically (median absolute
    probability difference ~0.027, some rows differing by ~1.0) they do not.

    So this test verifies the identity the brief actually cared about - that f()
    then link() are the model, not an approximation - the same way
    test_export.py::test_s_and_sigma_reproduce_y_star does: against a live
    `evaluate()` call, using ITS OWN `strike`, rather than through the persisted
    export joined to the unrelated real-world strikes table.
    """
    import numpy as np

    from fvmodel.base import CL_T0, load_params
    from fvmodel.build import build_model
    from fvmodel.engine import Window, evaluate, market_grid

    import f
    import link

    model = build_model()
    t0 = CL_T0 + 20 * 86400
    win = Window(load_params(), model.fp, t0, t0 + 2 * 86400, warm=0, chainlink=True,
                 log=lambda *a: None)
    T = market_grid(win, 300, burn_days=1)[:200]
    win.prepare(win.i(T) - 120)
    cell = evaluate(win, model, "chainlink_twap60", 300, 120, expiries=T)
    r = cell.rows
    s = r["strike"] - r["y_star"] * r["p_ref"] * r["omega"]
    sigma = np.sqrt(r["var_y"]) * r["p_ref"] * r["omega"]

    z = f.standardise(s, r["strike"], sigma)
    p = link._prob(z, r["nu"], r["mu"], r["sigma_t"])
    assert np.allclose(p, r["p_model"], atol=1e-9), (
        "the block chain disagrees with the model that produced the export")


@pytest.mark.slow
def test_precompute_covers_every_bucket_and_is_causal():
    import polars as pl

    import fair
    import vol
    from harness_paths import FAIR_DIR, STRIKES

    row = pl.read_parquet(STRIKES).row(100, named=True)
    ep = FakeEpisode(row["market_id"], row["open_ts"], row["strike"])
    s = fair.precompute(ep)
    sg = vol.precompute(ep)
    assert s.shape == (3000,) and sg.shape == (3000,)
    assert np.isfinite(s).mean() > 0.9 and np.all(sg[np.isfinite(sg)] > 0)
    # bucket 0 is served by the pre-open row; buckets 1..10 by t_s = 0
    df = pl.read_parquet(FAIR_DIR / "baseline.parquet").filter(
        pl.col("market_id") == ep.market_id).sort("t_s")
    pre = df.filter(pl.col("t_s") == -1)["s"][0]
    first = df.filter(pl.col("t_s") == 0)["s"][0]
    assert s[0] == pytest.approx(pre)
    assert s[1] == pytest.approx(first) and s[10] == pytest.approx(first)
    assert s[11] == pytest.approx(df.filter(pl.col("t_s") == 1)["s"][0])
