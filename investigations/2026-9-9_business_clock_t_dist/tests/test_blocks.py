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
def test_blocks_reproduce_the_export_probability_at_the_venue_strike():
    """f() then link(), fed the REAL venue strike, must reproduce the export's own
    `p_quoted` - the strike the harness will actually use.

    Restored after fix-round 1 (coordinator's Ruling 15): `export/build_export.py`
    now recomputes `p_model`/`p_quoted` at the venue's own strike (the 60 s Chainlink
    TWAP ending at open, from `strikes_5m.parquet`) via the model's own
    `SettlementTail.prob_up`, not via the `t.cdf((z+mu)/sigma_t)` form this test (and
    `link._prob`) uses - two independent formulas for the same quantity, so this
    comparison is not circular. `test_blocks_reproduce_the_export_probability` above
    is kept alongside this one: that test pins the block chain to the model's own
    math (via a live `evaluate()` call, at whatever strike `evaluate` chooses to
    linearise around); this one pins the whole chain at the strike the harness will
    actually price against.
    """
    import polars as pl

    import f
    import link
    from harness_paths import FAIR_DIR, STRIKES

    df = pl.read_parquet(FAIR_DIR / "baseline.parquet").filter(pl.col("ok"))
    strikes = pl.read_parquet(STRIKES).select(["market_id", "strike"])
    j = df.join(strikes, on="market_id").sample(2000, seed=0)
    z = f.standardise(j["s"].to_numpy(), j["strike"].to_numpy(), j["sigma"].to_numpy())
    # both signs of moneyness must be exercised, or a sign error could cancel
    assert (z > 0).sum() > 100 and (z < 0).sum() > 100, (
        "sample is one-sided; can't rule out a sign error cancelling")
    p = link._prob(z, j["nu"].to_numpy(), j["mu"].to_numpy(), j["sigma_t"].to_numpy())
    assert np.allclose(p, j["p_quoted"].to_numpy(), atol=1e-9), (
        "the block chain disagrees with the export at the venue's own strike")


@pytest.mark.slow
def test_blocks_reproduce_the_export_probability_for_a_normal_family_variant(tmp_path):
    """Fix-round 1 regression: `link._prob` used to call `stats.t.cdf` unconditionally
    on the exported `(nu, mu, sigma_t)`, and `fvmodel/overrides.py::_scaled_tail`
    leaves those three columns numerically UNCHANGED when `tail_family == "normal"`
    (it only flips an in-memory `.family` flag the export can't carry). So the
    harness's own quoting path (`link.py`) would have silently priced `normal_tail`
    identically to baseline - a shipped variant that does nothing, which is exactly
    the failure mode this scorecard exercise exists to catch. A test that only ever
    exercises a Student-t export cannot see this class of bug, which is why this one
    builds a `normal_tail` export directly rather than reusing `test_blocks_reproduce_
    the_export_probability_at_the_venue_strike`'s baseline fixture.
    """
    import polars as pl

    from export.build_export import build
    from harness_paths import STRIKES

    import json

    import f
    import link

    t0 = 1786665600
    p = build("normal_tail", t0, t0 + 3 * 3600, out_path=tmp_path / "normal_tail.parquet")
    df = pl.read_parquet(p).filter(pl.col("ok"))
    strikes = pl.read_parquet(STRIKES).select(["market_id", "strike"])
    j = df.join(strikes, on="market_id")
    assert len(j) > 100

    # the sidecar must actually say "normal" - the mechanism this fix-round added.
    # Read it directly rather than through `_fvexport.tail_family`, which always
    # looks under `harness_paths.FAIR_DIR`, not this test's `tmp_path` build.
    meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["provenance"]["overrides_non_default"]["tail_family"] == "normal"

    z = f.standardise(j["s"].to_numpy(), j["strike"].to_numpy(), j["sigma"].to_numpy())
    assert (z > 0).sum() > 20 and (z < 0).sum() > 20, (
        "sample is one-sided; can't rule out a sign error cancelling")

    # family="t" (the pre-fix default) must NOT reproduce p_quoted: this is the
    # regression check - if this assertion starts failing, the bug is back
    p_wrong = link._prob(z, j["nu"].to_numpy(), j["mu"].to_numpy(),
                         j["sigma_t"].to_numpy(), family="t")
    assert not np.allclose(p_wrong, j["p_quoted"].to_numpy(), atol=1e-6), (
        "the Student-t form should NOT match a normal-family export; if it does, "
        "the bug this test exists to catch has resurfaced")

    p_right = link._prob(z, j["nu"].to_numpy(), j["mu"].to_numpy(),
                         j["sigma_t"].to_numpy(), family="normal")
    assert np.allclose(p_right, j["p_quoted"].to_numpy(), atol=1e-9), (
        "the block chain disagrees with the normal-family export at the venue strike")


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
