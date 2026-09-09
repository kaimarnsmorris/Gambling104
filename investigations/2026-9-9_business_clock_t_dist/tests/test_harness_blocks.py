"""The three block files, against the harness's real contract.

The contract that matters is `link`. The harness holds a bare function reference
(`blocks["link"]`) and calls it `link(z_i)` inside its loop and again, twice, inside
`quote.quotes(...)`. An earlier version of this model shipped a `link(z, i)` that
needed a tick index, which is a TypeError on the first tick -- the block files ran
in tests and could never have run in the harness. So the first test here calls
`link` exactly as `loop.py` does, through the harness's own default `quote` block,
rather than however happens to be convenient.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest
from scipy import stats

MODEL_DIR = os.path.join(
    pathlib.Path(__file__).resolve().parents[3], "models", "chainlink_fv")


def _load(slot):
    from harness.core import provenance
    return provenance.load_slot(f"{MODEL_DIR}/{slot}.py", slot)


class FakeEpisode:
    """Only what `fair`/`vol` read: the market's identity and its length."""

    def __init__(self, market_id, n=3000):
        self.market_id = market_id
        self._n = n
        self.s = np.full(n, np.nan)

    def __len__(self):
        return self._n


# ---------------------------------------------------------------- the link
def test_link_takes_one_argument_and_nothing_else():
    """The whole reason the previous block layer could not run."""
    import inspect

    link = _load("link").link
    params = list(inspect.signature(link).parameters.values())
    assert len(params) == 1, [p.name for p in params]
    assert params[0].kind in (params[0].POSITIONAL_ONLY,
                              params[0].POSITIONAL_OR_KEYWORD)
    assert link(0.0) == pytest.approx(0.5)


def test_link_is_the_fixed_shape_the_fold_was_built_against():
    from tailfold import NU0

    link = _load("link").link
    for z in (-3.0, -0.7, 0.0, 0.7, 3.0):
        assert link(z) == pytest.approx(stats.t.cdf(z, df=NU0), abs=1e-12)


def test_link_is_monotone_and_bounded():
    link = _load("link").link
    zs = np.linspace(-60.0, 60.0, 400)
    ps = np.array([link(float(z)) for z in zs])
    assert np.all(np.diff(ps) >= 0.0)
    assert ps.min() >= 0.0 and ps.max() <= 1.0
    assert np.isfinite(ps).all(), "the harness clips prices, it does not clean NaNs"


def test_the_harness_default_quote_block_can_drive_our_link():
    """Exercises the real call path: quote() calls standardise() then link() twice."""
    from harness.core import provenance
    from harness.core.config import QuoteParams

    f = provenance.load_slot(
        provenance.resolve_slots("nonexistent_investigation")["f"], "f")
    quote = provenance.load_slot(
        provenance.resolve_slots("nonexistent_investigation")["quote"], "quote")
    link = _load("link").link

    params = QuoteParams(e_s=5.0, e_z=0.0, e_p=0.0, rpl_s=0.0, rpl_z=0.0,
                         rpl_p=0.0, max_pos=10.0)
    bid, ask = quote.quotes(100_000.0, 0.0, 100_000.0, 60.0, params,
                            f.standardise, link)
    assert 0.0 <= bid < ask <= 1.0


# ------------------------------------------------------------- fair and vol
@pytest.mark.slow
def test_fair_and_vol_cover_every_bucket_of_a_real_market():
    import polars as pl

    from harness_paths import FAIR_DIR

    mid = pl.read_parquet(FAIR_DIR / "baseline.parquet",
                          columns=["market_id"])["market_id"][0]
    ep = FakeEpisode(mid)
    s = _load("fair").precompute(ep)
    sigma = _load("vol").precompute(ep)

    assert s.shape == (3000,) and sigma.shape == (3000,)
    assert np.isfinite(s).all(), "every bucket must carry a fair value"
    assert np.isfinite(sigma).all() and np.all(sigma > 0)


@pytest.mark.slow
def test_the_bucket_expansion_matches_the_canonical_causality_shift():
    """`bucket_owner` is the single definition of the shift; the model dir carries
    its own copy so it can be shared, and this pins the two together."""
    import polars as pl

    from export.build_export import bucket_owner
    from harness_paths import FAIR_DIR
    import _chainlink_fv_export as fv

    assert np.array_equal(fv.bucket_owner(), bucket_owner())

    mid = pl.read_parquet(FAIR_DIR / "baseline.parquet",
                          columns=["market_id"])["market_id"][0]
    ep = FakeEpisode(mid)
    s = _load("fair").precompute(ep)
    rows = (pl.read_parquet(FAIR_DIR / "baseline.parquet")
            .filter(pl.col("market_id") == mid).sort("t_s"))
    own = bucket_owner()
    # bucket 0 is served by the pre-open row, buckets 1..10 by t_s = 0
    from tailfold import harness_columns
    def s_prime_at(t_s):
        r = rows.filter(pl.col("t_s") == t_s)
        return harness_columns(s=r["s"][0], sigma=r["sigma"][0], nu=r["nu"][0],
                               mu=r["mu"][0], sigma_t=r["sigma_t"][0])[0]
    assert own[0] == -1 and own[1] == 0 and own[10] == 0 and own[11] == 1
    assert s[0] == pytest.approx(s_prime_at(-1))
    assert s[1] == pytest.approx(s_prime_at(0))
    assert s[10] == pytest.approx(s_prime_at(0))
    assert s[11] == pytest.approx(s_prime_at(1))


@pytest.mark.slow
def test_the_whole_chain_reproduces_the_model_within_the_approved_budget():
    """f -> link on the block outputs, against the model's own probability, at the
    venue strike. This is the end-to-end statement of what the fold costs."""
    import polars as pl

    from harness.core import provenance
    from harness_paths import FAIR_DIR, STRIKES
    from export.build_export import bucket_owner

    f = provenance.load_slot(
        provenance.resolve_slots("nonexistent_investigation")["f"], "f")
    link = _load("link").link

    ids = (pl.read_parquet(FAIR_DIR / "baseline.parquet", columns=["market_id"])
           ["market_id"].unique().to_list()[:40])
    strikes = dict(zip(*pl.read_parquet(STRIKES)
                       .select(["market_id", "strike"]).to_dict(as_series=False)
                       .values()))
    export = pl.read_parquet(FAIR_DIR / "baseline.parquet")
    own = bucket_owner()

    errs = []
    for mid in ids:
        K = strikes[mid]
        ep = FakeEpisode(mid)
        s = _load("fair").precompute(ep)
        sigma = _load("vol").precompute(ep)
        rows = export.filter(pl.col("market_id") == mid).sort("t_s")
        t_s = rows["t_s"].to_numpy()
        pos = np.searchsorted(t_s, own)
        r = {c: rows[c].to_numpy()[pos] for c in
             ("s", "sigma", "nu", "mu", "sigma_t")}
        p_true = stats.t.cdf(((r["s"] - K) / r["sigma"] + r["mu"]) / r["sigma_t"],
                             df=r["nu"])
        p_chain = np.array([link(f.standardise(s[i], K, sigma[i]))
                            for i in range(len(own))])
        errs.append(np.abs(p_chain - p_true))
    e = np.concatenate(errs)
    assert np.median(e) < 0.002, np.median(e)
    assert np.percentile(e, 95) < 0.010, np.percentile(e, 95)


def test_link_applies_the_variant_temperature(monkeypatch, tmp_path):
    """Temperature is a quoting-layer reshape, `p -> sigmoid(logit(p)/T)`. It is not
    a location or scale change, so the fold cannot carry it -- but like the tail
    family it is a per-variant constant, so the link can apply it once for the run.

    If a temperature sweep ever shows identical prices at every temperature, this
    is the test that should have failed."""
    import json

    from fvmodel.overrides import Overrides, quoted_prob

    (tmp_path / "hot.json").write_text(json.dumps(
        {"provenance": {"overrides_non_default": {"temperature": 2.5}}}))
    (tmp_path / "hot.parquet").write_bytes(b"")          # never read by link
    monkeypatch.setenv("FV_FAIR_DIR", str(tmp_path))
    monkeypatch.setenv("FV_VARIANT", "hot")

    link = _load("link").link
    for z in (-1.5, -0.4, 0.4, 1.5):
        plain = stats.t.cdf(z, df=__import__("tailfold").NU0)
        assert link(z) == pytest.approx(
            float(quoted_prob(Overrides(temperature=2.5), plain)), abs=1e-12)
    assert link(0.0) == pytest.approx(0.5), "a half must stay a half"


def test_link_is_untouched_at_temperature_one(monkeypatch, tmp_path):
    import json

    from tailfold import NU0

    (tmp_path / "plain.json").write_text(json.dumps({"provenance": {}}))
    monkeypatch.setenv("FV_FAIR_DIR", str(tmp_path))
    monkeypatch.setenv("FV_VARIANT", "plain")
    link = _load("link").link
    for z in (-2.0, 0.0, 2.0):
        assert link(z) == pytest.approx(float(stats.t.cdf(z, df=NU0)), abs=1e-12)


@pytest.mark.slow
def test_two_variants_in_one_process_do_not_share_a_signal(tmp_path, monkeypatch):
    """The harness caches `s`/`sigma` on the CONTENT of fair.py and vol.py plus
    `signal_params`. Every variant of this model shares byte-identical block files
    -- only the export differs -- so selecting the variant through an environment
    variable puts nothing in that key, and the second variant in a process is served
    the first one's arrays. Silently, and reported as its result.

    That is the exact failure the harness's own signals docstring says it exists to
    prevent, so the variant must travel as a signal param."""
    from harness.core.signals import block_signature, precompute_signals
    from harness.core import provenance

    # the 16 variant exports live where the calibration runner wrote them, not in
    # the harness's own fair dir, which holds baseline alone
    monkeypatch.setenv("FV_FAIR_DIR",
                       str(pathlib.Path(__file__).resolve().parents[1] /
                           "runs" / "select"))

    resolved = provenance.resolve_slots("no_such_investigation", model_dir=MODEL_DIR)
    modules = {s: provenance.load_slot(resolved[s], s) for s in ("fair", "vol")}
    sig = block_signature(resolved)

    import polars as pl
    from harness_paths import FAIR_DIR
    mid = pl.read_parquet(FAIR_DIR / "baseline.parquet",
                          columns=["market_id"])["market_id"][0]
    ep = FakeEpisode(mid)

    cache = {}
    a = precompute_signals([ep], modules, sig, cache=cache,
                           signal_params={"fair": {"variant": "baseline"},
                                          "vol": {"variant": "baseline"}})
    b = precompute_signals([ep], modules, sig, cache=cache,
                           signal_params={"fair": {"variant": "rho_off"},
                                          "vol": {"variant": "rho_off"}})
    sa, sb = a[ep.market_id][1], b[ep.market_id][1]      # the sigma arrays
    assert not np.allclose(sa[np.isfinite(sa)], sb[np.isfinite(sb)]), (
        "rho_off was served baseline's sigma -- the variant is not in the cache key")
