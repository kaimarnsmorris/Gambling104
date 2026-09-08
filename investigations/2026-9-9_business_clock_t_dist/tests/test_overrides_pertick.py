"""The per-tick transforms, in isolation and then through the whole pipeline."""
from __future__ import annotations

import numpy as np
import pytest


def _ov(**kw):
    from fvmodel.overrides import Overrides
    return Overrides(**kw)


# ------------------------------------------------------------------ xi_adjust
def test_xi_adjust_is_identity_at_defaults():
    from fvmodel.overrides import xi_adjust

    xi = np.array([[1e-9, 2e-9, 3e-9]])
    d = np.array([[10.0, 20.0, 30.0]])
    assert xi_adjust(_ov(), xi, d) is xi


def test_kappa_vol_scales_every_second_equally():
    from fvmodel.overrides import xi_adjust

    xi = np.array([[1e-9, 2e-9, 3e-9]])
    d = np.array([[1.0, 30.0, 3000.0]])
    out = xi_adjust(_ov(kappa_vol=0.5), xi, d)
    assert np.allclose(out, xi * np.exp(1.0), rtol=1e-14)


def test_kappa_vol_short_only_bites_below_a_minute():
    from fvmodel.overrides import xi_adjust

    xi = np.ones((1, 3)) * 1e-9
    d = np.array([[15.0, 60.0, 600.0]])
    out = xi_adjust(_ov(kappa_vol_short=-0.22), xi, d)
    # weight = min(1, 60/D): 1.0 at 15 s, 1.0 at 60 s, 0.1 at 600 s
    want = xi * np.exp(2.0 * -0.22 * np.array([[1.0, 1.0, 0.1]]))
    assert np.allclose(out, want, rtol=1e-14)


def test_shrink_pulls_toward_the_unconditional_level_at_short_age():
    from fvmodel.overrides import xi_adjust

    xi = np.array([[4e-9, 4e-9]])
    bar = np.array([[1e-9, 1e-9]])
    d = np.array([[0.0, 1e6]])          # w = 0.5 at age 0, ~0 at age 1e6
    out = xi_adjust(_ov(shrink_w=0.5, shrink_decay_s=60.0), xi, d, bar)
    assert out[0, 0] == pytest.approx(np.exp(0.5 * np.log(4e-9) + 0.5 * np.log(1e-9)))
    assert out[0, 1] == pytest.approx(4e-9, rel=1e-12)


def test_shrink_is_applied_after_kappa_vol():
    """kappa_vol moves xi; the shrink then pulls the MOVED xi toward xi_bar."""
    from fvmodel.overrides import xi_adjust

    xi = np.array([[4e-9]])
    bar = np.array([[1e-9]])
    d = np.array([[0.0]])
    out = xi_adjust(_ov(kappa_vol=0.5, shrink_w=0.5, shrink_decay_s=60.0), xi, d, bar)
    moved = 4e-9 * np.exp(1.0)
    assert out[0, 0] == pytest.approx(np.exp(0.5 * np.log(moved) + 0.5 * np.log(1e-9)))


# ------------------------------------------- unconditional_xi, the shrink target
@pytest.fixture(scope="module")
def uncond():
    """`(model, dT over the block, dT at the second before it)` for a 300 s market.

    A 300 s `chainlink_twap60` market has `H = n - m_blk = 120 > 0`, so its block
    does NOT start at the quote origin - which is the only case in which the
    alignment below can be wrong, and exactly the case the shipped export spends
    121 of its 301 rows per market in. Needs no data files: the unconditional
    register bank and the seasonal clock are both in `model.json`.
    """
    from fvmodel.base import CL_T0
    from fvmodel.build import build_model
    from fvmodel.curve import dT_at
    from fvmodel.variance import PAD, block_length

    class _S:
        act = 1.0

    model = build_model()
    n = 300
    m_blk = block_length("chainlink_twap60", n, 60, PAD)
    H = n - m_blk
    assert H > 0, "this fixture is pointless unless the block has a head in front of it"
    dT = dT_at(model.v2, _S(), CL_T0 + 20 * 86400,
               np.arange(H, n + 1, dtype=np.int64))
    return model, dT[1:], float(dT[0])


def test_unconditional_xi_first_element_is_an_increment_not_the_head_integral(uncond):
    """`xi_bar[0]` is `g(H+1) - g(H)`, the block's FIRST increment.

    It is the shrink target for `xi[0]`, and `forward_block`'s own `xi[0]` is
    exactly `g(H+1) - g(H)` (its `blk = diff(g[k0:])`, whose `g[k0]` sits at offset
    H). Differencing against a prepended zero instead puts the whole head integral
    into `xi_bar[0]`; measured here that is ~161x the increment it displaced.
    """
    from fvmodel.curve import _iv_from_cum, unconditional_xi

    model, dT, dT_prev = uncond
    v0 = np.log(np.maximum(np.asarray(
        model.v2.params["registers"]["initial_value"], dtype=np.float64), 1e-300))
    g = _iv_from_cum(model.v2.forward, np.array([dT_prev, dT[0]]), v0)
    increment, head = float(g[1] - g[0]), float(g[1])

    xi_bar = unconditional_xi(model.v2, dT, dT_prev)
    assert xi_bar[0] == pytest.approx(increment, rel=1e-14)
    assert head > 50 * increment, (
        "the head integral must be far larger than the increment it is not; "
        "head %.6g vs increment %.6g" % (head, increment))


def test_unconditional_xi_is_aligned_with_forward_block(uncond):
    """The docstring's promise, checked against `forward_block` itself.

    Hand `forward_block` the unconditional register bank as the state and its block
    IS the unconditional curve, second for second. Nothing here is approximate: the
    two go through the same `_iv_from_cum` on the same `dT`, so the assertion is
    bit-equality and any re-introduced off-by-one breaks it.
    """
    from fvmodel.base import CL_T0
    from fvmodel.curve import forward_block, unconditional_xi
    from fvmodel.variance import PAD, block_length

    model, dT, dT_prev = uncond
    v0 = np.asarray(model.v2.params["registers"]["initial_value"], dtype=np.float64)

    class _S:
        act = 1.0
        v = v0

    n = 300
    _, blk = forward_block(model.v2, _S(), CL_T0 + 20 * 86400, n,
                           block_length("chainlink_twap60", n, 60, PAD))
    assert np.array_equal(blk, unconditional_xi(model.v2, dT, dT_prev))


def test_unconditional_xi_has_no_step_at_its_first_element(uncond):
    """No 87x discontinuity at index 0 can come back.

    The unconditional curve is bumpy in its own right - the corrected integral is
    interpolated on a log-spaced grid - so this is a magnitude guard, not a
    monotonicity one: measured, the first ten true increments span a factor 2.2,
    while the head-integral bug spans a factor ~117.
    """
    from fvmodel.curve import unconditional_xi

    model, dT, dT_prev = uncond
    x = unconditional_xi(model.v2, dT, dT_prev)[:10]
    assert x.max() / x.min() < 10.0, "index 0 is a different order of magnitude"
    assert x[0] < 5.0 * np.median(x[1:])


def test_unconditional_xi_batch_row_matches_the_scalar_call(uncond):
    """The 2-D path (the shrink under `engine.evaluate`) and the 1-D path (under
    `fairvalue.fair_value`) must give the same numbers, or the engine-equality test
    cannot see a shrink bug at all."""
    from fvmodel.curve import unconditional_xi

    model, dT, dT_prev = uncond
    one = unconditional_xi(model.v2, dT, dT_prev)
    many = unconditional_xi(model.v2, np.vstack([dT, dT]),
                            np.array([dT_prev, dT_prev]))
    assert many.shape == (2, dT.size)
    assert np.array_equal(many[0], one) and np.array_equal(many[1], one)


def test_unconditional_xi_with_no_earlier_second_is_the_whole_integral(uncond):
    """`dT_prev=None` is the H == 0 contract: the block starts at the quote origin,
    so its first increment really is the integral from zero."""
    from fvmodel.curve import _iv_from_cum, unconditional_xi

    model, dT, _ = uncond
    v0 = np.log(np.maximum(np.asarray(
        model.v2.params["registers"]["initial_value"], dtype=np.float64), 1e-300))
    head = float(_iv_from_cum(model.v2.forward, np.array([dT[0]]), v0)[0])
    assert unconditional_xi(model.v2, dT, None)[0] == pytest.approx(head, rel=1e-14)


# ------------------------------- the kappa_vol relocation (spec section 10, test 2)
def test_kappa_vol_on_the_forward_curve_differs_from_the_register_bank():
    """The one deliberate behaviour change in this branch, with its size recorded.

    v2.1 applied `kappa_vol` by shifting `log v` on the REGISTER BANK; this branch
    applies `2*kappa_vol` to `log xi` on the FORWARD CURVE (spec 4.2/4.3), which is
    what makes the bank variant-invariant and so cacheable across every variant run.
    The two are not the same map: the forward model's cumulative curve is
    `exp(theta0(z) + theta1(z) * (log v - centre))`, so a shift in `log v` scales
    each second by `exp(2*kappa*theta1(z))` with `theta1` varying along the curve,
    not by the single factor `exp(2*kappa)`.

    MEASURED, at the unconditional register bank, `chainlink_twap60`, n = 300
    (m_blk = 180, H = 120), kappa_vol = +0.15:

        per-second forward variance   median 1.66% apart, max 4.02%
        settlement return variance    6.17% apart (old 3.614e-07, new 3.404e-07)

    and 5.64% apart on the settlement variance at kappa_vol = -0.15. At
    kappa_vol = 0 the two are bit-identical, which is what lets `test_golden.py`
    hold with `==`. The same numbers are recorded in CHANGELOG.md.
    """
    from fvmodel.base import CL_T0
    from fvmodel.build import build_model
    from fvmodel.curve import forward_block
    from fvmodel.variance import PAD, block_length, settlement_variance
    from fvmodel.weights import settlement_weights

    model = build_model()
    v0 = np.asarray(model.v2.params["registers"]["initial_value"], dtype=np.float64)
    n, t = 300, CL_T0 + 20 * 86400
    m_blk = block_length("chainlink_twap60", n, 60, PAD)
    W = settlement_weights("chainlink_twap60", n, model.fp.to_dict(), L=60).W_raw
    rho = model.rho_for(1.0)

    def _var(h, b):
        return float(settlement_variance(np.array([h]), b[None, :], W, rho, n)[0])

    def both(kappa):
        class _New:                        # this branch: 2*kappa on log xi
            act = 1.0
            v = v0

        class _Old:                        # v2.1: log v + 2*kappa on the bank
            act = 1.0
            v = v0 * np.exp(2.0 * kappa)

        h_new, blk_new = forward_block(model.v2, _New(), t, n, m_blk)
        h_old, blk_old = forward_block(model.v2, _Old(), t, n, m_blk)
        h_new, blk_new = h_new * np.exp(2.0 * kappa), blk_new * np.exp(2.0 * kappa)
        return blk_old, blk_new, _var(h_old, blk_old), _var(h_new, blk_new)

    blk_old, blk_new, v_old, v_new = both(0.0)
    assert np.array_equal(blk_old, blk_new), "kappa_vol = 0 must be the identity"
    assert v_old == v_new

    blk_old, blk_new, v_old, v_new = both(0.15)
    rel_curve = np.abs(blk_old - blk_new) / np.maximum(np.abs(blk_new), 1e-300)
    rel_var = abs(v_old - v_new) / v_new
    assert v_old != v_new, "the relocation must actually change the answer"
    assert rel_var == pytest.approx(0.0617, abs=5e-4), (
        "recorded relocation difference at kappa_vol=+0.15 has moved: old %.6e vs "
        "new %.6e, rel %.4f" % (v_old, v_new, rel_var))
    assert float(np.median(rel_curve)) == pytest.approx(0.0166, abs=5e-4)
    assert float(rel_curve.max()) == pytest.approx(0.0402, abs=5e-4)

    _, _, v_old, v_new = both(-0.15)
    assert abs(v_old - v_new) / v_new == pytest.approx(0.0564, abs=5e-4)


# ---------------------------------------------------------------- cap_m_Y
def test_alpha_cap_clips_at_a_multiple_of_the_settlement_sd():
    from fvmodel.overrides import cap_m_Y

    var = np.array([1e-8])                       # sd = 1e-4
    assert cap_m_Y(_ov(), np.array([5e-4]), var)[0] == pytest.approx(5e-4)
    assert cap_m_Y(_ov(alpha_cap_sd=2.0), np.array([5e-4]), var)[0] == pytest.approx(2e-4)
    assert cap_m_Y(_ov(alpha_cap_sd=2.0), np.array([-5e-4]), var)[0] == pytest.approx(-2e-4)
    assert cap_m_Y(_ov(alpha_cap_sd=2.0), np.array([1e-5]), var)[0] == pytest.approx(1e-5)


# -------------------------------------------------------------- quoted_prob
def test_temperature_is_a_logit_rescale_and_fixes_a_half():
    from fvmodel.overrides import quoted_prob

    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(quoted_prob(_ov(), p), p)
    q = quoted_prob(_ov(temperature=2.0), p)
    assert q[1] == pytest.approx(0.5)
    assert q[0] > p[0] and q[2] < p[2], "T > 1 must pull toward a half"
    lg = np.log(p / (1 - p)) / 2.0
    assert np.allclose(q, 1.0 / (1.0 + np.exp(-lg)))


# ------------------------------------------------- through the whole pipeline
@pytest.mark.slow
@pytest.fixture(scope="module")
def priced():
    """One quote, priced under a series of overrides. perp_single: no Jensen term
    and no eps, which is what the orthogonality assertion needs (spec R3)."""
    import numpy as np

    from fvmodel.base import CL_T0, load_params
    from fvmodel.build import build_model
    from fvmodel.engine import Window, market_at, state_at
    from fvmodel.fairvalue import fair_value
    from fvmodel.overrides import apply_overrides

    t0 = CL_T0 + 20 * 86400
    base = build_model()
    win = Window(load_params(), base.fp, t0, t0 + 2 * 86400, warm=0, chainlink=True,
                 log=lambda *a: None)
    it = win.n - 20_000
    win.prepare(np.array([it, it - 2]))
    st = state_at(win, it, base)
    t = int(win.ts[it])

    def price(kind, n, **kw):
        m = apply_overrides(base, _ov(**kw))
        mk = market_at(win, kind, t + n, n, 900, 60)
        return fair_value(st, mk, m, t_now=t)

    return price


@pytest.mark.slow
def test_kappa_vol_and_tail_scale_are_the_same_knob(priced):
    """Spec R2/R3: kappa_vol = c and tail_scale = exp(c) must agree on p_model."""
    c = 0.2
    a = priced("perp_single", 300, kappa_vol=c)
    b = priced("perp_single", 300, tail_scale=float(np.exp(c)))
    assert a.p_model == pytest.approx(b.p_model, rel=1e-10), (
        "kappa_vol=%g gives %.12g, tail_scale=exp(%g) gives %.12g"
        % (c, a.p_model, c, b.p_model))


@pytest.mark.slow
def test_temperature_leaves_p_model_alone(priced):
    a = priced("chainlink_twap60", 120)
    b = priced("chainlink_twap60", 120, temperature=1.5)
    assert b.p_model == a.p_model
    lg = np.log(a.p_model / (1 - a.p_model)) / 1.5
    assert b.p_quoted == pytest.approx(1.0 / (1.0 + np.exp(-lg)))
    assert a.p_quoted == a.p_model


@pytest.mark.slow
def test_kappa_vol_raises_the_variance(priced):
    a = priced("chainlink_twap60", 600)
    b = priced("chainlink_twap60", 600, kappa_vol=0.25)
    assert b.var_y > a.var_y * 1.2
    assert b.omega == a.omega and b.known_value == pytest.approx(a.known_value)


@pytest.mark.slow
def test_eps_scale_zero_leaves_the_basis_channel_alone(priced):
    """`eps_scale = 0` switches off the fast eps residual and NOTHING else.

    `var_basis` is the variance of the slow basis's own drift across the settlement
    window. It is a different process from the fast Chainlink residual, it has its
    own selector (`basis_tracker`), and it must survive `eps_scale = 0`. Both
    pricing paths used to gate the whole residual block - `var_basis` included - on
    `model.eps.sigma_bp != 0.0`, so `eps_x0` was silently switching off two
    channels while claiming one: measured at n = 60, baseline `var_basis` =
    4.011e-11 and `eps_x0`'s was exactly 0.0, about 12% of that variant's whole
    var_y reduction.
    """
    a = priced("chainlink_twap60", 60)
    b = priced("chainlink_twap60", 60, eps_scale=0.0)
    assert a.var_basis > 0.0, "baseline must have a basis-drift variance to lose"
    assert b.var_basis == a.var_basis, (
        "eps_scale must not touch var_basis: baseline %.6g, eps_scale=0 %.6g"
        % (a.var_basis, b.var_basis))
    assert b.var_eps == 0.0 and b.eps_bar == 0.0, "the eps channel must be off"
    assert a.var_eps > 0.0, "baseline must have an eps variance to switch off"
    assert b.var_y > b.var_basis * 0.5, "var_basis must still be inside var_y"

    off = priced("chainlink_twap60", 60, basis_tracker="off")
    assert off.var_basis == 0.0, "basis_tracker is the selector that DOES kill it"
