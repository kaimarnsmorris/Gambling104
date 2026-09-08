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
