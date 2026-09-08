"""What apply_overrides folds into the model object at build time."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def base():
    from fvmodel.build import build_model
    return build_model()


def _ov(**kw):
    from fvmodel.overrides import Overrides
    return Overrides(**kw)


def test_defaults_return_an_equivalent_model(base):
    from fvmodel.overrides import apply_overrides

    m = apply_overrides(base, _ov())
    assert m.fp.tau_s == base.fp.tau_s and m.fp.w_spot == base.fp.w_spot
    assert m.eps.sigma_bp == base.eps.sigma_bp
    for kind, t in base.tails.items():
        assert np.array_equal(m.tails[kind].nu, t.nu)
        assert np.array_equal(m.tails[kind].sigma, t.sigma)
        assert np.array_equal(m.tails[kind].mu, t.mu)


def test_unknown_key_is_rejected():
    from fvmodel.overrides import Overrides
    with pytest.raises(ValueError, match="unknown override"):
        Overrides.from_dict({"kappa_volume": 1.0})


def test_bad_enum_is_rejected():
    from fvmodel.overrides import Overrides
    with pytest.raises(ValueError, match="rho_kernel"):
        Overrides.from_dict({"rho_kernel": "sideways"})


def test_label_lists_only_non_defaults():
    assert _ov().label() == "baseline"
    assert _ov(kappa_vol=0.095, rho_kernel="off").label() == "kappa_vol=0.095+rho_kernel=off"


def test_kappa_tail_fattens_nu_and_holds_the_variance(base):
    """nu' = 2 + (nu-2)exp(k) with sigma rescaled so nu/(nu-2) * sigma^2 is unchanged."""
    from fvmodel.overrides import apply_overrides

    m = apply_overrides(base, _ov(kappa_tail=-0.5))
    t0 = base.tails["chainlink_twap60"]
    t1 = m.tails["chainlink_twap60"]
    # A handful of the shortest-horizon bins are already pinned at NU_FLOOR in the
    # shipped fit, so fattening them further still floors at the same value: no
    # bin may move the *other* way, and at least one (an unpinned bin) must move.
    assert np.all(t1.nu <= t0.nu), "negative kappa_tail must never thin the tail"
    assert np.any(t1.nu < t0.nu), "negative kappa_tail must fatten the tail somewhere"
    v0 = t0.nu / (t0.nu - 2.0) * t0.sigma ** 2
    v1 = t1.nu / (t1.nu - 2.0) * t1.sigma ** 2
    assert np.allclose(v0, v1, rtol=1e-12), "standardised variance must not move"


def test_tail_scale_scales_mu_with_sigma(base):
    """Spec ruling R2: scaling sigma alone breaks the kappa_vol orthogonality."""
    from fvmodel.overrides import apply_overrides

    c = 0.3
    m = apply_overrides(base, _ov(tail_scale=np.exp(c)))
    t0 = base.tails["chainlink_twap60"]
    t1 = m.tails["chainlink_twap60"]
    assert np.allclose(t1.sigma, t0.sigma * np.exp(c), rtol=1e-14)
    assert np.allclose(t1.mu, t0.mu * np.exp(c), rtol=1e-14)
    assert np.array_equal(t1.nu, t0.nu)


def test_tail_family_normal(base):
    from fvmodel.overrides import apply_overrides
    m = apply_overrides(base, _ov(tail_family="normal"))
    assert m.tails["chainlink_twap60"].family == "normal"


def test_alpha_scales(base):
    from fvmodel.overrides import apply_overrides

    m = apply_overrides(base, _ov(alpha_scale=2.0))
    assert np.allclose(m.alpha.beta0_bp, base.alpha.beta0_bp * 2.0)
    assert np.allclose(m.alpha.beta1_bp, base.alpha.beta1_bp * 2.0)
    m0 = apply_overrides(base, _ov(alpha_scale=0.0))
    assert np.all(m0.alpha.beta0_bp == 0.0) and np.all(m0.alpha.beta1_bp == 0.0)
    ma = apply_overrides(base, _ov(alpha_scale_age=0.0))
    assert np.allclose(ma.alpha.beta0_bp, base.alpha.beta0_bp)
    assert np.all(ma.alpha.beta1_bp == 0.0), "alpha_scale_age must touch beta1 only"


def test_eps_scale_scales_the_process(base):
    """Spec ruling R1 (amended): eps_scale=0 removes the term entirely, via the
    sigma_bp == 0.0 gate in engine.py that skips the whole eps block, mean and
    variance both. Away from zero, eps_scale scales sigma_bp only - see
    test_eps_scale_is_variance_only_away_from_zero below for the mean/variance
    split this produces."""
    from fvmodel.overrides import apply_overrides

    m = apply_overrides(base, _ov(eps_scale=2.0))
    assert m.eps.sigma_bp == pytest.approx(base.eps.sigma_bp * 2.0)
    m0 = apply_overrides(base, _ov(eps_scale=0.0))
    assert m0.eps.sigma_bp == 0.0


def test_eps_scale_is_variance_only_away_from_zero(base):
    """Spec ruling R1 (amended): eps_scale scales sigma_eps, so the eps VARIANCE
    contribution scales as eps_scale**2. The conditional-mean coefficient `c` in
    eps_bar = c * eps_last depends only on the fitted autocorrelation (model.rho),
    never on sigma - eps_conditional is always called with sigma=1.0 and the
    result is multiplied by sig**2 afterward (fvmodel/engine.py). So eps_scale
    cannot move the mean channel; that is eps_condition's job instead.

    At eps_scale = 0.0 the sigma_bp == 0.0 gate skips the whole block, so both c
    and the variance are zero there - not because eps_scale reached the mean
    channel, but because the term is off entirely.
    """
    from fvmodel.chainlink import eps_conditional
    from fvmodel.overrides import apply_overrides

    m1 = apply_overrides(base, _ov(eps_scale=1.0))
    m2 = apply_overrides(base, _ov(eps_scale=2.0))
    ages = np.array([0.0, 5.0, 12.0, 30.0])
    w = np.full(ages.size, 1.0 / ages.size)

    # eps_conditional itself is scale-free: it is always called with sigma=1.0 and
    # the model's rho/kappa are untouched by eps_scale, so c and unit-variance are
    # identical regardless of which model's eps this came from.
    c1, q1 = eps_conditional(m1.eps, ages, w, 1.0, "full")
    c2, q2 = eps_conditional(m2.eps, ages, w, 1.0, "full")
    assert c1 == pytest.approx(c2), "the mean coefficient must not depend on eps_scale"
    assert q1 == pytest.approx(q2), "the unit variance must not depend on eps_scale"

    # the variance that actually reaches var_eps is q_unit * sigma_at(v)**2, and
    # sigma_at scales linearly with sigma_bp - so the realised variance ratio is
    # eps_scale**2 = 4.0, while the mean channel (c) is untouched.
    v_local = 0.0005
    sig1 = m1.eps.sigma_at(v_local)
    sig2 = m2.eps.sigma_at(v_local)
    var1 = q1 * sig1 ** 2
    var2 = q2 * sig2 ** 2
    assert var2 / var1 == pytest.approx(4.0, rel=1e-12)

    # at eps_scale = 0.0 the block is off entirely: both channels are zero.
    m0 = apply_overrides(base, _ov(eps_scale=0.0))
    assert m0.eps.sigma_bp == 0.0
    c0, q0 = eps_conditional(m0.eps, ages, w, 1.0, "none")
    assert c0 == 0.0 and q0 == 0.0


def test_w_spot_reads_the_per_w_table(base):
    from fvmodel.overrides import apply_overrides

    m = apply_overrides(base, _ov(w_spot=0.35))
    assert m.fp.w_spot == pytest.approx(0.35)
    assert m.fp.tau_s == pytest.approx(0.9865)
    assert m.fp.delta_s == pytest.approx(0.70)


def test_w_spot_default_keeps_the_shipped_fine_fit(base):
    """R11: w_spot=None is the shipped filter, not the coarse row at w=0.6."""
    from fvmodel.overrides import apply_overrides

    m = apply_overrides(base, _ov())
    assert m.fp.tau_s == pytest.approx(0.8447)
    m6 = apply_overrides(base, _ov(w_spot=0.6))
    assert m6.fp.tau_s == pytest.approx(0.7872)


def test_w_spot_off_grid_is_rejected():
    from fvmodel.overrides import Overrides
    with pytest.raises(ValueError, match="w_spot"):
        Overrides.from_dict({"w_spot": 0.42})


def test_rho_kernel_off_gives_a_delta_kernel(base):
    from fvmodel.overrides import apply_overrides
    m = apply_overrides(base, _ov(rho_kernel="off"))
    r = m.rho_for(1.0)
    assert r[0] == 1.0 and np.all(r[1:] == 0.0)


def test_xi_cap_selects_the_fitted_register(base):
    from fvmodel.overrides import apply_overrides
    m = apply_overrides(base, _ov(xi_cap_c=2.0))
    assert m.ov.xi_cap_c == 2.0 and m.xi_cap_i is not None


def test_filter_params_from_dict_maps_every_field_correctly():
    """FilterParams.from_dict must pass keyword args, not six positionals into a
    seven-field dataclass (w_spot, tau_s, delta_s, basis_hl_s, basis_hl_alt_s,
    lag_s, p_stamp_late) - the old positional call shifted lag_s's value into
    basis_hl_alt_s and p_stamp_late's value into lag_s, and never set
    p_stamp_late at all."""
    import json
    from fvmodel.base import TABLES_DIR
    from fvmodel.chainlink import FilterParams

    d = json.load(open(TABLES_DIR / "print_model.json"))["filter"]["shipped"]
    fp = FilterParams.from_dict(d)
    assert fp.w_spot == pytest.approx(0.6)
    assert fp.tau_s == pytest.approx(0.8447)
    assert fp.delta_s == pytest.approx(0.7)
    assert fp.p_stamp_late == pytest.approx(0.02)
    assert fp.lag_s == 2
    assert fp.basis_hl_alt_s == pytest.approx(15.0)


def test_build_model_keeps_the_shipped_alt_tracker_half_life(base):
    """basis_hl_alt_s must reach build_model() at its dataclass default (15.0):
    Task 9's basis_alt variant scores the 15 s tracker NOTES E19 identified as the
    constrained optimum, not whatever value from_dict's old positional bug left it
    at."""
    assert base.fp.basis_hl_alt_s == pytest.approx(15.0)


def test_filter_params_round_trips_through_to_dict():
    from fvmodel.chainlink import FilterParams

    fp = FilterParams(w_spot=0.35, tau_s=0.9865, delta_s=0.70, basis_hl_s=901.0,
                      basis_hl_alt_s=16.0, lag_s=3, p_stamp_late=0.05)
    fp2 = FilterParams.from_dict(fp.to_dict())
    assert fp2.w_spot == fp.w_spot
    assert fp2.tau_s == fp.tau_s
    assert fp2.delta_s == fp.delta_s
    assert fp2.basis_hl_s == fp.basis_hl_s
    assert fp2.basis_hl_alt_s == fp.basis_hl_alt_s
    assert fp2.lag_s == fp.lag_s
    assert fp2.p_stamp_late == fp.p_stamp_late


def test_apply_overrides_does_not_mutate_the_input_model(base):
    """Two independent calls must not share mutable state through the base model."""
    from fvmodel.overrides import apply_overrides

    before_sigma = base.eps.sigma_bp
    before_beta0 = np.array(base.alpha.beta0_bp, copy=True)
    before_tail_sigma = {k: np.array(t.sigma, copy=True) for k, t in base.tails.items()}

    m1 = apply_overrides(base, _ov(eps_scale=2.0, alpha_scale=3.0, kappa_tail=-0.5))
    m2 = apply_overrides(base, _ov(eps_scale=5.0, alpha_scale=0.1, tail_scale=2.0))

    assert base.eps.sigma_bp == before_sigma
    assert np.array_equal(base.alpha.beta0_bp, before_beta0)
    for kind, t in base.tails.items():
        assert np.array_equal(t.sigma, before_tail_sigma[kind])

    assert m1.eps.sigma_bp != m2.eps.sigma_bp
    assert not np.array_equal(m1.alpha.beta0_bp, m2.alpha.beta0_bp)
