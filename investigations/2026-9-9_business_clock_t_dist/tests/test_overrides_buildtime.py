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
    """Spec ruling R1: eps_scale=0 must remove the term, mean as well as variance."""
    from fvmodel.overrides import apply_overrides

    m = apply_overrides(base, _ov(eps_scale=2.0))
    assert m.eps.sigma_bp == pytest.approx(base.eps.sigma_bp * 2.0)
    m0 = apply_overrides(base, _ov(eps_scale=0.0))
    assert m0.eps.sigma_bp == 0.0


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
