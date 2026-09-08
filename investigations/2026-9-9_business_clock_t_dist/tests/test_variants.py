from __future__ import annotations

import pytest

SHIPPED = ["baseline", "market_observable", "no_alpha", "alpha_x2", "alpha_half",
           "vol_plus10", "vol_minus10", "short_vol_minus20", "shrink_05",
           "fat_tails", "thin_tails", "normal_tail", "eps_x0", "eps_x2",
           "basis_alt", "rho_off"]


def test_every_shipped_variant_exists():
    from fvmodel.variants import list_variants
    assert sorted(list_variants()) == sorted(SHIPPED)


@pytest.mark.parametrize("name", SHIPPED)
def test_every_variant_builds_a_valid_overrides(name):
    from fvmodel.overrides import Overrides
    from fvmodel.variants import read_variant

    v = read_variant(name)
    assert v["base_params"].startswith("model.json@fv-")
    assert v["notes"].strip(), "%s has no notes" % name
    Overrides.from_dict(v["overrides"])


def test_baseline_is_all_defaults():
    from fvmodel.overrides import Overrides
    from fvmodel.variants import read_variant

    assert Overrides.from_dict(read_variant("baseline")["overrides"]).label() == "baseline"


def test_variants_are_distinct():
    from fvmodel.overrides import Overrides
    from fvmodel.variants import list_variants, read_variant

    labels = {n: Overrides.from_dict(read_variant(n)["overrides"]).label()
              for n in list_variants()}
    assert len(set(labels.values())) == len(labels), "duplicate variants: %s" % labels


def test_sweep_expands_to_the_product_and_respects_the_cap():
    from fvmodel.variants import expand_sweep

    spec = {"name_prefix": "vol", "max_variants": 10,
            "grid": {"kappa_vol": [-0.1, 0.0, 0.1],
                     "kappa_tail": [-0.5, 0.0]}}
    out = expand_sweep(spec)
    assert len(out) == 6
    assert all(v["name"].startswith("vol_") for v in out)
    assert len({v["name"] for v in out}) == 6


def test_sweep_requires_a_name_prefix():
    from fvmodel.variants import expand_sweep
    with pytest.raises(ValueError, match="name_prefix"):
        expand_sweep({"max_variants": 4, "grid": {"kappa_vol": [0.0, 0.1]}})


def test_sweep_cap_is_enforced_before_anything_runs():
    from fvmodel.variants import expand_sweep
    with pytest.raises(ValueError, match="max_variants"):
        expand_sweep({"name_prefix": "big", "max_variants": 3,
                      "grid": {"kappa_vol": [0.0, 0.1, 0.2],
                               "tail_scale": [1.0, 1.1]}})


def test_sweep_generation_does_not_change_the_shipped_variant_list():
    """Running a sweep writes to variants/generated/, never to the top level, so
    list_variants() - the shipped set - is unaffected."""
    from fvmodel.variants import GENERATED, expand_sweep, list_variants
    import yaml

    before = sorted(list_variants())
    spec = {"name_prefix": "zzsweep_test", "max_variants": 4,
            "grid": {"kappa_vol": [0.0, 0.1]}}
    variants = expand_sweep(spec)
    GENERATED.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        for v in variants:
            p = GENERATED / ("%s.yaml" % v["name"])
            p.write_text(yaml.safe_dump(v, sort_keys=False))
            written.append(p)
        assert sorted(list_variants()) == before
    finally:
        for p in written:
            p.unlink(missing_ok=True)
