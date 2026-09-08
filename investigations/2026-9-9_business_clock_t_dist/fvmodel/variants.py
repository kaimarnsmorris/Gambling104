"""A variant is a file. Reading one never runs anything and never fits anything.

Shipped variants live at the top level of `variants/`. Sweep-generated variants are
written to `variants/generated/` instead (see `run_variants.py`) so that running a
sweep never changes what `list_variants()` reports - the shipped set is fixed, and
`tests/test_variants.py::test_every_shipped_variant_exists` depends on that.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import yaml

from .base import ROOT

VARIANTS = ROOT / "variants"
GENERATED = VARIANTS / "generated"


def _path(name: str) -> Path:
    for base in (VARIANTS, GENERATED):
        for ext in (".yaml", ".yml", ".json"):
            p = base / (name + ext)
            if p.exists():
                return p
    raise FileNotFoundError("no variant %r in %s (or %s); have %s"
                            % (name, VARIANTS, GENERATED, list_variants()))


def list_variants() -> list:
    """The shipped set only: the top level of `variants/`, never `generated/` or
    `sweeps/` beneath it."""
    return sorted({p.stem for p in VARIANTS.glob("*")
                   if p.is_file() and p.suffix in (".yaml", ".yml", ".json")})


def read_variant(name: str) -> dict:
    p = _path(name)
    d = (json.loads(p.read_text()) if p.suffix == ".json"
         else yaml.safe_load(p.read_text()))
    for k in ("name", "base_params", "overrides", "notes"):
        if k not in d:
            raise ValueError("variant %s is missing %r" % (p, k))
    if d["name"] != name:
        raise ValueError("variant %s declares name %r" % (p, d["name"]))
    return d


def expand_sweep(spec: dict) -> list:
    """The cartesian product of `grid`, capped, with the cap checked FIRST.

    Checking the cap before expanding is the point: a four-key grid is easy to write
    and expensive to discover by running it.

    Note on names: the tag uses `k[:6]` per grid key, so `kappa_vol` and `kappa_tail`
    both render as `kappa_` - they stay distinguishable only because their values
    differ in the same tag. Don't add a third `kappa_*` override to the same grid
    without checking uniqueness by hand.
    """
    prefix = spec.get("name_prefix")
    if not prefix:
        raise ValueError("a sweep needs a name_prefix, so its variants are "
                         "identifiable and cannot collide with the shipped set")
    cap = int(spec.get("max_variants", 0))
    if cap <= 0:
        raise ValueError("a sweep needs a positive max_variants")
    grid = spec["grid"]
    keys = sorted(grid)
    total = 1
    for k in keys:
        total *= len(grid[k])
    if total > cap:
        raise ValueError("sweep expands to %d variants, above max_variants = %d"
                         % (total, cap))
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        ov = dict(zip(keys, combo))
        tag = "_".join("%s%s" % (k[:6], ("%g" % v).replace("-", "m").replace(".", "p"))
                       for k, v in ov.items())
        out.append({"name": "%s_%s" % (prefix, tag),
                    "base_params": spec.get("base_params", "model.json@fv-1.0.0"),
                    "overrides": ov, "notes": spec.get("notes", "swept")})
    return out
