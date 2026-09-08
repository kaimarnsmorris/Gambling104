"""Every branch that survives must be selectable from an override key, and the
investigation root must contain only the block slots we mean to override."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

SLOTS = {"fair.py", "vol.py", "f.py", "link.py"}
OTHER_SLOTS = {"quote.py", "execution.py", "fill.py", "fees.py"}


def test_switches_is_gone():
    import fvmodel.fairvalue as fvv
    assert not hasattr(fvv, "Switches"), (
        "Switches is redundant with Overrides; two configuration objects is exactly "
        "the drift the consolidation is meant to remove")


def test_fair_value_takes_no_switches():
    from fvmodel.fairvalue import fair_value
    p = inspect.signature(fair_value).parameters
    assert list(p) == ["state", "market", "model", "t_now"]


def test_evaluate_takes_no_switches():
    from fvmodel.engine import evaluate
    p = inspect.signature(evaluate).parameters
    assert "sw" not in p and "emit_all" in p


@pytest.mark.parametrize("dead", ["iid", "v2_twap", "composed_weights", "use_blend",
                                  "jensen", "market_observable"])
def test_dead_branch_names_are_gone_from_the_model(dead):
    root = Path(__file__).resolve().parents[1] / "fvmodel"
    hits = [p for p in root.glob("*.py")
            if dead in p.read_text(encoding="utf-8", errors="ignore")]
    assert not hits, "%r still appears in %s" % (dead, [p.name for p in hits])


def test_eps_conditional_modes_match_the_overrides():
    from fvmodel.chainlink import EPS_MODES
    assert set(EPS_MODES) == {"full", "unconditional", "none"}


def test_investigation_root_holds_only_the_intended_slots():
    root = Path(__file__).resolve().parents[1]
    present = {p.name for p in root.glob("*.py")}
    assert not (present & OTHER_SLOTS), (
        "these filenames are harness block slots and would silently override the "
        "defaults: %s" % sorted(present & OTHER_SLOTS))
