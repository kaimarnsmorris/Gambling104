"""Pins three-level block resolution: investigation, model, defaults.

Copying blocks into each investigation was right for a one-off evaluation and
is why fair.py came to exist in three places that drifted -- the 2026-09-09
fair fix had to be applied twice by hand. Provenance is unaffected: the run
folder still freezes the resolved files with sha256s.
"""
import os

from harness.core import provenance


def test_a_model_file_beats_the_default(tmp_path):
    model = tmp_path / "model"
    os.makedirs(str(model))
    (model / "link.py").write_text("def link(z):\n    return 0.25\n")
    resolved = provenance.resolve_slots(str(tmp_path / "inv"),
                                       model_dir=str(model))
    assert resolved["link"] == str(model / "link.py")
    assert "defaults" in resolved["quote"]


def test_the_investigation_beats_the_model(tmp_path):
    inv, model = tmp_path / "inv", tmp_path / "model"
    os.makedirs(str(inv)); os.makedirs(str(model))
    (model / "link.py").write_text("def link(z):\n    return 0.25\n")
    (inv / "link.py").write_text("def link(z):\n    return 0.75\n")
    resolved = provenance.resolve_slots(str(inv), model_dir=str(model))
    assert resolved["link"] == str(inv / "link.py")


def test_no_model_dir_behaves_exactly_as_before(tmp_path):
    resolved = provenance.resolve_slots(str(tmp_path))
    assert all("defaults" in p for p in resolved.values())
