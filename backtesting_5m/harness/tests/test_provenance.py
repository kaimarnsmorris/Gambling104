"""Pins block resolution and run provenance.

A run folder must still answer 'what exactly was this model?' after the
investigation folder has moved on, which means the block FILES travel with the
run, not just their paths.
"""
import json
import os

import pytest

from harness.core import provenance


def test_every_slot_falls_back_to_defaults(tmp_path):
    resolved = provenance.resolve_slots(str(tmp_path))
    assert set(resolved) == set(provenance.SLOTS)
    assert all("defaults" in p for p in resolved.values())


def test_an_investigation_file_wins(tmp_path):
    (tmp_path / "link.py").write_text("def link(z):\n    return 0.5\n")
    resolved = provenance.resolve_slots(str(tmp_path))
    assert resolved["link"] == str(tmp_path / "link.py")
    assert "defaults" in resolved["quote"], "other slots still fall back"


def test_a_resolved_override_actually_loads(tmp_path):
    (tmp_path / "link.py").write_text("def link(z):\n    return 0.25\n")
    resolved = provenance.resolve_slots(str(tmp_path))
    module = provenance.load_slot(resolved["link"], "link")
    assert module.link(3.0) == 0.25


def test_the_manifest_records_a_hash_for_every_slot(tmp_path):
    resolved = provenance.resolve_slots(str(tmp_path))
    run_dir = provenance.new_run_dir(str(tmp_path), {"params": {}})
    provenance.write_manifest(run_dir, {"params": {}}, resolved)

    manifest = json.loads(
        open(os.path.join(run_dir, "manifest.json")).read())
    assert set(manifest["blocks"]) == set(provenance.SLOTS)
    assert all(len(v["sha256"]) == 64 for v in manifest["blocks"].values())


def test_the_block_files_are_frozen_into_the_run(tmp_path):
    (tmp_path / "link.py").write_text("def link(z):\n    return 0.25\n")
    resolved = provenance.resolve_slots(str(tmp_path))
    run_dir = provenance.new_run_dir(str(tmp_path), {})
    provenance.write_manifest(run_dir, {}, resolved)

    frozen = os.path.join(run_dir, "blocks", "link.py")
    assert os.path.exists(frozen)
    assert "0.25" in open(frozen).read()


def test_identical_configs_produce_the_same_run_hash(tmp_path):
    a = provenance.new_run_dir(str(tmp_path), {"e_p": 0.01})
    b = provenance.new_run_dir(str(tmp_path), {"e_p": 0.01})
    c = provenance.new_run_dir(str(tmp_path), {"e_p": 0.02})
    assert a.split("__")[-1] == b.split("__")[-1]
    assert a.split("__")[-1] != c.split("__")[-1]


def test_hashing_a_file_is_stable(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    assert provenance.sha256_file(str(p)) == provenance.sha256_file(str(p))


def test_a_missing_input_is_fingerprinted_as_absent(tmp_path):
    fp = provenance.fingerprint_input(str(tmp_path / "nope.parquet"))
    assert fp["present"] is False
