"""Pins block resolution and run provenance.

A run folder must still answer 'what exactly was this model?' after the
investigation folder has moved on, which means the block FILES travel with the
run, not just their paths.
"""
import json
import os

import pytest

from harness import paths
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


def test_fingerprinting_writes_nothing_beside_the_data_file(tmp_path,
                                                            monkeypatch):
    """`data/` is read-only input. The digest cache belongs to the harness.

    It used to be a `.sha256` sidecar written next to the parquet, which put
    harness output inside the one tree this project treats as untouchable.
    """
    cache = tmp_path / "cache"
    monkeypatch.setattr(paths, "FINGERPRINTS", str(cache))
    data = tmp_path / "data"
    data.mkdir()
    parquet = data / "spot_5m_100ms.parquet"
    parquet.write_bytes(b"x" * 64)

    first = provenance.fingerprint_input(str(parquet))
    assert [q.name for q in data.iterdir()] == ["spot_5m_100ms.parquet"], (
        "the harness wrote into the data tree")
    assert cache.is_dir() and any(cache.iterdir())

    cached = provenance.fingerprint_input(str(parquet))
    assert cached["sha256"] == first["sha256"]
    assert first["sha256"] == provenance.sha256_file(str(parquet))


def test_two_inputs_sharing_a_basename_do_not_share_a_cache_entry(
        tmp_path, monkeypatch):
    """Same name, same size, same mtime -- and different bytes.

    A cache keyed on the basename alone would hand the second file the first
    one's digest, and the manifest would swear to a file that was never read.
    """
    monkeypatch.setattr(paths, "FINGERPRINTS", str(tmp_path / "cache"))
    digests = []
    for folder, body in (("a", b"a" * 64), ("b", b"b" * 64)):
        d = tmp_path / folder
        d.mkdir()
        f = d / "panel.parquet"
        f.write_bytes(body)
        os.utime(f, (1_786_665_600, 1_786_665_600))
        fp = provenance.fingerprint_input(str(f))
        assert fp["sha256"] == provenance.sha256_file(str(f))
        digests.append(fp["sha256"])
    assert digests[0] != digests[1]


def test_two_runs_of_the_same_config_do_not_collide(tmp_path):
    """A second run must never silently overwrite the first one's manifest."""
    a = provenance.new_run_dir(str(tmp_path), {"e_p": 0.01})
    b = provenance.new_run_dir(str(tmp_path), {"e_p": 0.01})
    assert a != b
    assert a.split("__")[-1] == b.split("__")[-1], "config hash still shared"
