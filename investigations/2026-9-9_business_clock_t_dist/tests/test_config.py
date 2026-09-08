"""model.json is the single source of truth, and it verifies itself."""
from __future__ import annotations

import json

import pytest


def test_load_returns_a_model_and_provenance():
    from fvmodel import config

    loaded = config.load()
    assert loaded.cfg["version"].startswith("fv-")
    assert loaded.cfg["overrides"]["version"].startswith("ov-")
    assert loaded.provenance["params_version"] == "2.1.0"
    assert loaded.model.fp.w_spot == pytest.approx(0.6)
    assert loaded.model.fp.tau_s == pytest.approx(0.8447)


def test_every_table_hash_is_verified():
    from fvmodel import config

    cfg = json.loads(config.CONFIG_PATH.read_text())
    for name, entry in cfg["tables"].items():
        path = config.CONFIG_PATH.parent / entry["file"]
        assert path.exists(), "%s missing at %s" % (name, path)
        assert config.sha256_of(path) == entry["sha256"], (
            "%s hash mismatch; if you meant to change it, bump model.json's version "
            "and add a CHANGELOG entry" % name)
    assert config.sha256_of(
        config.CONFIG_PATH.parent / cfg["base"]["params"]) == cfg["base"]["params_sha256"]


def test_a_tampered_table_is_rejected(tmp_path, monkeypatch):
    from fvmodel import config

    cfg = json.loads(config.CONFIG_PATH.read_text())
    cfg["tables"]["tails"]["sha256"] = "0" * 64
    bad = tmp_path / "model.json"
    bad.write_text(json.dumps(cfg))
    monkeypatch.setattr(config, "CONFIG_PATH", bad)
    # the tables the tampered config points at still live next to the real one
    monkeypatch.setattr(config, "_table_root", lambda: config.ROOT)
    with pytest.raises(config.ProvenanceError, match="tails"):
        config.load()


def test_filter_by_w_covers_the_shipped_weight():
    from fvmodel import config

    tbl = config.filter_by_w()
    ws = [r[0] for r in tbl]
    assert len(tbl) == 13 and ws == sorted(ws)
    assert 0.6 in ws and 0.0 in ws and 1.0 in ws
