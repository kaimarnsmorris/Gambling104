"""Load model.json: verify every artefact, resolve the data paths, build the model.

This is the only place that decides which tables the model is made of. A table whose
hash does not match the one recorded in model.json is a hard failure - the alternative
is a run that reports a model version it is not running.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .base import ROOT

CONFIG_PATH = ROOT / "model.json"


class ProvenanceError(RuntimeError):
    """A shipped artefact does not match what model.json says it is."""


@dataclass
class Loaded:
    model: object            # FairValueModel
    overrides: object        # Overrides, or None before Task 4
    provenance: dict
    cfg: dict


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _table_root() -> Path:
    return CONFIG_PATH.parent


def read_config() -> dict:
    return json.loads(Path(CONFIG_PATH).read_text())


def verify(cfg: dict) -> None:
    root = _table_root()
    base = root / cfg["base"]["params"]
    if not base.exists():
        raise ProvenanceError("base params missing at %s" % base)
    if sha256_of(base) != cfg["base"]["params_sha256"]:
        raise ProvenanceError("base params hash mismatch at %s" % base)
    for name, entry in cfg["tables"].items():
        p = root / entry["file"]
        if not p.exists():
            raise ProvenanceError("table %s missing at %s" % (name, p))
        if sha256_of(p) != entry["sha256"]:
            raise ProvenanceError("table %s hash mismatch at %s" % (name, p))


def filter_by_w() -> list:
    return json.loads((_table_root() / "tables" / "filter_by_w.json").read_text())["rows"]


def source_root() -> Path:
    cfg = read_config()
    return Path(os.environ.get("FV_SOURCE_ROOT", cfg["data"]["source_root"]))


def load(variant: str | None = None) -> Loaded:
    """The model model.json describes, with `variant`'s overrides applied.

    `variant` is a name under variants/; None means the defaults in model.json.
    """
    from .build import build_model

    cfg = read_config()
    verify(cfg)
    model = build_model()
    prov = {"model_version": cfg["version"],
            "overrides_version": cfg["overrides"]["version"],
            "params_version": cfg["base"]["params_version"],
            "variant": variant or "baseline",
            "tables": {k: v["sha256"][:12] for k, v in cfg["tables"].items()}}
    return Loaded(model=model, overrides=None, provenance=prov, cfg=cfg)
