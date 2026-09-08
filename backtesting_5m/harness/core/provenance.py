"""Which files were this model, and can we prove it later.

Blocks are files, resolved investigation-first and defaults-second. Nothing is
registered and nothing is named: dropping `link.py` into an investigation
folder IS the override.

A run freezes copies of every block file it used, so the run folder remains a
complete answer to 'what was this model?' after the investigation moves on.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone

from harness import paths

SLOTS = ("fair", "vol", "f", "link", "quote", "execution", "fill", "fees")

DEFAULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "blocks", "defaults")


def resolve_slots(investigation_dir):
    """slot -> path. Investigation folder wins; defaults fill the rest."""
    resolved = {}
    for slot in SLOTS:
        local = os.path.join(investigation_dir, f"{slot}.py")
        resolved[slot] = local if os.path.exists(local) else os.path.join(
            DEFAULTS_DIR, f"{slot}.py")
    return resolved


def load_slot(path, slot):
    spec = importlib.util.spec_from_file_location(f"_block_{slot}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fingerprint_input(path):
    """Cheap, cached identity for a large data file."""
    if not os.path.exists(path):
        return {"path": path, "present": False}
    stat = os.stat(path)
    side = f"{path}.sha256"
    key = f"{stat.st_size}:{int(stat.st_mtime)}"
    digest = None
    if os.path.exists(side):
        cached_key, cached = open(side).read().split("\n")[:2]
        if cached_key == key:
            digest = cached
    if digest is None:
        digest = sha256_file(path)
        try:
            open(side, "w").write(f"{key}\n{digest}\n")
        except OSError:
            pass
    return {"path": path, "present": True, "bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc).isoformat(),
            "sha256": digest}


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=paths.PROJECT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _config_hash(config):
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:6]


def new_run_dir(investigation_dir, config):
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join(investigation_dir, "runs",
                           f"{stamp}__{_config_hash(config)}")
    os.makedirs(os.path.join(run_dir, "blocks"), exist_ok=True)
    return run_dir


def write_manifest(run_dir, config, resolved, inputs=(), seeds=()):
    blocks = {}
    for slot, path in resolved.items():
        blocks[slot] = {"path": path, "sha256": sha256_file(path)}
        shutil.copy2(path, os.path.join(run_dir, "blocks", f"{slot}.py"))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "harness_commit": _git_commit(),
        "blocks": blocks,
        "config": config,
        "inputs": [fingerprint_input(p) for p in inputs],
        "seeds": list(seeds),
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return manifest
