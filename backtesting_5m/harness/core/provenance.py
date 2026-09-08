"""Which files were this model, and can we prove it later.

Blocks are files, resolved investigation-first and defaults-second. Nothing is
registered and nothing is named: dropping `link.py` into an investigation
folder IS the override.

A run freezes copies of every block file it used, so the run folder remains a
complete answer to 'what was this model?' after the investigation moves on.

Fingerprinting a data file writes NOTHING next to that file. `data/` is this
harness's read-only input; the digest cache lives under `paths.FINGERPRINTS`,
inside the harness, keyed by the absolute path of the file it describes.
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


def _cache_path(path):
    """Where the cached digest for `path` lives -- in the harness, not in data.

    NOT a sidecar next to the file. `data/` is read-only input from the
    harness's point of view everywhere else in this project, and a cache is no
    reason to make an exception of it. The name carries a hash of the absolute
    path, so two data files sharing a basename cannot collide on one cache
    entry and hand back each other's digest.
    """
    full = os.path.abspath(path)
    tag = hashlib.sha256(full.encode("utf-8")).hexdigest()[:16]
    return os.path.join(paths.FINGERPRINTS,
                        f"{os.path.basename(full)}.{tag}.sha256")


def fingerprint_input(path):
    """Cheap, cached identity for a large data file.

    Keyed on (size, mtime), under `paths.FINGERPRINTS`. A miss costs only a
    re-hash, so a cache that cannot be read or written is never wrong -- which
    is why both failures are swallowed.
    """
    if not os.path.exists(path):
        return {"path": path, "present": False}
    stat = os.stat(path)
    side = _cache_path(path)
    key = f"{stat.st_size}:{int(stat.st_mtime)}"
    digest = None
    if os.path.exists(side):
        try:
            cached_key, cached = open(side).read().split("\n")[:2]
        except (OSError, ValueError):
            cached_key = cached = None
        if cached_key == key:
            digest = cached
    if digest is None:
        digest = sha256_file(path)
        try:
            os.makedirs(paths.FINGERPRINTS, exist_ok=True)
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
    config_hash = _config_hash(config)
    counter = None
    while True:
        if counter is None:
            run_name = f"{stamp}__{config_hash}"
        else:
            run_name = f"{stamp}-{counter}__{config_hash}"
        run_dir = os.path.join(investigation_dir, "runs", run_name)
        if not os.path.exists(run_dir):
            break
        counter = 2 if counter is None else counter + 1
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
