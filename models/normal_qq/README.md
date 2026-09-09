# Normal QQ Model

This directory contains the canonical block set for the normal Q-Q distribution investigation.

## Source and Drift Record

**Files copied from:** `Gambling104/backtesting_5m/investigations/2026-09-09-normal-qq-eval/`

**Date copied:** 2026-09-09

**Upstream original:** These blocks originated in `Gambling104/investigations/2026-09-09_normal_qq_basic/`, a concurrent investigation session. That directory is not modified by this harness and is the authoritative upstream reference for this model's provenance.

## Drift Evidence

At the time of copying to this shared model directory, the four block files were verified across the two investigations:

- `fair.py`: **identical** in `2026-09-09-normal-qq-eval` and `2026-09-09-vol-fixed`
- `vol.py`: **DIFFERS** — baseline EWMA implementation vs. calibrated lookup-table variant
- `f.py`: **identical** in both investigations
- `link.py`: **identical** in both investigations

**Why vol.py diverged:** The `2026-09-09-vol-fixed` investigation layered an experiment on top of this baseline model, replacing `vol.py` with a calibrated lookup-table implementation that reads panel-specific fitted parameters from `sigma_fit.json`. That variant belongs in the investigation that tested it, not in the shared canonical model.

**Resolution order:** The `2026-09-09-vol-fixed` investigation retains its own local `vol.py`, which shadows the model's by design (investigation → model → defaults). This is the intended use of shadowing — forking a single block remains a one-file act.

## Provenance Freezing

Every run freezes copies of all resolved block files into `runs/<id>/blocks/` with sha256 checksums recorded in `manifest.json`. This means:

- A run always answers "what exactly was this model?" byte-for-byte, regardless of later changes to the model or investigation directories.
- Changes to these files do not affect already-completed runs.
- The model directory eliminates the copy-per-investigation pattern that led to this drift (fair.py existed in three places and had to be patched twice by hand before this refactor).
