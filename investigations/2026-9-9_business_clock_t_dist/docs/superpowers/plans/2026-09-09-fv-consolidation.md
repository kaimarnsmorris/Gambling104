# Fair-Value Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the Chainlink fair-value model as four block files the 5 m backtesting harness can run, with an `overrides` layer so a model variant is a data change, and prove the default variant is bit-identical to the model we have today.

**Architecture:** Vendor `fv/` + `clkit/` + the eight `rvforecast` modules it needs into `fvmodel/` under this investigation folder. Collapse `Switches` and the loose `FairValueModel` knobs into one frozen `Overrides` dataclass that is the model's only configuration. Move `kappa_vol` from the register bank to the forward curve, which makes the register bank variant-invariant and so cacheable across every variant run. A builder writes a strike-free per-(market, second) fair export; the four root-level block files are thin readers over it.

**Tech Stack:** Python 3.13, numpy 2.2, scipy 1.15, polars 1.40, numba 0.62, pyyaml 6.0, pytest 8.4. No new dependencies.

**Spec:** `investigations/2026-9-9_business_clock_t_dist/docs/superpowers/specs/2026-09-09-fv-consolidation-design.md`

## Global Constraints

- **Working directory** for every path in this plan is `C:\Users\kaima\Github2\Gambling104\investigations\2026-9-9_business_clock_t_dist` unless stated otherwise. Call it `INV`.
- **Source repo**, read-only for the whole plan: `C:\Users\kaima\OneDrive\Documents\GitHub\autoresearch\btc_volatility_clock_chainlink`. Call it `SRC`. Its sibling `C:\Users\kaima\OneDrive\Documents\GitHub\autoresearch\btc_volatility_clock` is `VC`. **Never modify `SRC` or `VC`.**
- **Backtest window:** `t0 = 1786665600` (2026-08-14 00:00:00 UTC), `t1 = 1788220799` (2026-08-31 23:59:59 UTC). Selection sub-window ends `1787702399` (2026-08-25 23:59:59 UTC); holdout is `1787702400 .. 1788220799`.
- **Burn-in:** every `Window` is opened `14 * 86400` seconds before `t0`.
- **Every override defaults to its off value**, and with all defaults every output must be bit-identical to the pre-consolidation model. Task 1's golden file is the enforcement.
- **The investigation root is the harness's block-slot namespace.** Only `fair.py`, `vol.py`, `f.py`, `link.py` may exist there as slot names. Never create root-level `quote.py`, `execution.py`, `fill.py`, `fees.py`.
- **Shell is Git Bash or PowerShell on Windows.** Commands below are given for Git Bash.
- Run tests with `cd "$INV" && python -m pytest`. Slow tests are marked `@pytest.mark.slow` and need the 1 s data present.

---

## File Structure

| file | responsibility |
|---|---|
| `fvmodel/_vendor/rvforecast/*` | verbatim copy of `config, distribution, forward, kernels, regression, seasonality, streaming, tzclock` + trimmed `__init__` |
| `fvmodel/_vendor/clkit/{common,state}.py` | verbatim copy; `BatchV2` and the aligned-frame loader |
| `fvmodel/{base,weights,chainlink,curve,variance,alpha,tails,fairvalue,engine,build}.py` | the model, copied then edited |
| `fvmodel/overrides.py` | `Overrides`, `apply_overrides`, `xi_adjust`, `cap_m_Y`, `quoted_prob` — the only place override semantics live |
| `fvmodel/config.py` | loads `model.json`, verifies hashes, resolves data paths, returns `Loaded` |
| `model.json`, `CHANGELOG.md`, `tables/*` | the consolidated config and the fitted artefacts |
| `export/build_export.py` | variant → `backtesting_5m/data/fair/<variant>.parquet` |
| `export/cache.py` | L0–L2 cache layers keyed by content hash |
| `fair.py`, `vol.py`, `f.py`, `link.py` | the four harness block slots |
| `_fvexport.py` | shared export loader for the blocks; deliberately not a slot name |
| `variants/*.yaml`, `variants/README.md`, `variants/sweeps/*.yaml` | the variant set |
| `run_variants.py` | CLI: build exports and score them, with the `--holdout` guard |
| `score/scorecard.py`, `score/report_variants.py` | calibration metrics and the HTML report |
| `tests/*` | as listed per task |

---

## Task 1: Golden capture

Freeze the current model's outputs **before anything moves**. Everything after this is measured against it.

**Files:**
- Create: `tools/capture_golden.py`
- Create: `tests/golden/fv_golden.npz` (generated, committed — it is ~40 KB)

**Interfaces:**
- Produces: `tests/golden/fv_golden.npz` with arrays `kind` (str), `n`, `it`, `p_up`, `var_y`, `y_star`, `m_Y`, `eps_bar`, `var_eps`, `var_basis`, `carry`, `omega`, `known_value`, `nu`, `mu`, `sigma_t`, `z`, `p_ref`, and scalar metadata `window_t0`, `window_t1`.

- [ ] **Step 1: Write the capture script**

Create `tools/capture_golden.py`:

```python
"""Freeze the pre-consolidation model's outputs, so the consolidation can be proved
bit-identical. Runs against the SOURCE repo, never against the vendored copy.

    python tools/capture_golden.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(r"C:\Users\kaima\OneDrive\Documents\GitHub\autoresearch"
           r"\btc_volatility_clock_chainlink")
VC = SRC.parent / "btc_volatility_clock"
for p in (str(SRC), str(VC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fv.base import CL_T0, load_params            # noqa: E402
from fv.build import build_model                  # noqa: E402
from fv.engine import Window, market_at, state_at  # noqa: E402
from fv.fairvalue import Switches, fair_value     # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tests" / "golden" / "fv_golden.npz"
WIN_T0 = CL_T0 + 20 * 86400
WIN_T1 = WIN_T0 + 3 * 86400
# (kind, market length L, seconds remaining n) - spans the composed-weight regime
# (n < 60), the crossover, and the long horizons where the head integral dominates
CASES = [("chainlink_twap60", 300, n) for n in (300, 120, 60, 30, 10, 5, 2, 1)]
CASES += [("chainlink_twap60", 900, n) for n in (900, 300, 60, 5)]
CASES += [("perp_twap", 900, n) for n in (900, 120, 30)]
CASES += [("perp_single", 900, n) for n in (900, 300, 10, 1)]
BOOK = {"imbalance": 0.8, "px_age_s": 1.4}


def main() -> None:
    model = build_model()
    win = Window(load_params(), model.fp, WIN_T0, WIN_T1, warm=0, chainlink=True,
                 log=lambda *a: None)
    # eight quote origins spread across the window, all past the register burn-in
    its = np.linspace(win.n - 200_000, win.n - 5_000, 8).astype(np.int64)
    win.prepare(np.concatenate([its, its - 2]))

    cols = {k: [] for k in ("kind", "n", "it", "p_up", "var_y", "y_star", "m_Y",
                            "eps_bar", "var_eps", "var_basis", "carry", "omega",
                            "known_value", "nu", "mu", "sigma_t", "z", "p_ref")}
    for it in its:
        state = state_at(win, int(it), model)
        state.book = dict(BOOK)
        t = int(win.ts[it])
        for kind, L, n in CASES:
            mk = market_at(win, kind, t + n, n, L, 60)
            if not np.isfinite(mk.strike):
                continue
            fv = fair_value(state, mk, model, Switches(), t_now=t)
            cols["kind"].append(kind)
            cols["n"].append(n)
            cols["it"].append(int(it))
            for k in ("p_up", "var_y", "y_star", "m_Y", "eps_bar", "var_eps",
                      "var_basis", "carry", "omega", "known_value", "z", "p_ref"):
                cols[k].append(float(getattr(fv, k)))
            cols["nu"].append(float(fv.tail[0]))
            cols["mu"].append(float(fv.tail[1]))
            cols["sigma_t"].append(float(fv.tail[2]))

    assert len(cols["p_up"]) >= 100, "too few golden rows: %d" % len(cols["p_up"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, window_t0=WIN_T0, window_t1=WIN_T1,
             kind=np.array(cols.pop("kind")),
             **{k: np.asarray(v) for k, v in cols.items()})
    print("wrote %s: %d rows" % (OUT, len(cols["p_up"])))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd "$INV" && python tools/capture_golden.py`
Expected: `wrote .../fv_golden.npz: 152 rows` (the exact count may differ if some strikes are non-finite; anything ≥ 100 is fine).

- [ ] **Step 3: Sanity-check the capture is not degenerate**

Run:
```bash
cd "$INV" && python -c "
import numpy as np
z = np.load('tests/golden/fv_golden.npz')
p = z['p_up']
print('rows', p.size, 'finite', np.isfinite(p).all())
print('p_up range', p.min(), p.max(), 'distinct', np.unique(p).size)
print('var_y > 0', (z['var_y'] > 0).all())
print('m_Y nonzero', (z['m_Y'] != 0).sum(), 'eps_bar nonzero', (z['eps_bar'] != 0).sum())
"
```
Expected: all finite, `distinct` close to `rows` (not a constant), `var_y > 0` True, and both `m_Y` and `eps_bar` non-zero on a good fraction of rows — otherwise the alpha and ε paths are not being exercised and the golden file cannot catch a regression in them.

- [ ] **Step 4: Commit**

```bash
cd /c/Users/kaima/Github2/Gambling104
git add investigations/2026-9-9_business_clock_t_dist/tools/capture_golden.py \
        investigations/2026-9-9_business_clock_t_dist/tests/golden/fv_golden.npz
git commit -m "test: freeze pre-consolidation fair-value outputs as the golden file"
```

---

## Task 2: Vendor the model

Copy the code in, fix the imports, and make the golden file pass against the vendored copy. No behaviour changes.

**Files:**
- Create: `fvmodel/` (see File Structure), `pytest.ini`, `tests/conftest.py`, `tests/test_golden.py`
- Create: `tables/params_v2_1_0.json` and the five fitted-table copies

**Interfaces:**
- Consumes: `tests/golden/fv_golden.npz` from Task 1.
- Produces: `fvmodel.build.build_model()`, `fvmodel.fairvalue.fair_value(state, market, model, sw=None, t_now=None)`, `fvmodel.engine.{Window, evaluate, state_at, market_at, market_grid}`, `fvmodel.base.{CACHE, TABLES_DIR, load_params}` — same signatures as `SRC` at this point.

- [ ] **Step 1: Copy the trees**

```bash
cd "$INV"
SRC="/c/Users/kaima/OneDrive/Documents/GitHub/autoresearch/btc_volatility_clock_chainlink"
VC="/c/Users/kaima/OneDrive/Documents/GitHub/autoresearch/btc_volatility_clock"
mkdir -p fvmodel/_vendor/rvforecast fvmodel/_vendor/clkit tables
cp "$SRC"/fv/*.py fvmodel/
cp "$SRC"/clkit/common.py "$SRC"/clkit/state.py fvmodel/_vendor/clkit/
touch fvmodel/_vendor/__init__.py fvmodel/_vendor/clkit/__init__.py
for m in config distribution forward kernels regression seasonality streaming tzclock; do
  cp "$VC/rvforecast/$m.py" fvmodel/_vendor/rvforecast/
done
cp "$VC"/output/params.json tables/params_v2_1_0.json
for t in print_model kernels alpha tails xi_cap; do cp "$SRC/cache/$t.json" tables/; done
ls fvmodel fvmodel/_vendor/rvforecast tables
```

- [ ] **Step 2: Write the trimmed vendored `rvforecast/__init__.py`**

Create `fvmodel/_vendor/rvforecast/__init__.py`:

```python
"""The subset of rvforecast v2.1 that the fair-value model needs.

Vendored verbatim from btc_volatility_clock. Do not edit these modules: they are
the shipped v2.1 forecaster and `tables/params_v2_1_0.json` was fitted by them.
The upstream package's __init__ re-exports the whole pipeline; this one exports
only what fv imports, so nothing drags in the fitting or evaluation stack.
"""
from . import config, distribution, forward, kernels, regression  # noqa: F401
from . import seasonality, streaming, tzclock                     # noqa: F401

__all__ = ["config", "distribution", "forward", "kernels", "regression",
           "seasonality", "streaming", "tzclock"]
```

- [ ] **Step 3: Make the vendored packages importable and repoint `fvmodel/base.py`**

`fvmodel/_vendor/__init__.py` must put itself on `sys.path` so the vendored
`clkit` and `rvforecast` resolve as top-level names (they import each other that
way). Write `fvmodel/_vendor/__init__.py`:

```python
"""Put the vendored packages on sys.path under their own top-level names.

`clkit.state` does `from rvforecast.forward import ForwardModel`, and the fv
modules import `rvforecast.streaming` directly. Rewriting every one of those to a
relative import would be a diff against code we want to keep byte-comparable with
upstream, so the path is adjusted instead and the modules are left alone.
"""
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
```

Then in `fvmodel/base.py` replace the whole path/constants block. Find:

```python
ROOT = Path(__file__).resolve().parents[1]
VC = ROOT.parent / "btc_volatility_clock"
CACHE = ROOT / "cache"
OUT = ROOT / "output"
REPORT = ROOT / "report"
TABLES = OUT / "tables"
IMG = REPORT / "img"
for _p in (CACHE, OUT, TABLES, REPORT, IMG):
    _p.mkdir(parents=True, exist_ok=True)

PARAMS_JSON = VC / "output" / "params.json"
PERP_PARQUET = VC / "output" / "data" / "btcusdt_1s.parquet"

if str(VC) not in sys.path:
    sys.path.insert(0, str(VC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

Replace with:

```python
from . import _vendor  # noqa: F401  (puts the vendored packages on sys.path)

ROOT = Path(__file__).resolve().parents[1]          # the investigation folder
TABLES_DIR = ROOT / "tables"
CACHE = ROOT / "cache"
RUNS = ROOT / "runs"
for _p in (CACHE, RUNS):
    _p.mkdir(parents=True, exist_ok=True)

PARAMS_JSON = TABLES_DIR / "params_v2_1_0.json"

# The 1 s inputs are large and are not vendored. `fvmodel.config` resolves these
# and fails loudly when one is missing; these are only the defaults.
_SRC = Path(os.environ.get(
    "FV_SOURCE_ROOT",
    r"C:\Users\kaima\OneDrive\Documents\GitHub\autoresearch"
    r"\btc_volatility_clock_chainlink"))
DATA_ROOT = _SRC
PERP_PARQUET = _SRC.parent / "btc_volatility_clock" / "output" / "data" / "btcusdt_1s.parquet"
```

Add `import os` to the imports at the top of `fvmodel/base.py`.

- [ ] **Step 4: Repoint the remaining `CACHE` reads to `TABLES_DIR`**

In `fvmodel/build.py`, the fitted artefacts now live in `tables/`, not `cache/`.
Replace the import `from .base import CACHE, load_params` with
`from .base import TABLES_DIR, load_params`, and replace every `CACHE /` with
`TABLES_DIR /` in that file (five occurrences: `print_model.json`,
`kernels.json`, `alpha.json`, `tails.json`, and the `xi_cap.json` read if
present). Also delete the `sys.path.insert` block at the top of `build_model`
(lines beginning `import sys` through the `sys.path.insert(...)`) — `base`
already handles it — and change `from rvforecast.streaming import Model` to stay
as it is (the vendored path makes it resolve).

Verify no stale references remain:

```bash
cd "$INV" && grep -rn "CACHE /" fvmodel/ | grep -v _vendor
```
Expected: no output.

- [ ] **Step 5: Fix `clkit/common.py`'s params path**

`fvmodel/_vendor/clkit/common.py` imports `PARAMS_JSON`/`VC` and inserts `VC` on
`sys.path`. Open it and replace its path block so `PARAMS_JSON` points at
`tables/params_v2_1_0.json`:

```bash
cd "$INV" && grep -n "PARAMS_JSON\|VC\s*=\|sys.path" fvmodel/_vendor/clkit/common.py
```

Edit those lines to read from `fvmodel.base`:

```python
from fvmodel.base import PARAMS_JSON, load_params  # noqa: F401
```

and delete the local `VC` / `sys.path` manipulation. Leave `load_aligned` alone.

- [ ] **Step 6: Write `pytest.ini` and `tests/conftest.py`**

`pytest.ini`:

```ini
[pytest]
testpaths = tests
markers =
    slow: needs the 1 s Chainlink/perp/spot inputs present
```

`tests/conftest.py`:

```python
import sys
from pathlib import Path

INV = Path(__file__).resolve().parents[1]
if str(INV) not in sys.path:
    sys.path.insert(0, str(INV))
```

- [ ] **Step 7: Write the failing golden test**

Create `tests/test_golden.py`:

```python
"""The consolidation must not change a single number.

Task 1 froze the pre-consolidation model's answers on a fixed set of quotes. This
prices the same quotes through the vendored, consolidated model and requires exact
equality - not approximate. Anything else means the consolidation changed the model,
which is the one thing it is not allowed to do.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow

GOLDEN = "tests/golden/fv_golden.npz"
FIELDS = ("p_up", "var_y", "y_star", "m_Y", "eps_bar", "var_eps", "var_basis",
          "carry", "omega", "known_value", "z", "p_ref")


@pytest.fixture(scope="module")
def replay():
    from pathlib import Path

    from fvmodel.base import ROOT, load_params
    from fvmodel.build import build_model
    from fvmodel.engine import Window, market_at, state_at
    from fvmodel.fairvalue import fair_value

    g = np.load(Path(ROOT) / GOLDEN)
    model = build_model()
    win = Window(load_params(), model.fp, int(g["window_t0"]), int(g["window_t1"]),
                 warm=0, chainlink=True, log=lambda *a: None)
    its = np.unique(g["it"])
    win.prepare(np.concatenate([its, its - 2]))

    got = {k: np.empty(g["p_up"].size) for k in FIELDS}
    got.update({k: np.empty(g["p_up"].size) for k in ("nu", "mu", "sigma_t")})
    states = {}
    for r in range(g["p_up"].size):
        it = int(g["it"][r])
        if it not in states:
            st = state_at(win, it, model)
            st.book = {"imbalance": 0.8, "px_age_s": 1.4}
            states[it] = st
        st = states[it]
        t = int(win.ts[it])
        n = int(g["n"][r])
        kind = str(g["kind"][r])
        L = 300 if kind == "chainlink_twap60" and n <= 300 else 900
        mk = market_at(win, kind, t + n, n, L, 60)
        fv = fair_value(st, mk, model, t_now=t)
        for k in FIELDS:
            got[k][r] = float(getattr(fv, k))
        got["nu"][r], got["mu"][r], got["sigma_t"][r] = (float(x) for x in fv.tail)
    return g, got


@pytest.mark.parametrize("field", FIELDS + ("nu", "mu", "sigma_t"))
def test_bit_identical_to_pre_consolidation(replay, field):
    g, got = replay
    want = g[field]
    bad = ~(got[field] == want)
    assert not bad.any(), (
        "%s differs on %d of %d rows; worst |delta| = %.3g at row %d "
        "(golden %.17g, got %.17g)"
        % (field, bad.sum(), want.size, np.max(np.abs(got[field] - want)),
           int(np.argmax(np.abs(got[field] - want))),
           want[np.argmax(np.abs(got[field] - want))],
           got[field][np.argmax(np.abs(got[field] - want))]))
```

Note: `fair_value` is called **without** a `Switches` argument. That call
signature does not exist yet — it arrives in Task 6. Until then this test fails
on a `TypeError`, which is the correct failing state.

Note: `fair_value`'s signature at this point is already
`fair_value(state, market, model, sw=None, t_now=None)` with `sw` defaulting to
`None`, so the keyword call `fair_value(st, mk, model, t_now=t)` works unchanged.
Task 6 removes the `sw` parameter entirely.

- [ ] **Step 8: Run the golden test**

Run: `cd "$INV" && python -m pytest tests/test_golden.py -v`
Expected: all 15 parametrised cases PASS. A failure here means a copy or path edit
changed behaviour — fix it before going on; do **not** relax the equality.

- [ ] **Step 9: Add a `.gitignore` and commit**

Create `.gitignore` in `$INV`:

```
cache/
runs/
__pycache__/
*.pyc
.pytest_cache/
```

```bash
cd /c/Users/kaima/Github2/Gambling104
git add investigations/2026-9-9_business_clock_t_dist
git commit -m "feat: vendor the fair-value model, its tables and the v2.1 forecaster subset"
```

---

## Task 3: `model.json`, `config.py`, `CHANGELOG.md`, `filter_by_w.json`

**Files:**
- Create: `model.json`, `CHANGELOG.md`, `fvmodel/config.py`, `tools/extract_filter_by_w.py`, `tables/filter_by_w.json`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `fvmodel.config.load(variant: str | None = None) -> Loaded`
  - `Loaded` — a dataclass with `.model: FairValueModel`, `.overrides: Overrides`, `.provenance: dict`, `.cfg: dict`. In Task 3 `.overrides` is `None`; Task 4 fills it in.
  - `fvmodel.config.CONFIG_PATH`, `fvmodel.config.sha256_of(path) -> str`
  - `tables/filter_by_w.json` — `{"rows": [[w, tau_s, delta_s, p_stamp_late, rmse_bp], ...]}`, 13 rows ascending in `w`.

- [ ] **Step 1: Write the per-`w` extraction tool**

Create `tools/extract_filter_by_w.py`:

```python
"""Reduce the filter grid-search surface to one best row per blend weight.

`fit_filter` searches (w, tau, delta, p_late) and `print_model.json` keeps only the
winner, but the `w_spot` override needs the constants at every w. The full surface
survives in the source repo's cache/print_model.npz as `surface_train` (17,238 rows
of w, tau, delta, p_late, rmse). This takes the argmin per w and writes the 13-row
table, which is small enough to commit and read.

    python tools/extract_filter_by_w.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get(
    "FV_SOURCE_ROOT",
    r"C:\Users\kaima\OneDrive\Documents\GitHub\autoresearch"
    r"\btc_volatility_clock_chainlink"))
OUT = Path(__file__).resolve().parents[1] / "tables" / "filter_by_w.json"


def main() -> None:
    s = np.load(SRC / "cache" / "print_model.npz")["surface_train"].astype(np.float64)
    rows = []
    for w in np.unique(s[:, 0]):
        r = s[s[:, 0] == w]
        b = r[int(np.argmin(r[:, 4]))]
        rows.append([round(float(b[0]), 4), float(b[1]), float(b[2]),
                     float(b[3]), float(b[4])])
    rows.sort(key=lambda x: x[0])
    OUT.write_text(json.dumps({
        "source": "cache/print_model.npz:surface_train (coarse grid), argmin rmse per w",
        "cols": ["w_spot", "tau_s", "delta_s", "p_stamp_late", "rmse_bp"],
        "note": ("This is the COARSE grid. At the shipped w = 0.6 it gives "
                 "tau = 0.7872, not the fine-refined shipped tau = 0.8447, so "
                 "w_spot: 0.6 is NOT the baseline model. See spec ruling R11."),
        "rows": rows}, indent=2) + "\n")
    print("wrote %s: %d rows, w from %.2f to %.2f"
          % (OUT, len(rows), rows[0][0], rows[-1][0]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd "$INV" && python tools/extract_filter_by_w.py`
Expected: `wrote .../tables/filter_by_w.json: 13 rows, w from 0.00 to 1.00`

- [ ] **Step 3: Write the failing config test**

Create `tests/test_config.py`:

```python
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
```

- [ ] **Step 4: Run it to verify it fails**

Run: `cd "$INV" && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fvmodel.config'`

- [ ] **Step 5: Generate `model.json`**

Create `tools/write_model_json.py`:

```python
"""Generate model.json with the current table hashes. Re-run after changing a table,
and bump `version` and CHANGELOG.md when you do.

    python tools/write_model_json.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = {
    "print_model": ("scripts/10_print_model.py", "report_final.html section 4",
                    "filter constants, basis tracker, eps model, reconstruction"),
    "kernels": ("scripts/11_kernels.py", "report_final.html section 6",
                "activity-conditioned return autocorrelation, input variance ratio"),
    "alpha": ("scripts/12_alpha.py", "report_final.html section 10",
              "beta(dT) in bp with the quote-age interaction"),
    "tails": ("scripts/13_eval.py", "report_final.html section 11",
              "settlement tail (nu, mu, sigma) by log business time, per kind"),
    "xi_cap": ("scripts/22_fit_xi_cap.py", "report_final.html section 11.2",
               "short-business-time forward-curve cap; OFF by default"),
    "filter_by_w": ("tools/extract_filter_by_w.py", "spec ruling R11",
                    "best (tau, delta, p_late) per blend weight, coarse grid"),
}
DEFAULT_OVERRIDES = {
    "version": "ov-1.0.0",
    "source": "consolidation brief 2026-09-09",
    "kappa_vol": 0.0, "kappa_vol_short": 0.0,
    "shrink_w": 0.0, "shrink_decay_s": 60.0,
    "rho_kernel": "conditional",
    "eps_scale": 1.0, "eps_condition": True,
    "basis_tracker": "main", "information_set": "full",
    "reconstruct_in_transit": True,
    "kappa_tail": 0.0, "tail_scale": 1.0, "tail_family": "t",
    "alpha_scale": 1.0, "alpha_scale_age": 1.0, "alpha_cap_sd": None,
    "w_spot": None, "xi_cap_c": None, "temperature": 1.0,
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    params = ROOT / "tables" / "params_v2_1_0.json"
    cfg = {
        "version": "fv-1.0.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "tools/write_model_json.py",
        "source": ("btc_volatility_clock_chainlink, report_final.html; "
                   "vendored 2026-09-09, not refitted here"),
        "window": {"backtest_t0": 1786665600, "backtest_t1": 1788220799,
                   "select_t1": 1787702399, "holdout_t0": 1787702400,
                   "burn_days": 14,
                   "note": ("2026-08-14 .. 2026-08-31. The panel runs to 09-08 but "
                            "the perp/spot 1 s inputs end 08-31, so 09-01 onward is "
                            "unpriceable. See spec section 1.1.")},
        "base": {"params": "tables/params_v2_1_0.json", "params_version": "2.1.0",
                 "params_sha256": sha(params)},
        "tables": {
            name: {"file": "tables/%s.json" % name,
                   "sha256": sha(ROOT / "tables" / ("%s.json" % name)),
                   "fitted_by": by, "report": rep, "contents": what}
            for name, (by, rep, what) in TABLES.items()},
        "defaults": {"eps_fit": "train", "input_mode": "blend"},
        "overrides": DEFAULT_OVERRIDES,
        "data": {
            "note": ("Large 1 s inputs are not vendored. Override the root with the "
                     "FV_SOURCE_ROOT environment variable."),
            "source_root": (r"C:\Users\kaima\OneDrive\Documents\GitHub\autoresearch"
                            r"\btc_volatility_clock_chainlink"),
            "chainlink_1s": "data/chainlink/chainlink_1s.parquet",
            "perp_1s": "../btc_volatility_clock/output/data/btcusdt_1s.parquet",
            "spot_1s_glob": "cache/spot_1s_*.parquet"},
    }
    (ROOT / "model.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print("wrote model.json version %s with %d tables"
          % (cfg["version"], len(cfg["tables"])))


if __name__ == "__main__":
    main()
```

Run: `cd "$INV" && python tools/write_model_json.py`
Expected: `wrote model.json version fv-1.0.0 with 6 tables`

- [ ] **Step 6: Write `fvmodel/config.py`**

```python
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
```

- [ ] **Step 7: Run the config test**

Run: `cd "$INV" && python -m pytest tests/test_config.py -v`
Expected: 4 PASS.

- [ ] **Step 8: Write `CHANGELOG.md`**

Create `CHANGELOG.md`:

```markdown
# Changelog

## fv-1.0.0 — 2026-09-09

First frozen version. The model is `btc_volatility_clock_chainlink` as of
2026-09-09, vendored unchanged. Nothing is refitted here.

### Base

| | |
|---|---|
| forecaster | `btc_volatility_clock` v2.1.0 (`tables/params_v2_1_0.json`) |
| fit protocol | fitted through 2026-07-19; holdout 2026-07-20 .. 2026-08-31 |

### Fitted tables

| table | fitted by | window | report | what it carries |
|---|---|---|---|---|
| `print_model.json` | `scripts/10_print_model.py` | 2026-04-13 .. 2026-07-19, 07-07 masked | §4 | filter (w=0.6, tau=0.8447, delta=0.7, p_late=0.02), basis half-life 60 s, eps model, reconstruction |
| `kernels.json` | `scripts/11_kernels.py` | as above | §6 | activity-conditioned return autocorrelation, activity tercile cuts, input variance ratio |
| `alpha.json` | `scripts/12_alpha.py` | five book captures to 2026-04-02 | §10 | beta(dT) in bp per unit imbalance, quote-age interaction |
| `tails.json` | `scripts/13_eval.py` | as print_model | §11 | settlement tail (nu, mu, sigma) by log business time, per kind |
| `xi_cap.json` | `scripts/22_fit_xi_cap.py` | as print_model | §11.2 | short-business-time forward-curve cap. **Off by default** |
| `filter_by_w.json` | `tools/extract_filter_by_w.py` | as print_model | spec R11 | best (tau, delta, p_late) per blend weight, **coarse grid** |

### Known limitation

The order-book captures do not overlap the Chainlink history — they end
2026-04-02, Chainlink starts 2026-04-13. The location term is measured on the
perp and applied to a Chainlink settlement on the strength of the two being the
same price to within a basis point.

## ov-1.0.0 — 2026-09-09

The overrides block. Every key defaults to its off value; with all defaults the
model is bit-identical to fv-1.0.0, which `tests/test_golden.py` enforces.
```

- [ ] **Step 9: Commit**

```bash
cd /c/Users/kaima/Github2/Gambling104
git add investigations/2026-9-9_business_clock_t_dist
git commit -m "feat: consolidated model.json with verified provenance, and the CHANGELOG"
```

---

## Task 4: `Overrides` and the build-time layer

**Files:**
- Create: `fvmodel/overrides.py`, `tests/test_overrides_buildtime.py`
- Modify: `fvmodel/config.py` (fill in `Loaded.overrides`)

**Interfaces:**
- Produces:
  - `Overrides` — frozen dataclass, fields exactly the keys in `model.json:overrides` minus `version`/`source`.
  - `Overrides.from_dict(d: dict) -> Overrides` — rejects unknown keys and bad enum values.
  - `Overrides.non_default() -> dict[str, object]`
  - `Overrides.label() -> str` — `"baseline"` or `"kappa_vol=0.095+rho_kernel=off"`.
  - `apply_overrides(model, ov) -> FairValueModel` — returns a **new** model; never mutates.
- Consumes: `fvmodel.config.filter_by_w()`, `fvmodel.build.build_model`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_overrides_buildtime.py`:

```python
"""What apply_overrides folds into the model object at build time."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def base():
    from fvmodel.build import build_model
    return build_model()


def _ov(**kw):
    from fvmodel.overrides import Overrides
    return Overrides(**kw)


def test_defaults_return_an_equivalent_model(base):
    from fvmodel.overrides import apply_overrides

    m = apply_overrides(base, _ov())
    assert m.fp.tau_s == base.fp.tau_s and m.fp.w_spot == base.fp.w_spot
    assert m.eps.sigma_bp == base.eps.sigma_bp
    for kind, t in base.tails.items():
        assert np.array_equal(m.tails[kind].nu, t.nu)
        assert np.array_equal(m.tails[kind].sigma, t.sigma)
        assert np.array_equal(m.tails[kind].mu, t.mu)


def test_unknown_key_is_rejected():
    from fvmodel.overrides import Overrides
    with pytest.raises(ValueError, match="unknown override"):
        Overrides.from_dict({"kappa_volume": 1.0})


def test_bad_enum_is_rejected():
    from fvmodel.overrides import Overrides
    with pytest.raises(ValueError, match="rho_kernel"):
        Overrides.from_dict({"rho_kernel": "sideways"})


def test_label_lists_only_non_defaults():
    assert _ov().label() == "baseline"
    assert _ov(kappa_vol=0.095, rho_kernel="off").label() == "kappa_vol=0.095+rho_kernel=off"


def test_kappa_tail_fattens_nu_and_holds_the_variance(base):
    """nu' = 2 + (nu-2)exp(k) with sigma rescaled so nu/(nu-2) * sigma^2 is unchanged."""
    from fvmodel.overrides import apply_overrides

    m = apply_overrides(base, _ov(kappa_tail=-0.5))
    t0 = base.tails["chainlink_twap60"]
    t1 = m.tails["chainlink_twap60"]
    assert np.all(t1.nu < t0.nu), "negative kappa_tail must fatten the tail"
    v0 = t0.nu / (t0.nu - 2.0) * t0.sigma ** 2
    v1 = t1.nu / (t1.nu - 2.0) * t1.sigma ** 2
    assert np.allclose(v0, v1, rtol=1e-12), "standardised variance must not move"


def test_tail_scale_scales_mu_with_sigma(base):
    """Spec ruling R2: scaling sigma alone breaks the kappa_vol orthogonality."""
    from fvmodel.overrides import apply_overrides

    c = 0.3
    m = apply_overrides(base, _ov(tail_scale=np.exp(c)))
    t0 = base.tails["chainlink_twap60"]
    t1 = m.tails["chainlink_twap60"]
    assert np.allclose(t1.sigma, t0.sigma * np.exp(c), rtol=1e-14)
    assert np.allclose(t1.mu, t0.mu * np.exp(c), rtol=1e-14)
    assert np.array_equal(t1.nu, t0.nu)


def test_tail_family_normal(base):
    from fvmodel.overrides import apply_overrides
    m = apply_overrides(base, _ov(tail_family="normal"))
    assert m.tails["chainlink_twap60"].family == "normal"


def test_alpha_scales(base):
    from fvmodel.overrides import apply_overrides

    m = apply_overrides(base, _ov(alpha_scale=2.0))
    assert np.allclose(m.alpha.beta0_bp, base.alpha.beta0_bp * 2.0)
    assert np.allclose(m.alpha.beta1_bp, base.alpha.beta1_bp * 2.0)
    m0 = apply_overrides(base, _ov(alpha_scale=0.0))
    assert np.all(m0.alpha.beta0_bp == 0.0) and np.all(m0.alpha.beta1_bp == 0.0)
    ma = apply_overrides(base, _ov(alpha_scale_age=0.0))
    assert np.allclose(ma.alpha.beta0_bp, base.alpha.beta0_bp)
    assert np.all(ma.alpha.beta1_bp == 0.0), "alpha_scale_age must touch beta1 only"


def test_eps_scale_scales_the_process(base):
    """Spec ruling R1: eps_scale=0 must remove the term, mean as well as variance."""
    from fvmodel.overrides import apply_overrides

    m = apply_overrides(base, _ov(eps_scale=2.0))
    assert m.eps.sigma_bp == pytest.approx(base.eps.sigma_bp * 2.0)
    m0 = apply_overrides(base, _ov(eps_scale=0.0))
    assert m0.eps.sigma_bp == 0.0


def test_w_spot_reads_the_per_w_table(base):
    from fvmodel.overrides import apply_overrides

    m = apply_overrides(base, _ov(w_spot=0.35))
    assert m.fp.w_spot == pytest.approx(0.35)
    assert m.fp.tau_s == pytest.approx(0.9865)
    assert m.fp.delta_s == pytest.approx(0.70)


def test_w_spot_default_keeps_the_shipped_fine_fit(base):
    """R11: w_spot=None is the shipped filter, not the coarse row at w=0.6."""
    from fvmodel.overrides import apply_overrides

    m = apply_overrides(base, _ov())
    assert m.fp.tau_s == pytest.approx(0.8447)
    m6 = apply_overrides(base, _ov(w_spot=0.6))
    assert m6.fp.tau_s == pytest.approx(0.7872)


def test_w_spot_off_grid_is_rejected():
    from fvmodel.overrides import Overrides
    with pytest.raises(ValueError, match="w_spot"):
        Overrides.from_dict({"w_spot": 0.42})


def test_rho_kernel_off_gives_a_delta_kernel(base):
    from fvmodel.overrides import apply_overrides
    m = apply_overrides(base, _ov(rho_kernel="off"))
    r = m.rho_for(1.0)
    assert r[0] == 1.0 and np.all(r[1:] == 0.0)


def test_xi_cap_selects_the_fitted_register(base):
    from fvmodel.overrides import apply_overrides
    m = apply_overrides(base, _ov(xi_cap_c=2.0))
    assert m.ov.xi_cap_c == 2.0 and m.xi_cap_i is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "$INV" && python -m pytest tests/test_overrides_buildtime.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fvmodel.overrides'`

- [ ] **Step 3: Write `fvmodel/overrides.py` (build-time half)**

```python
"""Every knob the model exposes, and the one place their semantics live.

Two layers, because six application points cannot honestly be one call site.

`apply_overrides` is the build-time layer: everything statically resolvable is
folded into a new model object. The per-tick transforms below it - `xi_adjust`,
`cap_m_Y`, `quoted_prob` - are pure functions called by BOTH the scalar and the
vectorised path, which is what stops those two implementations drifting apart.

Nothing here mutates the model it is given.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields

import numpy as np

RHO_KERNELS = ("conditional", "unconditional", "off")
BASIS_TRACKERS = ("main", "alt", "off")
INFORMATION_SETS = ("full", "prints_only")
TAIL_FAMILIES = ("t", "normal")
NU_FLOOR = 2.5


@dataclass(frozen=True)
class Overrides:
    """All defaults are the off value. See variants/README.md for sign conventions."""
    kappa_vol: float = 0.0
    kappa_vol_short: float = 0.0
    shrink_w: float = 0.0
    shrink_decay_s: float = 60.0
    rho_kernel: str = "conditional"
    eps_scale: float = 1.0
    eps_condition: bool = True
    basis_tracker: str = "main"
    information_set: str = "full"
    reconstruct_in_transit: bool = True
    kappa_tail: float = 0.0
    tail_scale: float = 1.0
    tail_family: str = "t"
    alpha_scale: float = 1.0
    alpha_scale_age: float = 1.0
    alpha_cap_sd: float | None = None
    w_spot: float | None = None
    xi_cap_c: float | None = None
    temperature: float = 1.0

    def __post_init__(self):
        for name, allowed in (("rho_kernel", RHO_KERNELS),
                              ("basis_tracker", BASIS_TRACKERS),
                              ("information_set", INFORMATION_SETS),
                              ("tail_family", TAIL_FAMILIES)):
            v = getattr(self, name)
            if v not in allowed:
                raise ValueError("%s must be one of %s, got %r" % (name, allowed, v))
        if self.temperature <= 0:
            raise ValueError("temperature must be positive, got %r" % self.temperature)
        if self.eps_scale < 0 or self.tail_scale <= 0:
            raise ValueError("eps_scale must be >= 0 and tail_scale > 0")
        if self.w_spot is not None:
            from .config import filter_by_w
            grid = [r[0] for r in filter_by_w()]
            if not any(abs(self.w_spot - g) < 1e-9 for g in grid):
                raise ValueError(
                    "w_spot must be a value in tables/filter_by_w.json %s, got %r; "
                    "the filter is re-read from that table, never refitted"
                    % (grid, self.w_spot))

    # ---- io ----------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Overrides":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known - {"version", "source"}
        if unknown:
            raise ValueError("unknown override key(s): %s; known keys are %s"
                             % (sorted(unknown), sorted(known)))
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    def non_default(self) -> dict:
        d = Overrides()
        return {f.name: getattr(self, f.name) for f in fields(self)
                if getattr(self, f.name) != getattr(d, f.name)}

    def label(self) -> str:
        nd = self.non_default()
        return "baseline" if not nd else "+".join(
            "%s=%s" % (k, v) for k, v in sorted(nd.items()))


# ==================================================================== build time
def _scaled_tail(tail, ov: "Overrides"):
    """kappa_tail, tail_scale and tail_family, applied to a stored (nu, mu, sigma).

    kappa_tail moves the shape and holds Var(z) = nu/(nu-2) * sigma^2 fixed, so it is
    a pure tail-thickness knob and does not smuggle in a width change. tail_scale is
    the width knob, and scales mu with sigma (spec ruling R2) so that it rescales the
    standardised variable as a whole - which is what makes it exactly equivalent to
    kappa_vol at the other end of the pipeline.
    """
    t = copy.deepcopy(tail)
    if ov.tail_family == "normal":
        t.family = "normal"
        return t
    if ov.kappa_tail != 0.0:
        nu0 = np.asarray(t.nu, dtype=np.float64)
        nu1 = np.maximum(2.0 + (nu0 - 2.0) * np.exp(ov.kappa_tail), NU_FLOOR)
        t.sigma = t.sigma * np.sqrt((nu0 / (nu0 - 2.0)) / (nu1 / (nu1 - 2.0)))
        t.nu = nu1
    if ov.tail_scale != 1.0:
        t.sigma = t.sigma * ov.tail_scale
        t.mu = t.mu * ov.tail_scale
    return t


def apply_overrides(model, ov: "Overrides"):
    """A new model with every statically resolvable override folded in."""
    from .chainlink import FilterParams
    from .curve import slow_index

    m = copy.copy(model)                       # shallow: v2 and rho are read-only
    m.ov = ov

    # ---- the input blend, and the filter that goes with it -------------------
    if ov.w_spot is not None:
        from .config import filter_by_w
        row = min(filter_by_w(), key=lambda r: abs(r[0] - ov.w_spot))
        fp = FilterParams.from_dict(dict(model.fp.to_dict(),
                                         w_spot=row[0], tau_s=row[1],
                                         delta_s=row[2], p_stamp_late=row[3]))
        fp.basis_hl_s = model.fp.basis_hl_s
        fp.basis_hl_alt_s = model.fp.basis_hl_alt_s
        fp.lag_s = model.fp.lag_s
        m.fp = fp
    else:
        m.fp = copy.copy(model.fp)

    # ---- the eps process (R1: scale the process, not only its sd) ------------
    m.eps = copy.deepcopy(model.eps)
    if ov.eps_scale != 1.0:
        m.eps.sigma_bp = model.eps.sigma_bp * ov.eps_scale

    # ---- the location term ---------------------------------------------------
    m.alpha = copy.deepcopy(model.alpha)
    m.alpha.beta0_bp = model.alpha.beta0_bp * ov.alpha_scale
    m.alpha.beta1_bp = model.alpha.beta1_bp * ov.alpha_scale * ov.alpha_scale_age

    # ---- the tails -----------------------------------------------------------
    m.tails = {k: _scaled_tail(v, ov) for k, v in model.tails.items()}

    # ---- the forward-curve cap ----------------------------------------------
    if ov.xi_cap_c is not None:
        m.xi_cap_c = float(ov.xi_cap_c)
        m.xi_cap_i = slow_index(np.asarray(
            model.v2.params["registers"]["half_life_business"], dtype=np.float64))
    else:
        m.xi_cap_c = 0.0
        m.xi_cap_i = None
    return m
```

- [ ] **Step 4: Add `ov` to `FairValueModel` and route `rho_for` / `tail_for` through it**

In `fvmodel/fairvalue.py`, in the `FairValueModel` dataclass: delete the
`kappa_vol: float = 0.0` field and add

```python
    ov: object = None                            # fvmodel.overrides.Overrides
```

Change `rho_for` and `tail_for` to read the mode off `self.ov` instead of taking
a `mode` argument, keeping the argument as an optional override for tests:

```python
    def rho_for(self, act: float, mode: str = None) -> np.ndarray:
        mode = mode or (self.ov.rho_kernel if self.ov else "conditional")
        if mode == "off":
            z = np.zeros(2)
            z[0] = 1.0
            return z
        if mode != "conditional" or "act0" not in self.rho:
            return self.rho.get("all", np.array([1.0]))
        k = 0 if act < self.act_cuts[0] else (1 if act < self.act_cuts[1] else 2)
        return self.rho["act%d" % k]

    def tail_for(self, kind: str) -> SettlementTail:
        return self.tails.get(kind) or SettlementTail.normal(kind)
```

`SettlementTail.normal` and the `"normal"` family are now reached through
`_scaled_tail`, so `tail_for` no longer needs a mode.

- [ ] **Step 5: Give `build_model` a default `ov`**

At the end of `fvmodel/build.py`'s `build_model`, replace the return with:

```python
    from .overrides import Overrides, apply_overrides

    m = FairValueModel(Model(params), fp, eps, alpha, rho, cuts, tl, ivr)
    m.v2.params = params          # apply_overrides needs the register half-lives
    return apply_overrides(m, Overrides())
```

and remove `0.0` (the old `kappa_vol` positional) from the constructor call.

- [ ] **Step 6: Run the build-time tests**

Run: `cd "$INV" && python -m pytest tests/test_overrides_buildtime.py -v`
Expected: 13 PASS.

- [ ] **Step 7: Re-run the golden test**

Run: `cd "$INV" && python -m pytest tests/test_golden.py tests/test_config.py -v`
Expected: all PASS. If the golden test now fails, `apply_overrides` is not a
no-op at defaults — fix that, do not touch the golden file.

- [ ] **Step 8: Wire overrides into `config.load` and commit**

In `fvmodel/config.py`, change `load` to build and apply the overrides:

```python
def load(variant: str | None = None) -> Loaded:
    from .build import build_model
    from .overrides import Overrides, apply_overrides

    cfg = read_config()
    verify(cfg)
    d = {k: v for k, v in cfg["overrides"].items() if k not in ("version", "source")}
    notes = ""
    if variant and variant != "baseline":
        from .variants import read_variant      # Task 9; imported lazily so that
        v = read_variant(variant)               # config.load works before it exists
        d.update(v["overrides"])
        notes = v.get("notes", "")
    ov = Overrides.from_dict(d)
    model = apply_overrides(build_model(), ov)
    prov = {"model_version": cfg["version"],
            "overrides_version": cfg["overrides"]["version"],
            "params_version": cfg["base"]["params_version"],
            "variant": variant or "baseline",
            "overrides_non_default": ov.non_default(),
            "label": ov.label(), "notes": notes,
            "tables": {k: v["sha256"][:12] for k, v in cfg["tables"].items()}}
    return Loaded(model=model, overrides=ov, provenance=prov, cfg=cfg)
```

```bash
cd /c/Users/kaima/Github2/Gambling104
git add investigations/2026-9-9_business_clock_t_dist
git commit -m "feat: Overrides dataclass and the build-time apply_overrides layer"
```

---

## Task 5: The per-tick transforms and the `kappa_vol` relocation

**Files:**
- Modify: `fvmodel/overrides.py`, `fvmodel/curve.py`, `fvmodel/fairvalue.py`, `fvmodel/engine.py`
- Create: `tests/test_overrides_pertick.py`

**Interfaces:**
- Produces:
  - `xi_adjust(ov, xi, dT_business_s, xi_bar=None) -> np.ndarray` — `xi` shaped `(..., m)`, `dT_business_s` the business age in seconds of each of those m seconds, broadcastable to `xi`. Returns a new array; returns `xi` unchanged (same object) when the three vol overrides are all default.
  - `cap_m_Y(ov, m_Y, var_y) -> np.ndarray | float`
  - `quoted_prob(ov, p) -> np.ndarray | float`
  - `unconditional_xi(model, dT) -> np.ndarray` in `fvmodel/curve.py` — the forward curve from `registers.initial_value`, the shrink target (spec R6).
  - `FairValue` gains `p_model: float` and `p_quoted: float`; `p_up` stays as an alias of `p_model` so nothing downstream breaks.

- [ ] **Step 1: Write the failing per-tick tests**

Create `tests/test_overrides_pertick.py`:

```python
"""The per-tick transforms, in isolation and then through the whole pipeline."""
from __future__ import annotations

import numpy as np
import pytest


def _ov(**kw):
    from fvmodel.overrides import Overrides
    return Overrides(**kw)


# ------------------------------------------------------------------ xi_adjust
def test_xi_adjust_is_identity_at_defaults():
    from fvmodel.overrides import xi_adjust

    xi = np.array([[1e-9, 2e-9, 3e-9]])
    d = np.array([[10.0, 20.0, 30.0]])
    assert xi_adjust(_ov(), xi, d) is xi


def test_kappa_vol_scales_every_second_equally():
    from fvmodel.overrides import xi_adjust

    xi = np.array([[1e-9, 2e-9, 3e-9]])
    d = np.array([[1.0, 30.0, 3000.0]])
    out = xi_adjust(_ov(kappa_vol=0.5), xi, d)
    assert np.allclose(out, xi * np.exp(1.0), rtol=1e-14)


def test_kappa_vol_short_only_bites_below_a_minute():
    from fvmodel.overrides import xi_adjust

    xi = np.ones((1, 3)) * 1e-9
    d = np.array([[15.0, 60.0, 600.0]])
    out = xi_adjust(_ov(kappa_vol_short=-0.22), xi, d)
    # weight = min(1, 60/D): 1.0 at 15 s, 1.0 at 60 s, 0.1 at 600 s
    want = xi * np.exp(2.0 * -0.22 * np.array([[1.0, 1.0, 0.1]]))
    assert np.allclose(out, want, rtol=1e-14)


def test_shrink_pulls_toward_the_unconditional_level_at_short_age():
    from fvmodel.overrides import xi_adjust

    xi = np.array([[4e-9, 4e-9]])
    bar = np.array([[1e-9, 1e-9]])
    d = np.array([[0.0, 1e6]])          # w = 0.5 at age 0, ~0 at age 1e6
    out = xi_adjust(_ov(shrink_w=0.5, shrink_decay_s=60.0), xi, d, bar)
    assert out[0, 0] == pytest.approx(np.exp(0.5 * np.log(4e-9) + 0.5 * np.log(1e-9)))
    assert out[0, 1] == pytest.approx(4e-9, rel=1e-12)


def test_shrink_is_applied_after_kappa_vol():
    """kappa_vol moves xi; the shrink then pulls the MOVED xi toward xi_bar."""
    from fvmodel.overrides import xi_adjust

    xi = np.array([[4e-9]])
    bar = np.array([[1e-9]])
    d = np.array([[0.0]])
    out = xi_adjust(_ov(kappa_vol=0.5, shrink_w=0.5, shrink_decay_s=60.0), xi, d, bar)
    moved = 4e-9 * np.exp(1.0)
    assert out[0, 0] == pytest.approx(np.exp(0.5 * np.log(moved) + 0.5 * np.log(1e-9)))


# ---------------------------------------------------------------- cap_m_Y
def test_alpha_cap_clips_at_a_multiple_of_the_settlement_sd():
    from fvmodel.overrides import cap_m_Y

    var = np.array([1e-8])                       # sd = 1e-4
    assert cap_m_Y(_ov(), np.array([5e-4]), var)[0] == pytest.approx(5e-4)
    assert cap_m_Y(_ov(alpha_cap_sd=2.0), np.array([5e-4]), var)[0] == pytest.approx(2e-4)
    assert cap_m_Y(_ov(alpha_cap_sd=2.0), np.array([-5e-4]), var)[0] == pytest.approx(-2e-4)
    assert cap_m_Y(_ov(alpha_cap_sd=2.0), np.array([1e-5]), var)[0] == pytest.approx(1e-5)


# -------------------------------------------------------------- quoted_prob
def test_temperature_is_a_logit_rescale_and_fixes_a_half():
    from fvmodel.overrides import quoted_prob

    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(quoted_prob(_ov(), p), p)
    q = quoted_prob(_ov(temperature=2.0), p)
    assert q[1] == pytest.approx(0.5)
    assert q[0] > p[0] and q[2] < p[2], "T > 1 must pull toward a half"
    lg = np.log(p / (1 - p)) / 2.0
    assert np.allclose(q, 1.0 / (1.0 + np.exp(-lg)))


# ------------------------------------------------- through the whole pipeline
@pytest.mark.slow
@pytest.fixture(scope="module")
def priced():
    """One quote, priced under a series of overrides. perp_single: no Jensen term
    and no eps, which is what the orthogonality assertion needs (spec R3)."""
    import numpy as np

    from fvmodel.base import CL_T0, load_params
    from fvmodel.build import build_model
    from fvmodel.engine import Window, market_at, state_at
    from fvmodel.fairvalue import fair_value
    from fvmodel.overrides import apply_overrides

    t0 = CL_T0 + 20 * 86400
    base = build_model()
    win = Window(load_params(), base.fp, t0, t0 + 2 * 86400, warm=0, chainlink=True,
                 log=lambda *a: None)
    it = win.n - 20_000
    win.prepare(np.array([it, it - 2]))
    st = state_at(win, it, base)
    t = int(win.ts[it])

    def price(kind, n, **kw):
        m = apply_overrides(base, _ov(**kw))
        mk = market_at(win, kind, t + n, n, 900, 60)
        return fair_value(st, mk, m, t_now=t)

    return price


@pytest.mark.slow
def test_kappa_vol_and_tail_scale_are_the_same_knob(priced):
    """Spec R2/R3: kappa_vol = c and tail_scale = exp(c) must agree on p_model."""
    c = 0.2
    a = priced("perp_single", 300, kappa_vol=c)
    b = priced("perp_single", 300, tail_scale=float(np.exp(c)))
    assert a.p_model == pytest.approx(b.p_model, rel=1e-10), (
        "kappa_vol=%g gives %.12g, tail_scale=exp(%g) gives %.12g"
        % (c, a.p_model, c, b.p_model))


@pytest.mark.slow
def test_temperature_leaves_p_model_alone(priced):
    a = priced("chainlink_twap60", 120)
    b = priced("chainlink_twap60", 120, temperature=1.5)
    assert b.p_model == a.p_model
    lg = np.log(a.p_model / (1 - a.p_model)) / 1.5
    assert b.p_quoted == pytest.approx(1.0 / (1.0 + np.exp(-lg)))
    assert a.p_quoted == a.p_model


@pytest.mark.slow
def test_kappa_vol_raises_the_variance(priced):
    a = priced("chainlink_twap60", 600)
    b = priced("chainlink_twap60", 600, kappa_vol=0.25)
    assert b.var_y > a.var_y * 1.2
    assert b.omega == a.omega and b.known_value == pytest.approx(a.known_value)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "$INV" && python -m pytest tests/test_overrides_pertick.py -v -m "not slow"`
Expected: FAIL — `ImportError: cannot import name 'xi_adjust'`

- [ ] **Step 3: Add the per-tick transforms to `fvmodel/overrides.py`**

Append:

```python
# ===================================================================== per tick
def xi_adjust(ov: "Overrides", xi, dT_business_s, xi_bar=None):
    """The vol overrides, applied to the forward curve.

    `xi` is the per-second forward variance over the exact block, `dT_business_s` the
    business age in seconds of each of those seconds measured from the quote origin
    (spec R7: business days times 86400), broadcastable against `xi`.

    Order matters and is the brief's: kappa_vol and kappa_vol_short move log xi, then
    the shrink pulls the MOVED curve toward the seasonal-unconditional level `xi_bar`
    (spec R6). Applying the shrink first would let a vol override escape it.

    Returns `xi` itself when nothing is on, so the default path allocates nothing and
    is bit-identical.
    """
    if ov.kappa_vol == 0.0 and ov.kappa_vol_short == 0.0 and ov.shrink_w == 0.0:
        return xi
    logxi = np.log(np.maximum(np.asarray(xi, dtype=np.float64), 1e-300))
    D = np.maximum(np.asarray(dT_business_s, dtype=np.float64), 1e-12)
    if ov.kappa_vol != 0.0:
        logxi = logxi + 2.0 * ov.kappa_vol
    if ov.kappa_vol_short != 0.0:
        logxi = logxi + 2.0 * ov.kappa_vol_short * np.minimum(1.0, 60.0 / D)
    if ov.shrink_w != 0.0:
        if xi_bar is None:
            raise ValueError("shrink_w needs the unconditional curve xi_bar")
        w = ov.shrink_w * np.exp(-D / max(ov.shrink_decay_s, 1e-9))
        logbar = np.log(np.maximum(np.asarray(xi_bar, dtype=np.float64), 1e-300))
        logxi = (1.0 - w) * logxi + w * logbar
    return np.exp(logxi)


def cap_m_Y(ov: "Overrides", m_Y, var_y):
    """Clip the location term at `alpha_cap_sd` settlement standard deviations."""
    if ov.alpha_cap_sd is None:
        return m_Y
    lim = float(ov.alpha_cap_sd) * np.sqrt(np.maximum(var_y, 0.0))
    return np.clip(m_Y, -lim, lim)


def quoted_prob(ov: "Overrides", p):
    """The quoting-layer temperature. Never touches `p_model`."""
    if ov.temperature == 1.0:
        return p
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    lg = np.log(q / (1.0 - q)) / float(ov.temperature)
    out = 1.0 / (1.0 + np.exp(-lg))
    return float(out) if np.ndim(p) == 0 else out
```

- [ ] **Step 4: Run the fast per-tick tests**

Run: `cd "$INV" && python -m pytest tests/test_overrides_pertick.py -v -m "not slow"`
Expected: 8 PASS.

- [ ] **Step 5: Add `unconditional_xi` to `fvmodel/curve.py`**

Append to `fvmodel/curve.py`:

```python
def unconditional_xi(model, dT: np.ndarray) -> np.ndarray:
    """The forward curve the unconditional register bank would give, at ages `dT`.

    This is the shrink target (spec ruling R6): the seasonal level with no volatility
    news in it at all, which is what `registers.initial_value` encodes. `dT` is in
    business days, matching `dT_at`, and the returned curve is per-second increments
    aligned with `forward_block`'s block.
    """
    v0 = np.asarray(model.params["registers"]["initial_value"], dtype=np.float64)
    g = _iv_from_cum(model.forward, np.asarray(dT, dtype=np.float64),
                     np.log(np.maximum(v0, 1e-300)))
    g = np.atleast_2d(g)
    inc = np.maximum(np.diff(np.concatenate(
        [np.zeros((g.shape[0], 1)), g], axis=1), axis=1), 0.0)
    return inc if np.ndim(dT) > 1 else inc[0]
```

- [ ] **Step 6: Rewire `fvmodel/fairvalue.py` — move `kappa_vol`, apply the transforms**

In `fair_value`, replace the variance block. Find:

```python
    logv = np.log(np.maximum(state.v2.v, 1e-300))
    if model.kappa_vol:
        logv = logv + 2.0 * model.kappa_vol
        v_state = state.v2.copy()
        v_state.v = np.exp(logv)
    else:
        v_state = state.v2
    m_blk = block_length(kind if sw.composed_weights else "perp_twap", n, Lw, PAD)
    iv_head, xi = forward_block(model.v2, v_state, t, n, m_blk,
                                cap_c=model.xi_cap_c, cap_i=model.xi_cap_i)
    rho = model.rho_for(state.v2.act, sw.rho_mode)
    ratio = model.input_var_ratio if (kind == "chainlink_twap60" and sw.use_blend) else 1.0
```

Replace with:

```python
    # The register bank is NOT touched: kappa_vol moved to the forward curve
    # (brief section 4.2), which is what makes the bank variant-invariant and so
    # cacheable across every variant run. See the spec, section 4.3.
    v_state = state.v2
    m_blk = block_length(kind, n, Lw, PAD)
    iv_head, xi = forward_block(model.v2, v_state, t, n, m_blk,
                                cap_c=model.xi_cap_c, cap_i=model.xi_cap_i)
    ov = model.ov
    if xi.size:
        u_blk = np.arange(n - xi.size + 1, n + 1, dtype=np.int64)
        dT_blk = dT_at(model.v2, v_state, t, u_blk) * 86400.0     # business seconds
        xi_bar = (unconditional_xi(model.v2, dT_blk / 86400.0)
                  if ov.shrink_w != 0.0 else None)
        xi = xi_adjust(ov, xi, dT_blk, xi_bar)
        # the head integral carries the same uniform scaling; the short-end and
        # shrink knobs are defined on the block, where the settlement weights live
        if ov.kappa_vol != 0.0:
            iv_head = iv_head * np.exp(2.0 * ov.kappa_vol)
    rho = model.rho_for(state.v2.act)
    ratio = model.input_var_ratio if kind == "chainlink_twap60" else 1.0
```

Add to the imports at the top of `fvmodel/fairvalue.py`:

```python
from .curve import dT_at, forward_block, unconditional_xi
from .overrides import cap_m_Y, quoted_prob, xi_adjust
```

Then, after `m_Y` is computed, cap it:

```python
    m_Y = cap_m_Y(ov, m_Y, var_y)
```

and at the tail, replace the `p_up` line and the return with:

```python
    tail = model.tail_for(kind)
    nu, mu, sg = tail.params(z)
    p_model = float(tail.prob_up(y_star, var_y, z))
    p_quoted = float(quoted_prob(ov, p_model))
```

and in the `FairValue(...)` constructor pass `p_up=p_model, p_model=p_model,
p_quoted=p_quoted` and `switches=ov.label()`.

In the `FairValue` dataclass, add after `p_up`:

```python
    p_model: float = float("nan")
    p_quoted: float = float("nan")
```

- [ ] **Step 7: Rewire `fvmodel/engine.py` the same way**

In `evaluate`, replace:

```python
    logv = win.logv_at(it_q)
    if model.kappa_vol:
        logv = logv + 2.0 * model.kappa_vol
    m_blk = block_length(kind if sw.composed_weights else "perp_twap", n_q, Lw, PAD)
    iv_head, xi = win.clock.forward_block(logv, it_q, n_q, m_blk,
                                          cap_c=model.xi_cap_c,
                                          cap_i=model.xi_cap_i)
    ratio = model.input_var_ratio if (kind == "chainlink_twap60" and sw.use_blend) else 1.0
```

with:

```python
    logv = win.logv_at(it_q)                     # variant-invariant: see spec 4.3
    m_blk = block_length(kind, n_q, Lw, PAD)
    iv_head, xi = win.clock.forward_block(logv, it_q, n_q, m_blk,
                                          cap_c=model.xi_cap_c,
                                          cap_i=model.xi_cap_i)
    ov = model.ov
    if xi.size:
        u_blk = np.arange(n_q - xi.shape[1] + 1, n_q + 1, dtype=np.int64)
        dT_blk = win.clock.dT_at(it_q, u_blk) * 86400.0
        xi_bar = (unconditional_xi(win.bv, dT_blk / 86400.0)
                  if ov.shrink_w != 0.0 else None)
        xi = xi_adjust(ov, xi, dT_blk, xi_bar)
        if ov.kappa_vol != 0.0:
            iv_head = iv_head * np.exp(2.0 * ov.kappa_vol)
    ratio = model.input_var_ratio if kind == "chainlink_twap60" else 1.0
```

Add to `fvmodel/engine.py`'s imports:

```python
from .curve import BatchClock, unconditional_xi
from .overrides import cap_m_Y, quoted_prob, xi_adjust
```

After `m_Y` is computed in `evaluate`, add `m_Y = cap_m_Y(ov, m_Y, var_y)`, and
after `p_up = tail.prob_up(...)` add `p_quoted = quoted_prob(ov, p_up)` and carry
`"p_model": p_up[ok], "p_quoted": np.asarray(p_quoted)[ok]` in the returned rows
(keep the existing `"p_up"` key as well).

`unconditional_xi` reads `model.params`; `BatchV2` stores the params as `self.p`,
so add `self.params = params` next to it in
`fvmodel/_vendor/clkit/state.py`'s `BatchV2.__init__`.

- [ ] **Step 8: Run everything**

Run: `cd "$INV" && python -m pytest tests/ -v`
Expected: all PASS, including `tests/test_golden.py`. The golden test is the
proof that moving `kappa_vol` and threading `xi_adjust` through both paths did
not change the default model.

- [ ] **Step 9: Commit**

```bash
cd /c/Users/kaima/Github2/Gambling104
git add investigations/2026-9-9_business_clock_t_dist
git commit -m "feat: per-tick override transforms; move kappa_vol to the forward curve

The register bank is now variant-invariant, which is what lets one sweep be
shared across every variant run."
```

---

## Task 6: Delete the dead paths

Brief §4.3. `Switches` becomes fully redundant with `Overrides` once the
unreachable branches go, so it is deleted rather than left as a second
configuration object.

**Files:**
- Modify: `fvmodel/fairvalue.py`, `fvmodel/engine.py`, `fvmodel/chainlink.py`, `fvmodel/build.py`, `fvmodel/__init__.py`
- Create: `tests/test_reachability.py`

**Interfaces:**
- Produces: `fair_value(state, market, model, t_now=None)` and
  `evaluate(win, model, kind, L, n, expiries=None, twap_len=60, book=None, emit_all=False)` — **no `sw` parameter on either.** `Overrides` is the only configuration object.

- [ ] **Step 1: Write the failing reachability test**

Create `tests/test_reachability.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "$INV" && python -m pytest tests/test_reachability.py -v`
Expected: FAIL on `test_switches_is_gone` and the dead-branch cases.

- [ ] **Step 3: Delete `Switches` and its branches from `fvmodel/fairvalue.py`**

- Delete the whole `@dataclass class Switches` block.
- Change the signature to `def fair_value(state, market, model, t_now=None) -> FairValue:` and delete the `sw = sw or Switches()` line.
- Replace `sw.*` reads with `model.ov` equivalents:
  - the `if kind == "chainlink_twap60" and not sw.composed_weights:` branch — delete the branch, keep only the `s = settlement_weights(kind, n, model.fp.to_dict(), L=Lw)` line.
  - `b_t = (state.filt.b if (kind == "chainlink_twap60" and sw.use_basis and np.isfinite(state.filt.b)) else 0.0)` becomes a `basis_tracker` switch:

```python
    if kind == "chainlink_twap60" and ov.basis_tracker != "off":
        raw_b = state.filt.b if ov.basis_tracker == "main" else state.filt.b_alt
        b_t = float(raw_b) if np.isfinite(raw_b) else 0.0
    else:
        b_t = 0.0
```
    (move `ov = model.ov` to just after `t` is computed, so it is available here).
  - `d_sum += (lam ** K) * (state.filt.xf - x_t) if sw.composed_weights else 0.0` becomes `d_sum += (lam ** K) * (state.filt.xf - x_t)`.
  - `elif sw.use_reconstruction and not sw.market_observable:` becomes `elif ov.reconstruct_in_transit and ov.information_set == "full":`.
  - `mode = sw.eps_mode if not sw.market_observable else "unconditional"` becomes:

```python
        mode = ("none" if model.eps.sigma_bp == 0.0
                else ("full" if (ov.eps_condition and ov.information_set == "full")
                      else "unconditional"))
```
  - the `if kind == "chainlink_twap60" and sw.eps_mode != "none" and unknown_ages:` guard becomes `if kind == "chainlink_twap60" and model.eps.sigma_bp != 0.0 and unknown_ages:`.
  - `if sw.use_basis:` before `basis_drift_window_var` becomes `if ov.basis_tracker != "off":`.
  - `if sw.use_alpha and market.book_snapshot:` becomes `if market.book_snapshot is not None:` (alpha is turned off by `alpha_scale=0`, which zeroes the betas).
  - `if sw.jensen:` — delete the guard, always subtract.
  - `switches=sw.label()` becomes `switches=ov.label()`.

- [ ] **Step 4: Do the same in `fvmodel/engine.py`**

- Signature becomes `def evaluate(win, model, kind, L, n, expiries=None, twap_len=60, book=None, emit_all=False) -> Cell:` and `ov = model.ov` on the first line.
- `mo = bool(sw.market_observable) and kind == "chainlink_twap60"` becomes `mo = ov.information_set == "prints_only" and kind == "chainlink_twap60"`.
- Delete the `if kind == "chainlink_twap60" and not sw.composed_weights:` branch and the `if not sw.composed_weights: d_bar = np.zeros(...)` line.
- `b_t = win.b[it] if sw.use_basis else np.zeros(it.size)` becomes:

```python
            src = {"main": win.b, "alt": win.b_alt}.get(ov.basis_tracker)
            b_t = np.zeros(it.size) if src is None else src[it]
```
- `if sw.rho_mode == "conditional" and "act0" in model.rho:` becomes `if ov.rho_kernel == "conditional" and "act0" in model.rho:`; `model.rho_for(1.0, sw.rho_mode)` becomes `model.rho_for(1.0)`.
- `if kind == "chainlink_twap60" and sw.eps_mode != "none":` becomes `if kind == "chainlink_twap60" and model.eps.sigma_bp != 0.0:`; the `mode` line mirrors Step 3.
- `if sw.use_alpha and book is not None:` becomes `if book is not None:`.
- `if sw.use_basis:` becomes `if ov.basis_tracker != "off":`.
- `tail = model.tail_for(kind, sw.tail_mode)` becomes `tail = model.tail_for(kind)`.
- `if sw.jensen:` — delete the guard, always subtract.

Then replace the whole `return Cell(...)` statement with the block below. It adds
the tail parameters and both probabilities as columns (the export needs them),
and makes `emit_all` keep the unpriceable rows with an `ok` flag instead of
dropping them — the export wants a row for every market second, priceable or not.

```python
    nu_, mu_, sg_ = tail.params(z)
    p_quoted = np.asarray(quoted_prob(ov, p_up), dtype=np.float64)
    sel = np.ones(ok.size, dtype=bool) if emit_all else ok
    m = int(sel.sum())
    rows = {
        "T": T[sel], "t": T[sel] - n, "p_up": p_up[sel], "p_model": p_up[sel],
        "p_quoted": p_quoted[sel], "var_y": var_y[sel], "y_star": y_star[sel],
        "resid": resid[sel], "z": z[sel], "up": up[sel],
        "nu": np.asarray(nu_)[sel], "mu": np.asarray(mu_)[sel],
        "sigma_t": np.asarray(sg_)[sel],
        "omega": np.full(m, omega), "n_known": np.full(m, n_recv),
        "n_transit": np.full(m, n_transit),
        "m_Y": m_Y[sel] if np.ndim(m_Y) else np.zeros(m),
        "eps_bar": eps_bar[sel],
        "var_eps": var_eps[sel] if np.ndim(var_eps) else np.full(m, var_eps),
        "var_basis": np.full(m, float(var_basis)),
        "carry": d_bar[sel] if np.ndim(d_bar) else np.zeros(m),
        "act": act[sel], "settle": settle[sel], "strike": strike[sel],
        "it": it[sel], "it_q": it_q[sel], "p_ref": p_ref[sel],
        "ok": ok[sel],
    }
    return Cell(kind, L, n, rows)
```

`tail.params(z)` returns scalars when the family is normal, so the
`np.asarray(...)[sel]` calls need `nu_`, `mu_`, `sg_` broadcast to `z`'s shape
first: put `nu_, mu_, sg_ = (np.broadcast_to(np.asarray(a), z.shape) for a in
tail.params(z))` in place of the plain unpack.

- [ ] **Step 5: Add `b_alt` to `Window` and `EPS_MODES` to `chainlink.py`**

In `fvmodel/engine.py`'s `Window.__init__`, right after `self.b = basis_series(...)`:

```python
            # the 15 s tracker, so basis_tracker="alt" is a selection rather than
            # a refit; it costs one more pass over the same series
            self.b_alt = basis_series(d_lvl, usable, fp.basis_hl_alt_s, lag_s=LAG_S)
            self.eps_alt = np.where(usable & np.isfinite(self.b_alt),
                                    d_lvl - self.b_alt, np.nan)
```

and in `trim()` add `"b_alt"`, `"eps_alt"` to the float32 list.

In `fvmodel/chainlink.py`, add near the top:

```python
EPS_MODES = ("full", "unconditional", "none")
```

and delete the `if mode == "iid":` branch from `eps_conditional`, raising on an
unknown mode instead:

```python
    if mode not in EPS_MODES:
        raise ValueError("eps mode must be one of %s, got %r" % (EPS_MODES, mode))
```

- [ ] **Step 6: Clean `build.py` and `__init__.py`**

In `fvmodel/build.py` delete the `input_mode="perp_only"` branch of `build_model`
and the `perp_only` reads (the perp-only ablation is not selectable and is report
material). Reduce the signature to
`def build_model(eps_fit: str = "train", tails: dict = None, basis_hl_s: float = None) -> FairValueModel:`.

In `fvmodel/__init__.py`, remove `Switches` from the imports and `__all__`, and
add `Overrides`, `apply_overrides`.

- [ ] **Step 7: Port the engine-equality test and extend it to every override**

Spec §10 item 6 and brief §4.4. This is the mechanism that satisfies ruling R4:
the scalar and vectorised paths stay one model because they share `Overrides`,
share the per-tick transforms, and are pinned together here — not because they
were merged into one function body, which would cost the vectorised path.

Create `tests/test_engine_matches_fairvalue.py`:

```python
"""The batch evaluation and the production call must be the same model.

Two implementations of the same arithmetic drift the moment either is touched, and
the drift is invisible: the export would keep producing plausible numbers for a model
the harness does not run. So the same quotes are priced both ways, under EVERY
override, and the answers have to agree to floating point.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow

# one representative value per override key, chosen to actually bite
OVERRIDE_CASES = [
    {},
    {"kappa_vol": 0.15},
    {"kappa_vol_short": -0.22},
    {"shrink_w": 0.5, "shrink_decay_s": 60.0},
    {"rho_kernel": "unconditional"},
    {"rho_kernel": "off"},
    {"eps_scale": 2.0},
    {"eps_scale": 0.0},
    {"eps_condition": False},
    {"basis_tracker": "alt"},
    {"basis_tracker": "off"},
    {"information_set": "prints_only"},
    {"reconstruct_in_transit": False},
    {"kappa_tail": -0.5},
    {"tail_scale": 1.2},
    {"tail_family": "normal"},
    {"alpha_scale": 0.0},
    {"alpha_scale": 2.0},
    {"alpha_scale_age": 0.0},
    {"alpha_cap_sd": 0.5},
    {"w_spot": 0.35},
    {"xi_cap_c": 2.0},
    {"temperature": 1.5},
]
CELLS = [("chainlink_twap60", 300, 300), ("chainlink_twap60", 300, 60),
         ("chainlink_twap60", 300, 5), ("perp_single", 900, 300),
         ("perp_twap", 900, 120)]


@pytest.fixture(scope="module")
def win_base():
    from fvmodel.base import CL_T0, load_params
    from fvmodel.build import build_model
    from fvmodel.engine import Window

    base = build_model()
    t0 = CL_T0 + 20 * 86400
    win = Window(load_params(), base.fp, t0, t0 + 3 * 86400, warm=0, chainlink=True,
                 log=lambda *a: None)
    return win, base


@pytest.mark.parametrize("ovkw", OVERRIDE_CASES,
                         ids=[",".join(d) or "defaults" for d in OVERRIDE_CASES])
@pytest.mark.parametrize("kind,L,n", CELLS)
def test_batch_equals_single_quote(win_base, ovkw, kind, L, n):
    from fvmodel.engine import evaluate, market_at, market_grid, state_at
    from fvmodel.fairvalue import fair_value
    from fvmodel.overrides import Overrides, apply_overrides

    win, base = win_base
    ov = Overrides(**ovkw)
    # w_spot changes the blend, so the window's print-model series must be rebuilt
    if ovkw.get("w_spot") is not None:
        pytest.skip("w_spot needs its own Window; covered by test_w_spot_rebuild")
    model = apply_overrides(base, ov)

    T = market_grid(win, L, burn_days=2)
    assert T.size > 20
    win.prepare(np.unique(np.concatenate([win.i(T) - n, win.i(T) - n - 2])))
    cell = evaluate(win, model, kind, L, n, expiries=T)
    assert len(cell) > 10

    picked = 0
    for k in range(0, len(cell), max(len(cell) // 6, 1)):
        Tk = int(cell.rows["T"][k])
        it = win.i(Tk) - n
        if it < 64:
            continue
        st = state_at(win, it, model)
        mk = market_at(win, kind, Tk, n, L, 60)
        one = fair_value(st, mk, model, t_now=int(win.ts[it]))
        for field, tol in (("var_y", 1e-8), ("y_star", 1e-7), ("omega", 0),
                           ("carry", 1e-7), ("eps_bar", 1e-8), ("var_eps", 1e-8),
                           ("p_model", 1e-8), ("p_quoted", 1e-8)):
            a = getattr(one, field)
            b = float(cell.rows[field][k])
            assert a == pytest.approx(b, rel=tol, abs=1e-14), (
                "%s under %s at kind=%s n=%d: single quote %.12g vs batch %.12g"
                % (field, ov.label(), kind, n, a, b))
        picked += 1
    assert picked >= 3


def test_w_spot_rebuild(win_base):
    """w_spot changes the input blend, so the batch path needs its own Window."""
    from fvmodel.base import load_params
    from fvmodel.engine import Window, evaluate, market_at, market_grid, state_at
    from fvmodel.fairvalue import fair_value
    from fvmodel.overrides import Overrides, apply_overrides

    win0, base = win_base
    model = apply_overrides(base, Overrides(w_spot=0.35))
    win = Window(load_params(), model.fp, int(win0.ts[0]), int(win0.ts[-1]),
                 warm=0, chainlink=True, log=lambda *a: None)
    T = market_grid(win, 300, burn_days=2)
    win.prepare(np.unique(np.concatenate([win.i(T) - 60, win.i(T) - 62])))
    cell = evaluate(win, model, "chainlink_twap60", 300, 60, expiries=T)
    k = len(cell) // 2
    Tk = int(cell.rows["T"][k])
    it = win.i(Tk) - 60
    one = fair_value(state_at(win, it, model),
                     market_at(win, "chainlink_twap60", Tk, 60, 300, 60),
                     model, t_now=int(win.ts[it]))
    assert one.var_y == pytest.approx(float(cell.rows["var_y"][k]), rel=1e-8)
    assert one.y_star == pytest.approx(float(cell.rows["y_star"][k]), rel=1e-7)


def test_no_lookahead(win_base):
    """Perturbing data strictly after the quote must not move the quote."""
    from fvmodel.base import load_params
    from fvmodel.engine import Window, market_at, state_at
    from fvmodel.fairvalue import fair_value

    win, model = win_base
    it = win.n - 4000
    win.prepare(np.array([it, it - 2]))
    st = state_at(win, it, model)
    mk = market_at(win, "chainlink_twap60", int(win.ts[it]) + 300, 300, 900, 60)
    ref = fair_value(st, mk, model, t_now=int(win.ts[it]))

    win2 = Window(load_params(), model.fp, int(win.ts[0]), int(win.ts[-1]), warm=0,
                  chainlink=True, log=lambda *a: None)
    for name in ("perp", "cl_step"):
        a = getattr(win2, name)
        a[it + 1:] = a[it + 1:] * 1.05
    win2.prepare(np.array([it, it - 2]))
    got = fair_value(state_at(win2, it, model), mk, model, t_now=int(win.ts[it]))
    for field in ("var_y", "carry", "eps_bar", "var_eps", "omega"):
        assert getattr(ref, field) == pytest.approx(getattr(got, field), abs=1e-15), (
            "%s moved when data after the quote time changed" % field)
```

- [ ] **Step 8: Run it**

Run: `cd "$INV" && python -m pytest tests/test_engine_matches_fairvalue.py -v`
Expected: 115 parametrised cases plus two named tests, all PASS. A failure names
the override and the field, which is exactly the drift this test exists to catch.

- [ ] **Step 9: Run everything**

Run: `cd "$INV" && python -m pytest tests/ -v`
Expected: all PASS. `tests/test_golden.py` passing here is the proof that
deleting the dead branches did not disturb the live one.

- [ ] **Step 10: Commit**

```bash
cd /c/Users/kaima/Github2/Gambling104
git add investigations/2026-9-9_business_clock_t_dist
git commit -m "refactor: delete Switches and the unreachable ablation branches

Overrides is now the model's only configuration object."
```

---

## Task 7: The export builder and its caches

**Files:**
- Create: `export/__init__.py`, `export/cache.py`, `export/build_export.py`, `tests/test_export.py`

**Interfaces:**
- Consumes: `evaluate(..., emit_all=True)` and the row keys added in Task 6 Step 4 — `nu`, `mu`, `sigma_t`, `p_model`, `p_quoted`, `ok`.
- Produces:
  - `export.cache.register_bank(win, idx: np.ndarray, params_sha: str) -> np.ndarray` — memoised `logv`, keyed by `(params_sha, win.ts[0], win.n, idx)`.
  - `export.build_export.build(variant: str, t0: int, t1: int, out_path: Path | None = None) -> Path`
  - `export.build_export.usable_t_ms(t_s: int) -> int` — the first 100 ms bucket at which a row is usable.
  - `export.build_export.bucket_owner() -> np.ndarray` — (3000,) of `t_s`, the inverse of `usable_t_ms`. **The single definition of the causality shift.**
  - `export.build_export.T_S_RANGE` — `range(-1, 300)`, 301 rows per market.
  - Export columns: `market_id (str), open_ts (int64), t_s (int32), s (float64), sigma (float64), nu, mu, sigma_t, p_model, p_quoted, omega, n_known, n_transit, m_Y, eps_bar, carry, var_eps, var_basis, z_business, p_ref, ok (bool)`

- [ ] **Step 1: Write the failing export tests**

Create `tests/test_export.py`:

```python
"""The export: its algebra, its causality, and that it is strike-free."""
from __future__ import annotations

import numpy as np
import pytest


def test_usable_bucket_is_one_shift_after_the_instant():
    from export.build_export import usable_t_ms

    # a value carrying information through open_ts + t_s becomes usable one 100 ms
    # bucket later; t_s = -1 is the pre-open row that fills bucket 0
    assert usable_t_ms(-1) == 0
    assert usable_t_ms(0) == 100
    assert usable_t_ms(1) == 1100
    assert usable_t_ms(299) == 299900


def test_bucket_owner_partitions_the_grid():
    from export.build_export import T_S_RANGE, bucket_owner

    own = bucket_owner()
    assert own.shape == (3000,)
    assert own[0] == -1, "bucket 0 is owned by the pre-open row"
    assert own[1] == 0 and own[10] == 0, "t_s=0 owns buckets 1..10"
    assert own[11] == 1
    assert own[2999] == 299
    assert set(own.tolist()) == set(T_S_RANGE)
    assert len(T_S_RANGE) == 301


@pytest.mark.slow
def test_s_and_sigma_reproduce_y_star():
    """s is strike-free and y*(K) = (K - s) / (p_ref * omega). If that identity
    holds, f() and link() in the blocks are the model, not an approximation."""
    import numpy as np

    from fvmodel.base import CL_T0, load_params
    from fvmodel.build import build_model
    from fvmodel.engine import Window, evaluate, market_grid

    model = build_model()
    t0 = CL_T0 + 20 * 86400
    win = Window(load_params(), model.fp, t0, t0 + 2 * 86400, warm=0, chainlink=True,
                 log=lambda *a: None)
    T = market_grid(win, 300, burn_days=1)[:200]
    win.prepare(win.i(T) - 120)
    cell = evaluate(win, model, "chainlink_twap60", 300, 120, expiries=T)
    r = cell.rows
    s = r["strike"] - r["y_star"] * r["p_ref"] * r["omega"]
    y_back = (r["strike"] - s) / (r["p_ref"] * r["omega"])
    assert np.allclose(y_back, r["y_star"], rtol=1e-12, atol=1e-18)
    # and s must not depend on the strike: shift the strike, s must move with it
    # exactly, leaving the implied breakeven unchanged in the model's own terms
    assert np.all(np.isfinite(s))


@pytest.mark.slow
def test_export_has_a_row_for_every_market_second(tmp_path):
    import polars as pl

    from export.build_export import T_S_RANGE, build

    t0 = 1786665600
    p = build("baseline", t0, t0 + 3 * 3600, out_path=tmp_path / "baseline.parquet")
    df = pl.read_parquet(p)
    per = df.group_by("market_id").len()
    assert per["len"].min() == len(T_S_RANGE) == per["len"].max()
    assert df["ok"].sum() > 0.8 * len(df), "most seconds should be priceable"
    assert set(df.columns) >= {"market_id", "open_ts", "t_s", "s", "sigma", "nu",
                               "mu", "sigma_t", "p_model", "p_quoted", "ok"}


@pytest.mark.slow
def test_export_is_causal(tmp_path):
    """Every bucket must be served by a row whose information instant precedes it."""
    import polars as pl

    from export.build_export import bucket_owner

    own = bucket_owner()
    bucket_ms = np.arange(own.size, dtype=np.int64) * 100
    # the row owning a bucket carries information through the instant open_ts + t_s,
    # which is t_s * 1000 ms after the open. That instant must be STRICTLY before the
    # bucket it serves, or a block is reading its own tick or the future.
    info_ms = own.astype(np.int64) * 1000
    assert np.all(info_ms < bucket_ms), (
        "%d bucket(s) served by a row from their own instant or later"
        % int((info_ms >= bucket_ms).sum()))

- [ ] **Step 2: Run to verify it fails**

Run: `cd "$INV" && python -m pytest tests/test_export.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'export'`

- [ ] **Step 3: Write `export/cache.py`**

```python
"""The layers that are shared across variant runs.

L1 - the register bank - is the expensive one and, because kappa_vol moved to the
forward curve, it does not depend on any override. It is swept once for the window
and every variant reads it. That is the warm start: the 14-day burn-in is paid once
rather than once per variant, and no market is lost to it.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from fvmodel.base import CACHE


def _key(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def register_bank(win, idx: np.ndarray, params_sha: str) -> np.ndarray:
    """`logv` at `idx`, cached on disk. Variant-invariant by construction."""
    idx = np.unique(np.asarray(idx, dtype=np.int64))
    path = Path(CACHE) / ("logv_%s.npz" % _key(params_sha, int(win.ts[0]), win.n,
                                               idx.size, int(idx[0]), int(idx[-1])))
    if path.exists():
        z = np.load(path)
        if np.array_equal(z["idx"], idx):
            win._cache_idx = idx
            win._cache_logv = z["logv"].astype(np.float64)
            return win._cache_logv
    win.prepare(idx)
    np.savez_compressed(path, idx=idx, logv=win._cache_logv.astype(np.float32))
    return win._cache_logv
```

- [ ] **Step 4: Write `export/build_export.py`**

```python
"""Build the fair export one variant reads: strike-free, one row per market second.

`s` is the settlement level at which the model is indifferent, in USD, which is the
harness's `E[A]`; `sigma` is the settlement standard deviation in the same units.
Because y* is affine in the strike, those two plus the tail parameters are the whole
model - the strike work happens in `f.py` and `link.py`, which is exactly the split
the harness's slot contract asks for.

    python -m export.build_export --variant baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

INV = Path(__file__).resolve().parents[1]
if str(INV) not in sys.path:
    sys.path.insert(0, str(INV))

from fvmodel import config                                  # noqa: E402
from fvmodel.base import load_params                        # noqa: E402
from fvmodel.engine import Window, evaluate                 # noqa: E402
from export.cache import register_bank                      # noqa: E402

MARKET_LEN = 300           # the 5 m window
TWAP_LEN = 60              # the Chainlink settlement average
BUCKET_MS = 100
# t_s = -1 fills bucket 0; t_s = 299 fills bucket 2999. See usable_t_ms.
T_S_RANGE = list(range(-1, MARKET_LEN))
FAIR_DIR = INV.parents[1] / "backtesting_5m" / "data" / "fair"


def usable_t_ms(t_s: int) -> int:
    """The first 100 ms bucket at which a row carrying information through the
    instant `open_ts + t_s` may be used.

    The harness's convention (its spec section 1.4) is that a decision at bucket i
    may only use information from strictly before it. The model's state at instant
    `t` is the price at `t`, so a value becomes usable one 100 ms bucket later and
    holds until the next second's value supersedes it. `t_s = -1` is the pre-open row
    that serves bucket 0.
    """
    return max(0, int(t_s) * 1000 + 100)


def bucket_owner() -> np.ndarray:
    """For each of the 3,000 buckets, the `t_s` whose value it must use.

    The inverse of `usable_t_ms`, materialised once. Every block reads the export
    through this, so the causality shift is defined in exactly one place.
    """
    i = np.arange(MARKET_LEN * 1000 // BUCKET_MS, dtype=np.int64)
    return (i * BUCKET_MS - 100) // 1000


_WINDOWS: dict = {}


def _window(model, t0: int, t1: int, burn: int):
    """One Window per (span, filter constants), shared across variant runs.

    Building a Window sweeps the seasonality, the activity factor and the whole print
    model over a month of seconds. Only `w_spot` and the basis half-lives change any
    of that, so every other variant reuses the same object and pays for it once.
    """
    fp = model.fp
    key = (t0, t1, burn, fp.w_spot, fp.tau_s, fp.delta_s, fp.basis_hl_s,
           fp.basis_hl_alt_s, fp.lag_s)
    if key not in _WINDOWS:
        _WINDOWS.clear()                   # one at a time; a Window is large
        _WINDOWS[key] = Window(load_params(), fp, t0 - burn, t1, warm=0,
                               chainlink=True,
                               log=lambda *a: print("   ", *a, flush=True))
    return _WINDOWS[key]


def _strikes(t0: int, t1: int) -> pl.DataFrame:
    from harness_paths import STRIKES                      # see Step 5
    return (pl.read_parquet(STRIKES)
            .filter((pl.col("open_ts") >= t0) & (pl.col("open_ts") + MARKET_LEN <= t1))
            .sort("open_ts"))


def build(variant: str, t0: int, t1: int, out_path: Path | None = None) -> Path:
    loaded = config.load(variant)
    model, ov = loaded.model, loaded.overrides
    print("variant %s: %s" % (variant, ov.label()), flush=True)

    mk = _strikes(t0, t1)
    T = (mk["open_ts"].to_numpy() + MARKET_LEN).astype(np.int64)
    ids = mk["market_id"].to_numpy()
    opens = mk["open_ts"].to_numpy().astype(np.int64)

    burn = loaded.cfg["window"]["burn_days"] * 86400
    win = _window(model, t0, t1, burn)
    iT = win.i(T)
    idx = np.unique(np.concatenate([iT - (MARKET_LEN - t) for t in T_S_RANGE]))
    idx = idx[(idx > 0) & (idx < win.n)]
    register_bank(win, idx, loaded.cfg["base"]["params_sha256"])

    frames = []
    for t_s in T_S_RANGE:
        n = MARKET_LEN - t_s                # seconds remaining at the info instant
        cell = evaluate(win, model, "chainlink_twap60", MARKET_LEN, n,
                        expiries=T, book=None, emit_all=True)
        r = cell.rows
        s = r["strike"] - r["y_star"] * r["p_ref"] * r["omega"]
        pos = np.searchsorted(T, r["T"])
        frames.append(pl.DataFrame({
            "market_id": ids[pos], "open_ts": opens[pos],
            "t_s": np.full(r["T"].size, t_s, dtype=np.int32),
            "s": s, "sigma": np.sqrt(r["var_y"]) * r["p_ref"] * r["omega"],
            "nu": r["nu"], "mu": r["mu"], "sigma_t": r["sigma_t"],
            "p_model": r["p_model"], "p_quoted": r["p_quoted"],
            "omega": r["omega"], "n_known": r["n_known"], "n_transit": r["n_transit"],
            "m_Y": r["m_Y"], "eps_bar": r["eps_bar"], "carry": r["carry"],
            "var_eps": r["var_eps"], "var_basis": r["var_basis"],
            "z_business": r["z"], "p_ref": r["p_ref"], "ok": r["ok"]}))
    df = pl.concat(frames).sort(["open_ts", "t_s"])

    out = Path(out_path) if out_path else FAIR_DIR / ("%s.parquet" % variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    meta = out.with_suffix(".json")
    meta.write_text(json.dumps({"provenance": loaded.provenance,
                                "window": {"t0": t0, "t1": t1},
                                "rows": len(df),
                                "markets": int(df["market_id"].n_unique()),
                                "ok_fraction": float(df["ok"].mean())}, indent=2,
                               default=str) + "\n")
    print("wrote %s: %d rows, %d markets, ok %.4f"
          % (out, len(df), df["market_id"].n_unique(), df["ok"].mean()))
    return out


def main() -> None:
    cfg = config.read_config()["window"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--t0", type=int, default=cfg["backtest_t0"])
    ap.add_argument("--t1", type=int, default=cfg["backtest_t1"])
    a = ap.parse_args()
    build(a.variant, a.t0, a.t1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add the harness path shim**

Create `harness_paths.py` at `$INV` root — **not** a slot name, so it is safe:

```python
"""Where the 5 m harness keeps its panel. Kept in one place so a harness move is a
one-line change here rather than a search across the export and the scorecard."""
from pathlib import Path

BACKTESTING = Path(__file__).resolve().parents[2] / "backtesting_5m"
PANEL = BACKTESTING / "data" / "book_5m_100ms.parquet"
STRIKES = BACKTESTING / "data" / "strikes_5m.parquet"
FAIR_DIR = BACKTESTING / "data" / "fair"
```

and change `build_export.py`'s import to `from harness_paths import STRIKES`.

- [ ] **Step 6: Run the fast export tests**

Run: `cd "$INV" && python -m pytest tests/test_export.py -v -m "not slow"`
Expected: 2 PASS.

- [ ] **Step 7: Run the slow export tests**

Run: `cd "$INV" && python -m pytest tests/test_export.py -v`
Expected: 4 PASS. `ok` fraction should be > 0.8; if it is much lower, print the
`ok` breakdown by `t_s` before assuming the model is wrong — the most likely
cause is missing Chainlink seconds in the settlement window.

- [ ] **Step 8: Build the baseline export end to end**

Run: `cd "$INV" && python -m export.build_export --variant baseline`
Expected: `wrote .../backtesting_5m/data/fair/baseline.parquet: ~1.5M rows, ~4900 markets, ok 0.9x`, in under about five minutes. Record the wall time — Task 9's runner reports it.

- [ ] **Step 9: Commit**

```bash
cd /c/Users/kaima/Github2/Gambling104
git add investigations/2026-9-9_business_clock_t_dist
git commit -m "feat: strike-free fair export with a shared register-bank cache"
```

---

## Task 8: The four block files

**Files:**
- Create: `_fvexport.py`, `fair.py`, `vol.py`, `f.py`, `link.py`, `tests/test_blocks.py`

**Interfaces:**
- Produces:
  - `_fvexport.for_episode(ep) -> dict[str, np.ndarray]` — every export column on the episode's 3,000-bucket grid, NaN where absent.
  - `fair.precompute(ep) -> np.ndarray` (N,) USD
  - `vol.precompute(ep) -> np.ndarray` (N,) USD
  - `f.standardise(level, strike, sigma) -> float | np.ndarray`
  - `link.bind(ep) -> Callable[[float, int], float]` and `link.link(z, i)`

- [ ] **Step 1: Write the failing block tests**

Create `tests/test_blocks.py`:

```python
"""The blocks must reproduce the model's own p_up from s, sigma and the tail."""
from __future__ import annotations

import numpy as np
import pytest


class FakeEpisode:
    """The minimum of the harness's Episode that the blocks read."""

    def __init__(self, market_id, open_ts, strike, n=3000):
        self.market_id = market_id
        self.open_ts = open_ts
        self.strike = strike
        self.t_ms = np.arange(n, dtype=np.int64) * 100


def test_standardise_is_the_signed_moneyness():
    import f

    assert f.standardise(101.0, 100.0, 2.0) == pytest.approx(0.5)
    assert f.standardise(99.0, 100.0, 2.0) == pytest.approx(-0.5)
    assert np.isnan(f.standardise(101.0, 100.0, 0.0))


def test_link_is_increasing_and_centres_on_the_tail_location():
    import link

    p = [link._prob(z, nu=6.0, mu=0.0, sigma_t=1.0) for z in (-2.0, 0.0, 2.0)]
    assert p[0] < p[1] < p[2]
    assert p[1] == pytest.approx(0.5)
    # a non-zero tail location shifts where p crosses a half
    assert link._prob(0.0, nu=6.0, mu=0.3, sigma_t=1.0) > 0.5


@pytest.mark.slow
def test_blocks_reproduce_the_export_probability():
    """f() then link() must give back p_quoted for the market's own strike."""
    import polars as pl

    import f
    import link
    from harness_paths import FAIR_DIR, STRIKES

    df = pl.read_parquet(FAIR_DIR / "baseline.parquet").filter(pl.col("ok"))
    strikes = pl.read_parquet(STRIKES).select(["market_id", "strike"])
    j = df.join(strikes, on="market_id").sample(2000, seed=0)
    z = f.standardise(j["s"].to_numpy(), j["strike"].to_numpy(), j["sigma"].to_numpy())
    p = link._prob(z, j["nu"].to_numpy(), j["mu"].to_numpy(), j["sigma_t"].to_numpy())
    assert np.allclose(p, j["p_quoted"].to_numpy(), atol=1e-9), (
        "the block chain disagrees with the model that produced the export")


@pytest.mark.slow
def test_precompute_covers_every_bucket_and_is_causal():
    import polars as pl

    import fair
    import vol
    from harness_paths import FAIR_DIR, STRIKES

    row = pl.read_parquet(STRIKES).row(100, named=True)
    ep = FakeEpisode(row["market_id"], row["open_ts"], row["strike"])
    s = fair.precompute(ep)
    sg = vol.precompute(ep)
    assert s.shape == (3000,) and sg.shape == (3000,)
    assert np.isfinite(s).mean() > 0.9 and np.all(sg[np.isfinite(sg)] > 0)
    # bucket 0 is served by the pre-open row; buckets 1..10 by t_s = 0
    df = pl.read_parquet(FAIR_DIR / "baseline.parquet").filter(
        pl.col("market_id") == ep.market_id).sort("t_s")
    pre = df.filter(pl.col("t_s") == -1)["s"][0]
    first = df.filter(pl.col("t_s") == 0)["s"][0]
    assert s[0] == pytest.approx(pre)
    assert s[1] == pytest.approx(first) and s[10] == pytest.approx(first)
    assert s[11] == pytest.approx(df.filter(pl.col("t_s") == 1)["s"][0])
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "$INV" && python -m pytest tests/test_blocks.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'f'`

- [ ] **Step 3: Write `_fvexport.py`**

```python
"""Shared loader for the four block files.

Deliberately NOT named after a harness block slot: the investigation root is the
slot namespace, so a helper called `fees.py` here would silently replace the
harness's fee model.

The export is one second per row; the harness grid is 100 ms. `for_episode` expands
one to the other through `export.build_export.bucket_owner`, so the causality shift
lives in exactly one place and is not restated here.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
import polars as pl

from export.build_export import MARKET_LEN, bucket_owner
from harness_paths import FAIR_DIR

N_BUCKET = MARKET_LEN * 10
OWNER = bucket_owner()
COLS = ("s", "sigma", "nu", "mu", "sigma_t", "p_model", "p_quoted", "omega", "ok")


def variant() -> str:
    return os.environ.get("FV_VARIANT", "baseline")


@functools.lru_cache(maxsize=4)
def _table(name: str) -> pl.DataFrame:
    p = Path(FAIR_DIR) / ("%s.parquet" % name)
    if not p.exists():
        raise FileNotFoundError(
            "no fair export for variant %r at %s; build it with "
            "`python -m export.build_export --variant %s`" % (name, p, name))
    return pl.read_parquet(p).sort(["market_id", "t_s"])


@functools.lru_cache(maxsize=4096)
def _for_market(name: str, market_id: str) -> dict:
    d = _table(name).filter(pl.col("market_id") == market_id).sort("t_s")
    if len(d) == 0:
        return {c: np.full(N_BUCKET, np.nan) for c in COLS}
    # gather, do not scatter: for every bucket, take the row OWNER says serves it.
    # A row missing from the export leaves its buckets NaN, which the harness already
    # treats as "no fair value here" rather than forward-filling.
    t_s = d["t_s"].to_numpy()
    pos = np.searchsorted(t_s, OWNER)
    have = (pos < t_s.size) & (t_s[np.clip(pos, 0, t_s.size - 1)] == OWNER)
    pos = np.clip(pos, 0, t_s.size - 1)
    out = {}
    for c in COLS:
        v = d[c].to_numpy().astype(np.float64)
        out[c] = np.where(have, v[pos], np.nan)
    return out


def for_episode(ep) -> dict:
    return _for_market(variant(), ep.market_id)
```

- [ ] **Step 4: Write the four slot files**

`fair.py`:

```python
"""fair block: the settlement level the model is indifferent at, in USD.

This is the harness's `E[A]`. It is strike-free by construction - y* is affine in
the strike - so the strike enters in `f.py` and nowhere else.
"""
from _fvexport import for_episode


def precompute(ep):
    return for_episode(ep)["s"]
```

`vol.py`:

```python
"""vol block: the settlement standard deviation, in the same USD units as `s`."""
from _fvexport import for_episode


def precompute(ep):
    return for_episode(ep)["sigma"]
```

`f.py`:

```python
"""f block: standardise a settlement level against the strike.

    z = (level - strike) / sigma

so z is the signed moneyness in settlement standard deviations and increases with
the level, which is the direction `link` expects: a higher fair value means UP is
more likely.
"""
import numpy as np


def standardise(level, strike, sigma):
    sg = np.asarray(sigma, dtype=np.float64)
    return np.where(sg > 0, (np.asarray(level, dtype=np.float64)
                             - np.asarray(strike, dtype=np.float64))
                    / np.where(sg > 0, sg, 1.0), np.nan)
```

`link.py`:

```python
"""link block: the fitted settlement tail.

    P(up) = P(Y > y*) = 1 - F_t(-z; mu, sigma_t) = F_t((z + mu) / sigma_t)

by the symmetry of the Student-t, with (nu, mu, sigma_t) read from the export at
the tick's own business time.

NOTE FOR THE HARNESS: the slot contract is a pure `link(z) -> p`, but this tail's
parameters are a function of business time left and are therefore per-tick, the
same way `f` is bound to `strike` and `sigma[i]`. `bind(ep)` returns the per-tick
callable; if the harness cannot bind it, `link(z, i)` takes the index directly.
"""
import numpy as np
from scipy import stats

from _fvexport import for_episode

_BOUND = {}


def _prob(z, nu, mu, sigma_t):
    z = np.asarray(z, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    sg = np.maximum(np.asarray(sigma_t, dtype=np.float64), 1e-12)
    return stats.t.cdf((z + np.asarray(mu, dtype=np.float64)) / sg, df=nu)


def bind(ep):
    c = for_episode(ep)

    def link(z, i):
        return float(_prob(z, c["nu"][i], c["mu"][i], c["sigma_t"][i]))

    _BOUND[id(ep)] = link
    return link


def link(z, i, ep=None):
    if ep is not None:
        return bind(ep)(z, i)
    if not _BOUND:
        raise RuntimeError("link.bind(ep) must be called once per episode")
    return next(reversed(list(_BOUND.values())))(z, i)
```

- [ ] **Step 5: Run the block tests**

Run: `cd "$INV" && python -m pytest tests/test_blocks.py -v`
Expected: 4 PASS (the slow ones need `baseline.parquet` from Task 7 Step 8).

- [ ] **Step 6: Re-run the reachability test**

Run: `cd "$INV" && python -m pytest tests/test_reachability.py -v`
Expected: PASS — in particular `test_investigation_root_holds_only_the_intended_slots`, which now sees `fair.py`, `vol.py`, `f.py`, `link.py` and must **not** see `quote.py`, `execution.py`, `fill.py` or `fees.py`.

- [ ] **Step 7: Commit**

```bash
cd /c/Users/kaima/Github2/Gambling104
git add investigations/2026-9-9_business_clock_t_dist
git commit -m "feat: the four harness block slots over the fair export"
```

---

## Task 9: Variants, the sweep spec and the runner

**Files:**
- Create: `fvmodel/variants.py`, `variants/*.yaml`, `variants/README.md`, `variants/sweeps/example.yaml`, `run_variants.py`, `tests/test_variants.py`

**Interfaces:**
- Produces:
  - `fvmodel.variants.read_variant(name: str) -> dict` — `{name, base_params, overrides, notes}`
  - `fvmodel.variants.list_variants() -> list[str]`
  - `fvmodel.variants.expand_sweep(spec: dict) -> list[dict]`
  - `run_variants.py` CLI: `--variants a,b,c | --all | --sweep FILE`, `--holdout`, `--i-know`, `--no-score`

- [ ] **Step 1: Write the failing variants tests**

Create `tests/test_variants.py`:

```python
from __future__ import annotations

import pytest

SHIPPED = ["baseline", "market_observable", "no_alpha", "alpha_x2", "alpha_half",
           "vol_plus10", "vol_minus10", "short_vol_minus20", "shrink_05",
           "fat_tails", "thin_tails", "normal_tail", "eps_x0", "eps_x2",
           "basis_alt", "rho_off"]


def test_every_shipped_variant_exists():
    from fvmodel.variants import list_variants
    assert sorted(list_variants()) == sorted(SHIPPED)


@pytest.mark.parametrize("name", SHIPPED)
def test_every_variant_builds_a_valid_overrides(name):
    from fvmodel.overrides import Overrides
    from fvmodel.variants import read_variant

    v = read_variant(name)
    assert v["base_params"].startswith("model.json@fv-")
    assert v["notes"].strip(), "%s has no notes" % name
    Overrides.from_dict(v["overrides"])


def test_baseline_is_all_defaults():
    from fvmodel.overrides import Overrides
    from fvmodel.variants import read_variant

    assert Overrides.from_dict(read_variant("baseline")["overrides"]).label() == "baseline"


def test_variants_are_distinct():
    from fvmodel.overrides import Overrides
    from fvmodel.variants import list_variants, read_variant

    labels = {n: Overrides.from_dict(read_variant(n)["overrides"]).label()
              for n in list_variants()}
    assert len(set(labels.values())) == len(labels), "duplicate variants: %s" % labels


def test_sweep_expands_to_the_product_and_respects_the_cap():
    from fvmodel.variants import expand_sweep

    spec = {"name_prefix": "vol", "max_variants": 10,
            "grid": {"kappa_vol": [-0.1, 0.0, 0.1],
                     "kappa_tail": [-0.5, 0.0]}}
    out = expand_sweep(spec)
    assert len(out) == 6
    assert all(v["name"].startswith("vol_") for v in out)
    assert len({v["name"] for v in out}) == 6


def test_sweep_requires_a_name_prefix():
    from fvmodel.variants import expand_sweep
    with pytest.raises(ValueError, match="name_prefix"):
        expand_sweep({"max_variants": 4, "grid": {"kappa_vol": [0.0, 0.1]}})


def test_sweep_cap_is_enforced_before_anything_runs():
    from fvmodel.variants import expand_sweep
    with pytest.raises(ValueError, match="max_variants"):
        expand_sweep({"name_prefix": "big", "max_variants": 3,
                      "grid": {"kappa_vol": [0.0, 0.1, 0.2],
                               "tail_scale": [1.0, 1.1]}})
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "$INV" && python -m pytest tests/test_variants.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fvmodel.variants'`

- [ ] **Step 3: Write `fvmodel/variants.py`**

```python
"""A variant is a file. Reading one never runs anything and never fits anything."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import yaml

from .base import ROOT

VARIANTS = ROOT / "variants"


def _path(name: str) -> Path:
    for ext in (".yaml", ".yml", ".json"):
        p = VARIANTS / (name + ext)
        if p.exists():
            return p
    raise FileNotFoundError("no variant %r in %s; have %s"
                            % (name, VARIANTS, list_variants()))


def list_variants() -> list:
    return sorted({p.stem for p in VARIANTS.glob("*")
                   if p.suffix in (".yaml", ".yml", ".json")})


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
```

- [ ] **Step 4: Write the sixteen variant files**

Each is the same shape. `variants/baseline.yaml`:

```yaml
name: baseline
base_params: model.json@fv-1.0.0
overrides: {}
notes: >
  Every override at its default. Bit-identical to the shipped model, which
  tests/test_golden.py enforces. Every other variant is reported as a delta
  against this row.
```

`variants/market_observable.yaml`:

```yaml
name: market_observable
base_params: model.json@fv-1.0.0
overrides:
  information_set: prints_only
  reconstruct_in_transit: false
  alpha_scale: 0.0
notes: >
  The counterparty's information set: received Chainlink prints and nothing
  else - no exchange feed, so the quote origin steps back by the receive lag and
  every later print's forecast is the last one received. Expected to lose on
  every calibration metric, most of all in the last 30 s, and the size of that
  gap is the lag edge.
```

`variants/no_alpha.yaml`:

```yaml
name: no_alpha
base_params: model.json@fv-1.0.0
overrides: {alpha_scale: 0.0}
notes: >
  The order-book location term off. Should move m_Y to exactly zero and barely
  move pooled log-loss; if it moves it a lot, the alpha is doing more than the
  0.18 bp per unit of imbalance it was fitted at, which would be suspicious
  given the book captures do not overlap the Chainlink history.
```

`variants/alpha_x2.yaml`:

```yaml
name: alpha_x2
base_params: model.json@fv-1.0.0
overrides: {alpha_scale: 2.0}
notes: >
  Double the location term. If the fitted alpha is right this must be worse
  than baseline; if it is better, the term is under-scaled on this settlement.
```

`variants/alpha_half.yaml`:

```yaml
name: alpha_half
base_params: model.json@fv-1.0.0
overrides: {alpha_scale: 0.5}
notes: >
  Half the location term. Brackets alpha_x2, so the three alpha variants read as
  a crude one-dimensional profile of the right scale.
```

`variants/vol_plus10.yaml`:

```yaml
name: vol_plus10
base_params: model.json@fv-1.0.0
overrides: {kappa_vol: 0.095}
notes: >
  Volatility times exp(0.095) = 1.100, at every horizon. Expected to move QLIKE
  excess and E[resid^2/Var_Y] symmetrically against vol_minus10; if one side
  wins clearly, the shipped variance level is off in that direction.
```

`variants/vol_minus10.yaml`:

```yaml
name: vol_minus10
base_params: model.json@fv-1.0.0
overrides: {kappa_vol: -0.095}
notes: >
  Volatility times exp(-0.095) = 0.909. See vol_plus10.
```

`variants/short_vol_minus20.yaml`:

```yaml
name: short_vol_minus20
base_params: model.json@fv-1.0.0
overrides: {kappa_vol_short: -0.22}
notes: >
  Volatility times exp(-0.22) = 0.80 at the sub-minute end only, tapering as
  min(1, 60/D). Report section 7b found the short end can be off independently
  of the body, so this is expected to move the near-strike last-30 s cell and to
  leave the pooled numbers alone. If it moves both, the knob is not doing what
  it says.
```

`variants/shrink_05.yaml`:

```yaml
name: shrink_05
base_params: model.json@fv-1.0.0
overrides: {shrink_w: 0.5, shrink_decay_s: 60.0}
notes: >
  At short business age, shrink log xi half-way to the seasonal-unconditional
  level, decaying with a 60 s business-time constant. A different remedy for the
  same complaint as short_vol_minus20 - it shrinks toward a level rather than
  scaling - so the interesting comparison is between those two, not against
  baseline.
```

`variants/fat_tails.yaml`:

```yaml
name: fat_tails
base_params: model.json@fv-1.0.0
overrides: {kappa_tail: -0.5}
notes: >
  nu' = 2 + (nu-2)*exp(-0.5), with sigma rescaled so the standardised variance is
  unchanged. Pure shape: only the far quantiles move, so this should show up in
  log-loss on the confident quotes and not in QLIKE.
```

`variants/thin_tails.yaml`:

```yaml
name: thin_tails
base_params: model.json@fv-1.0.0
overrides: {kappa_tail: 0.5}
notes: >
  The other side of fat_tails. The shipped tails are already fitted, so both
  should lose; if one wins, the tail fit is biased in that direction on the 5 m
  window, which is a different sample from the one it was fitted on.
```

`variants/normal_tail.yaml`:

```yaml
name: normal_tail
base_params: model.json@fv-1.0.0
overrides: {tail_family: normal}
notes: >
  Replace the fitted Student-t with a normal. Expected to be clearly worse near
  expiry, where the fitted nu approaches its 2.5 floor, and almost indistinguishable
  at long horizons. The size of the near-expiry gap is what the tail is worth.
```

`variants/eps_x0.yaml`:

```yaml
name: eps_x0
base_params: model.json@fv-1.0.0
overrides: {eps_scale: 0.0}
notes: >
  The fast Chainlink residual removed entirely - variance and conditional mean
  both (spec ruling R1). Expected to under-state Var_Y in the last ten seconds,
  where the residual is a material share of it, and so to show up as
  E[resid^2/Var_Y] above one in the near-strike last-30 s cell.
```

`variants/eps_x2.yaml`:

```yaml
name: eps_x2
base_params: model.json@fv-1.0.0
overrides: {eps_scale: 2.0}
notes: >
  Double the residual process. Brackets eps_x0; both should lose if the fitted
  sigma_eps is right.
```

`variants/basis_alt.yaml`:

```yaml
name: basis_alt
base_params: model.json@fv-1.0.0
overrides: {basis_tracker: alt}
notes: >
  The 15 s basis tracker instead of the shipped 60 s one. NOTES E19 found 15 s is
  the constrained optimum of the extended sweep, and the shipped 60 s was a
  boundary optimum, so this is the open question that a live comparison was meant
  to settle. Expected to move the level terms (m_Y is untouched; eps_bar and the
  carry are not) rather than the variance.
```

`variants/rho_off.yaml`:

```yaml
name: rho_off
base_params: model.json@fv-1.0.0
overrides: {rho_kernel: "off"}
notes: >
  Set the return autocorrelation kernel to a delta. Removes the 1.18-1.57
  variance inflation the kernel supplies, so Var_Y falls and
  E[resid^2/Var_Y] should rise above one across the board. A large move here is
  the kernel earning its place.
```

- [ ] **Step 5: Write the sweep example**

`variants/sweeps/example.yaml`:

```yaml
name_prefix: volsweep
max_variants: 12
base_params: model.json@fv-1.0.0
notes: A two-dimensional sweep of the level and the short end.
grid:
  kappa_vol: [-0.095, 0.0, 0.095]
  kappa_vol_short: [-0.22, 0.0]
```

- [ ] **Step 6: Run the variants tests**

Run: `cd "$INV" && python -m pytest tests/test_variants.py -v`
Expected: 22 PASS.

- [ ] **Step 7: Write `run_variants.py`**

```python
"""Build and score variants.

    python run_variants.py --all
    python run_variants.py --variants baseline,rho_off
    python run_variants.py --sweep variants/sweeps/example.yaml
    python run_variants.py --variants baseline --holdout --i-know

A variant run is pure evaluation: nothing is fitted, and the register bank is
shared across every variant (see export/cache.py).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

INV = Path(__file__).resolve().parent
if str(INV) not in sys.path:
    sys.path.insert(0, str(INV))

from export.build_export import build            # noqa: E402
from fvmodel import config                       # noqa: E402
from fvmodel.variants import expand_sweep, list_variants  # noqa: E402

HOLDOUT_LOG = INV / "runs" / "holdout_log.tsv"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(INV), text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    w = config.read_config()["window"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sweep", default="")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--i-know", dest="i_know", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args()

    if a.holdout and not a.i_know:
        raise SystemExit(
            "--holdout scores the reserved window 2026-08-26 .. 08-31. It is for the "
            "ONE chosen variant, once, at the end. If that is what you are doing, "
            "pass --i-know as well; the run is logged to runs/holdout_log.tsv.")

    if a.sweep:
        names = []
        spec = yaml.safe_load(Path(a.sweep).read_text())
        for v in expand_sweep(spec):
            p = INV / "variants" / ("%s.yaml" % v["name"])
            p.write_text(yaml.safe_dump(v, sort_keys=False))
            names.append(v["name"])
    elif a.all:
        names = list_variants()
    else:
        names = [s for s in a.variants.split(",") if s]
    if not names:
        raise SystemExit("nothing to run; pass --variants, --all or --sweep")

    t0 = w["holdout_t0"] if a.holdout else w["backtest_t0"]
    t1 = w["backtest_t1"] if a.holdout else w["select_t1"]
    tag = "holdout" if a.holdout else "select"
    print("window %s: %d .. %d, %d variant(s)" % (tag, t0, t1, len(names)))

    # baseline first, and always: every other variant is scored on ITS strike grid
    # and reported as a delta against it
    names = ["baseline"] + [n for n in names if n != "baseline"]

    results, ref_path = {}, None
    for name in names:
        start = time.time()
        path = build(name, t0, t1,
                     out_path=INV / "runs" / tag / ("%s.parquet" % name))
        took = time.time() - start
        if name == "baseline":
            ref_path = path
        print("  %-22s built in %.1f s" % (name, took))
        if not a.no_score:
            from score.scorecard import score_variant
            results[name] = score_variant(name, path, ref_path=ref_path)
            results[name]["build_seconds"] = round(took, 1)

    out = INV / "runs" / ("scores_%s.json" % tag)
    out.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print("wrote %s" % out)

    if a.holdout:
        HOLDOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
        new = not HOLDOUT_LOG.exists()
        with open(HOLDOUT_LOG, "a", encoding="utf-8") as fh:
            if new:
                fh.write("utc\tgit_sha\tvariant\tt0\tt1\tlog_loss\tnear30_log_loss\n")
            for name, r in results.items():
                fh.write("%s\t%s\t%s\t%d\t%d\t%.6f\t%.6f\n" % (
                    dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    _git_sha(), name, t0, t1,
                    r.get("log_loss", float("nan")),
                    r.get("near30_log_loss", float("nan"))))
        print("logged %d holdout run(s) to %s" % (len(results), HOLDOUT_LOG))


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Write `variants/README.md`**

Brief §4.5. It must give, per override, the sign convention, the typical range,
and the metric it is expected to move.

```markdown
# Variants

A variant is a file: `{name, base_params, overrides, notes}`. Reading one runs
nothing. `python run_variants.py --all` builds and scores every one of them.

Every override defaults to its off value, and with all defaults the model is
bit-identical to `fv-1.0.0` — `tests/test_golden.py` enforces that.

## The overrides

| key | sign convention | typical range | expected to move |
|---|---|---|---|
| `kappa_vol` | positive widens; vol × exp(κ) | −0.2 … +0.2 | QLIKE excess, `E[resid²/Var_Y]`, pooled log-loss |
| `kappa_vol_short` | as above, sub-minute only, weight `min(1, 60/D)` | −0.4 … +0.2 | the near-strike last-30 s cell, and little else |
| `shrink_w` | positive shrinks toward the unconditional level | 0 … 1 | the same cell as `kappa_vol_short`, by a different mechanism |
| `shrink_decay_s` | larger = the shrink reaches further out | 15 … 300 | how far up the horizon the shrink is felt |
| `rho_kernel` | `off` removes the variance inflation | — | `E[resid²/Var_Y]` upward when off |
| `eps_scale` | scales the ε **process**, mean and sd (R1); 0 = off | 0 … 2 | Var_Y in the last ~10 s; the near-strike cell |
| `eps_condition` | false forgets the last observed residual | — | `eps_bar`, and log-loss near expiry |
| `basis_tracker` | `main` 60 s, `alt` 15 s, `off` no basis | — | the level terms, not the variance |
| `information_set` | `prints_only` is the counterparty (R9) | — | everything; this is the lag edge |
| `reconstruct_in_transit` | false stops reconstructing stamped-unreceived prints | — | the carry, in the last few seconds |
| `kappa_tail` | **negative = fatter**; Var(z) held fixed | −1 … +1 | log-loss on confident quotes; not QLIKE |
| `tail_scale` | multiplies μ and σ of the t together (R2) | 0.8 … 1.25 | the same as `kappa_vol`, at the other end |
| `tail_family` | `normal` drops the fitted t | — | near-expiry log-loss |
| `alpha_scale` | 0 = alpha off | 0 … 2 | `m_Y`; small effect on pooled log-loss |
| `alpha_scale_age` | scales the quote-age interaction only | 0 … 2 | `m_Y` on stale quotes |
| `alpha_cap_sd` | caps \|m_Y\| in settlement sd; null = uncapped | 0.5 … 3 | the tail of `m_Y`, not its centre |
| `w_spot` | blend weight; τ, δ re-read, never refitted | grid only | the whole print model |
| `xi_cap_c` | the short-business-time cap; null = off | 1.5 … 4 | the forward curve during vol bursts |
| `temperature` | > 1 pulls toward a half; quoting layer only | 0.8 … 1.5 | `p_quoted` **only** — never `p_model` |

## Two traps

**`w_spot: 0.6` is not baseline.** The per-`w` table is the coarse grid, so it
gives τ = 0.7872 where the shipped fine-refined value is τ = 0.8447. Leave
`w_spot` unset to get the shipped filter. (Spec ruling R11.)

**`temperature` never touches `p_model`.** It is reported as `p_quoted` beside it
precisely so it cannot pollute a calibration statistic. If you see it move
log-loss, something has wired it into the wrong column.

## Sweeps

`variants/sweeps/*.yaml` take `{name_prefix, max_variants, grid}` and expand to
the product. `name_prefix` is required and `max_variants` is checked **before**
expansion, because a four-key grid is easy to write and expensive to discover by
running it.
```

- [ ] **Step 9: Commit**

```bash
cd /c/Users/kaima/Github2/Gambling104
git add investigations/2026-9-9_business_clock_t_dist
git commit -m "feat: the shipped variant set, the sweep spec and the runner"
```

---

## Task 10: Scorecard and report

**Files:**
- Create: `score/__init__.py`, `score/scorecard.py`, `score/report_variants.py`, `tests/test_scorecard.py`
- Create (generated): `report_variants.html`

**Interfaces:**
- Consumes: an export parquet from Task 7, `harness_paths.STRIKES`, `chainlink_1s.parquet`.
- Produces:
  - `score.scorecard.settlements(t0, t1) -> pl.DataFrame` — `market_id, open_ts, strike, settle, up`
  - `score.scorecard.log_loss(p, up) -> float`, `brier(p, up) -> float`
  - `score.scorecard.proxy_pnl(p_model, p_market, up, edge, fee_rate=0.07, taker_rebate=0.0833) -> dict` with `n_trades, total, mean, sd, sharpe, pnl, take`
  - `score.scorecard.book_mid(d: pl.DataFrame) -> np.ndarray`
  - `score.scorecard.score_variant(name: str, export_path: Path, ref: pl.DataFrame | None = None) -> dict` with keys `log_loss`, `brier`, `log_loss_grid`, `brier_grid`, `near30_log_loss`, `near30_brier`, `near30_n`, `qlike_excess_by_tte`, `level_by_tte`, `reliability`, `pnl`, `n_rows`, `n_markets`, `excluded_fraction`. `ref` is the **baseline's** joined frame, so the strike grid is placed on the baseline's sd and every variant answers the same question.
  - `score.report_variants.main()` → `report_variants.html`

- [ ] **Step 1: Write the failing scorecard tests**

Create `tests/test_scorecard.py`:

```python
from __future__ import annotations

import numpy as np
import pytest


def test_log_loss_is_the_mean_negative_log_likelihood():
    from score.scorecard import log_loss

    p = np.array([0.9, 0.1])
    up = np.array([1.0, 0.0])
    assert log_loss(p, up) == pytest.approx(-np.log(0.9))
    assert log_loss(np.array([0.5, 0.5]), up) == pytest.approx(np.log(2.0))


def test_log_loss_clips_rather_than_returning_inf():
    from score.scorecard import log_loss
    assert np.isfinite(log_loss(np.array([0.0]), np.array([1.0])))


def test_pnl_only_trades_past_the_edge_and_sizes_by_the_gap():
    from score.scorecard import proxy_pnl

    p_model = np.array([0.60, 0.51, 0.40])
    p_market = np.array([0.50, 0.50, 0.50])
    up = np.array([1.0, 1.0, 0.0])
    out = proxy_pnl(p_model, p_market, up, edge=0.05, fee_rate=0.0)
    assert out["n_trades"] == 2, "the 0.01 gap must not trade at edge 0.05"
    # trade 1: buy UP 0.10 shares at 0.50, settles 1 -> +0.05
    # trade 3: sell UP 0.10 shares at 0.50, settles 0 -> +0.05
    assert out["total"] == pytest.approx(0.10)


def test_pnl_charges_the_fee_on_both_sides():
    from score.scorecard import proxy_pnl

    p_model = np.array([0.60])
    p_market = np.array([0.50])
    up = np.array([1.0])
    free = proxy_pnl(p_model, p_market, up, edge=0.05, fee_rate=0.0)["total"]
    paid = proxy_pnl(p_model, p_market, up, edge=0.05, fee_rate=0.07,
                     taker_rebate=0.0)["total"]
    # base fee = 0.07 * p * (1-p) * shares = 0.07 * 0.5 * 0.5 * 0.10
    assert free - paid == pytest.approx(0.07 * 0.5 * 0.5 * 0.10)
    rebated = proxy_pnl(p_model, p_market, up, edge=0.05, fee_rate=0.07,
                        taker_rebate=0.0833)["total"]
    assert rebated > paid, "the Silver-tier taker rebate must reduce the fee"


@pytest.mark.slow
def test_settlement_matches_the_venue_strike_convention():
    """The realised 60 s TWAP we compute must agree with the venue's own strike
    for the NEXT market, which opens at the same instant this one settles."""
    import polars as pl

    from score.scorecard import settlements
    from harness_paths import STRIKES

    t0, t1 = 1786665600, 1786665600 + 6 * 3600
    s = settlements(t0, t1)
    k = pl.read_parquet(STRIKES).select(["market_id", "open_ts", "strike"])
    j = s.join(k, on="market_id").sort("open_ts")
    nxt = j.with_columns((pl.col("open_ts") + 300).alias("next_open"))
    m = nxt.join(k.rename({"open_ts": "next_open", "strike": "next_strike"}),
                 on="next_open")
    rel = ((m["settle"] - m["next_strike"]) / m["next_strike"]).abs()
    assert float(rel.median()) < 1e-4, (
        "our 60 s TWAP disagrees with the venue's next-market strike by %.2e"
        % float(rel.median()))


@pytest.mark.slow
def test_score_variant_returns_every_headline():
    from pathlib import Path

    from score.scorecard import score_variant
    from harness_paths import FAIR_DIR

    r = score_variant("baseline", Path(FAIR_DIR) / "baseline.parquet")
    for k in ("log_loss", "brier", "log_loss_grid", "near30_log_loss", "near30_n",
              "n_markets", "reliability", "pnl", "level_by_tte"):
        assert k in r, "missing %s" % k
    assert 0.0 < r["log_loss"] < np.log(2.0) * 1.5
    assert r["near30_n"] > 100
    # the near-strike last-30 s cell is the hard question: it must be close to
    # coin-flip, and a value far below log(2) means the cell is not what it says
    assert 0.5 < r["near30_log_loss"] < np.log(2.0) * 1.1
    assert set(r["pnl"]) == {"edge_0.02", "edge_0.05", "edge_0.10"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "$INV" && python -m pytest tests/test_scorecard.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'score'`

- [ ] **Step 3: Write `score/scorecard.py`**

```python
"""Calibration on the panel's real markets.

Interim, until the execution harness can rank variants by PnL. It scores the export
against real strikes and the realised 60 s Chainlink TWAP, which we compute ourselves
rather than take on trust - and which therefore also cross-checks the venue's own
resolution (see tests/test_scorecard.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from fvmodel import config
from harness_paths import STRIKES

MARKET_LEN = 300
TWAP_LEN = 60
GRID = (-1.0, -0.5, 0.5, 1.0)
EDGES = (0.02, 0.05, 0.10)
BASE_FEE_RATE = 0.07
EPS = 1e-9


def log_loss(p, up) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    up = np.asarray(up, dtype=np.float64)
    return float(-np.mean(up * np.log(p) + (1 - up) * np.log(1 - p)))


def brier(p, up) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(up)) ** 2))


def settlements(t0: int, t1: int) -> pl.DataFrame:
    """market_id, strike, settle, up - the realised 60 s Chainlink TWAP."""
    src = config.source_root()
    cl = pl.read_parquet(src / "data" / "chainlink" / "chainlink_1s.parquet")
    ts = cl["ts"].to_numpy().astype(np.int64)
    px = cl["cl"].to_numpy().astype(np.float64)
    grid = np.full(int(ts.max() - ts.min() + 1), np.nan)
    grid[ts - ts.min()] = px
    # a null second means no NEW mark, not a silent feed: a settlement averages the
    # feed's value each second, which is the step function
    ok = np.isfinite(grid)
    idx = np.where(ok, np.arange(grid.size), 0)
    np.maximum.accumulate(idx, out=idx)
    step = grid[idx]
    step[: int(np.argmax(ok))] = np.nan
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(step))])
    cnt = np.concatenate([[0], np.cumsum(ok[idx].astype(np.int64))])

    k = pl.read_parquet(STRIKES).filter(
        (pl.col("open_ts") >= t0) & (pl.col("open_ts") + MARKET_LEN <= t1)).sort("open_ts")
    end = (k["open_ts"].to_numpy() + MARKET_LEN - ts.min()).astype(np.int64)
    good = (end >= TWAP_LEN) & (end < step.size)
    settle = np.full(end.size, np.nan)
    full = (cnt[end[good] + 1] - cnt[end[good] + 1 - TWAP_LEN]) == TWAP_LEN
    v = (csum[end[good] + 1] - csum[end[good] + 1 - TWAP_LEN]) / TWAP_LEN
    settle[np.where(good)[0][full]] = v[full]
    return k.with_columns(pl.Series("settle", settle)).with_columns(
        (pl.col("settle") > pl.col("strike")).alias("up")).drop_nulls("settle")


def _p_at(z, nu, mu, sg):
    return stats.t.cdf((z + mu) / np.maximum(sg, 1e-12), df=nu)


def proxy_pnl(p_model, p_market, up, edge: float, fee_rate: float = BASE_FEE_RATE,
              taker_rebate: float = 0.0833) -> dict:
    """Buy the side the model likes when the gap clears `edge`, sized by the gap.

    Every trade is a taker at the market price. This is a proxy, not the harness:
    no queue, no latency, no inventory. It exists so a variant can be sanity-checked
    against a real price series before the execution harness lands.
    """
    gap = np.asarray(p_model, dtype=np.float64) - np.asarray(p_market, dtype=np.float64)
    take = np.abs(gap) > edge
    if not take.any():
        return {"n_trades": 0, "total": 0.0, "mean": 0.0, "sd": 0.0, "sharpe": 0.0}
    g = gap[take]
    pm = np.asarray(p_market, dtype=np.float64)[take]
    u = np.asarray(up, dtype=np.float64)[take]
    shares = np.abs(g)                                   # size proportional to the gap
    side = np.sign(g)                                    # +1 buy UP, -1 sell UP
    payoff = side * shares * (u - pm)
    fee = fee_rate * pm * (1.0 - pm) * shares * (1.0 - taker_rebate)
    p = payoff - fee
    sd = float(np.std(p))
    return {"n_trades": int(take.sum()), "total": float(p.sum()),
            "mean": float(np.mean(p)), "sd": sd,
            "sharpe": float(np.mean(p) / sd * np.sqrt(p.size)) if sd > 0 else 0.0,
            "pnl": p, "take": take}


def score_variant(name: str, export_path: Path, ref_path: Path = None) -> dict:
    df = pl.read_parquet(export_path).filter(pl.col("ok"))
    t0 = int(df["open_ts"].min())
    t1 = int(df["open_ts"].max()) + MARKET_LEN
    st = settlements(t0, t1).select(["market_id", "strike", "settle", "up"])
    d = df.join(st, on="market_id")
    if ref_path is not None and Path(ref_path) != Path(export_path):
        # the strike grid goes on the BASELINE's sd, so every variant is asked the
        # same question; a grid at each model's own sd asks the wider model an
        # easier one and the wider model then appears to win
        r = (pl.read_parquet(ref_path).filter(pl.col("ok"))
             .select(["market_id", "t_s", pl.col("sigma").alias("sigma_ref")]))
        d = d.join(r, on=["market_id", "t_s"], how="inner")
    else:
        d = d.with_columns(pl.col("sigma").alias("sigma_ref"))
    excluded = 1.0 - len(d) / max(len(df), 1)

    s = d["s"].to_numpy()
    sg = d["sigma"].to_numpy()
    K = d["strike"].to_numpy()
    nu, mu, sgt = (d[c].to_numpy() for c in ("nu", "mu", "sigma_t"))
    up = d["up"].to_numpy().astype(np.float64)
    tte = MARKET_LEN - d["t_s"].to_numpy()
    z = (s - K) / np.maximum(sg, 1e-300)
    p = _p_at(z, nu, mu, sgt)

    out = {"variant": name, "n_rows": len(d),
           "n_markets": int(d["market_id"].n_unique()),
           "excluded_fraction": float(excluded),
           "log_loss": log_loss(p, up), "brier": brier(p, up)}

    sd_ref = d["sigma_ref"].to_numpy()
    ll, br = [], []
    for c in GRID:
        Kc = s - c * sd_ref
        pc = _p_at((s - Kc) / np.maximum(sg, 1e-300), nu, mu, sgt)
        uc = (d["settle"].to_numpy() > Kc).astype(np.float64)
        ll.append(log_loss(pc, uc))
        br.append(brier(pc, uc))
    out["log_loss_grid"] = float(np.mean(ll))
    out["brier_grid"] = float(np.mean(br))

    # the cell the brief singles out: near the strike, in the last thirty seconds
    near = (np.abs(z) < 1.0) & (tte <= 30)
    out["near30_n"] = int(near.sum())
    out["near30_log_loss"] = log_loss(p[near], up[near]) if near.sum() > 30 else float("nan")
    out["near30_brier"] = brier(p[near], up[near]) if near.sum() > 30 else float("nan")

    # variance calibration by remaining time
    resid = (d["settle"].to_numpy() - s) / np.maximum(d["p_ref"].to_numpy()
                                                      * d["omega"].to_numpy(), 1e-300)
    var_y = (sg / np.maximum(d["p_ref"].to_numpy() * d["omega"].to_numpy(),
                             1e-300)) ** 2
    bins = [(1, 5), (6, 15), (16, 30), (31, 60), (61, 120), (121, 300)]
    lvl, qk = {}, {}
    for lo, hi in bins:
        m = (tte >= lo) & (tte <= hi)
        if m.sum() < 200:
            continue
        r2 = resid[m] ** 2
        v = np.maximum(var_y[m], 1e-30)
        lvl["%d-%d" % (lo, hi)] = float(np.mean(r2) / np.mean(v))
        nz = r2 > 0
        ratio = r2[nz] / v[nz]
        qk["%d-%d" % (lo, hi)] = float(np.mean(ratio - np.log(ratio) - 1.0)
                                       - 1.2704034809047095)
    out["level_by_tte"], out["qlike_excess_by_tte"] = lvl, qk

    # reliability by (tte bucket, moneyness bucket)
    rel = []
    for lo, hi in bins:
        for zlo, zhi in ((-99, -1), (-1, -0.25), (-0.25, 0.25), (0.25, 1), (1, 99)):
            m = (tte >= lo) & (tte <= hi) & (z >= zlo) & (z < zhi)
            if m.sum() < 100:
                continue
            rel.append({"tte": "%d-%d" % (lo, hi), "z": "%g..%g" % (zlo, zhi),
                        "n": int(m.sum()), "p_mean": float(p[m].mean()),
                        "freq_up": float(up[m].mean())})
    out["reliability"] = rel

    # the trading proxy, against the panel's real book mid
    mid = book_mid(d)
    have = np.isfinite(mid)
    out["pnl"] = {}
    for e in EDGES:
        r = proxy_pnl(p[have], mid[have], up[have], edge=e)
        frac = 0.0
        if r["n_trades"] and r["total"]:
            frac = float(r["pnl"][tte[have][r["take"]] <= 30].sum() / r["total"])
        out["pnl"]["edge_%.2f" % e] = dict(
            {k: r[k] for k in ("n_trades", "total", "mean", "sd", "sharpe")},
            frac_last30=frac)
    return out


def book_mid(d: pl.DataFrame) -> np.ndarray:
    """The UP token's book mid at each row's own bucket, from the 5 m panel.

    The panel is 100 ms and carries no row for a bucket nobody observed, so this
    joins on (market_id, the first bucket the row is usable in) and leaves NaN where
    the book was not seen. Nothing is forward-filled: a stale book is worse than a
    missing one, and the proxy simply does not trade where it cannot see a price.
    """
    from export.build_export import usable_t_ms
    from harness_paths import PANEL

    key = d.select(["market_id", "t_s"]).with_columns(
        pl.col("t_s").map_elements(usable_t_ms, return_dtype=pl.Int64).alias("t_ms"))
    panel = (pl.scan_parquet(PANEL).select(["market_id", "t_ms", "mid"])
             .collect())
    j = key.join(panel, on=["market_id", "t_ms"], how="left")
    return j["mid"].to_numpy().astype(np.float64)
```

- [ ] **Step 4: Run the fast scorecard tests**

Run: `cd "$INV" && python -m pytest tests/test_scorecard.py -v -m "not slow"`
Expected: 4 PASS.

- [ ] **Step 5: Run the slow scorecard tests**

Run: `cd "$INV" && python -m pytest tests/test_scorecard.py -v`
Expected: 6 PASS. `test_settlement_matches_the_venue_strike_convention` is the
one that matters: it checks our realised TWAP against the venue's own next-market
strike. If it fails, do not adjust the tolerance — the settlement convention is
wrong and every calibration number downstream would be wrong with it.

- [ ] **Step 6: Write `score/report_variants.py`**

```python
"""report_variants.html: baseline and every shipped variant, on the selection window.

Sorted by the near-strike last-30 s log-loss with the pooled log-loss beside it, which
is the brief's ordering: the cell where the model is actually asked a hard question,
with the pooled number next to it so a variant that wins the cell by losing everywhere
else is visible.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

INV = Path(__file__).resolve().parents[1]
if str(INV) not in sys.path:
    sys.path.insert(0, str(INV))

from fvmodel.variants import read_variant           # noqa: E402

SCORES = INV / "runs" / "scores_select.json"
OUT = INV / "report_variants.html"

CSS = """
:root { color-scheme: light dark; --fg:#1a1a1a; --bg:#fff; --line:#d8d8d8;
        --muted:#666; --good:#0a7; --bad:#c33; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e8; --bg:#141414; --line:#333; --muted:#999; } }
body { margin:0 auto; padding:2rem 1.25rem; max-width:70rem; color:var(--fg);
       background:var(--bg); font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif; }
h1 { font-size:1.5rem; margin:0 0 .25rem; }
.sub { color:var(--muted); margin:0 0 2rem; }
.wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th,td { padding:.4rem .6rem; border-bottom:1px solid var(--line); text-align:right;
        white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
thead th { border-bottom:2px solid var(--line); font-weight:600; }
tr.base { font-weight:600; }
.d-good { color:var(--good); } .d-bad { color:var(--bad); }
.note { margin:.4rem 0 1.4rem; color:var(--muted); max-width:60rem; }
h3 { margin:1.6rem 0 .2rem; font-size:1rem; }
"""


def _delta(v, b, lower_is_better=True):
    if v != v or b != b:
        return ""
    d = v - b
    cls = "d-good" if (d < 0) == lower_is_better and d != 0 else (
        "d-bad" if d != 0 else "")
    return '<span class="%s">%+.4f</span>' % (cls, d)


def main() -> None:
    scores = json.loads(SCORES.read_text())
    base = scores["baseline"]
    rows = sorted(scores.items(),
                  key=lambda kv: (kv[1].get("near30_log_loss") or 9e9))

    head = ("<tr><th>variant</th><th>near-strike last 30 s</th><th>Δ</th>"
            "<th>pooled log-loss</th><th>Δ</th><th>grid log-loss</th>"
            "<th>Brier</th><th>markets</th><th>build s</th></tr>")
    body = []
    for name, r in rows:
        body.append(
            '<tr class="%s"><td>%s</td><td>%.5f</td><td>%s</td><td>%.5f</td>'
            '<td>%s</td><td>%.5f</td><td>%.5f</td><td>%d</td><td>%.0f</td></tr>'
            % ("base" if name == "baseline" else "", html.escape(name),
               r["near30_log_loss"], _delta(r["near30_log_loss"],
                                            base["near30_log_loss"]),
               r["log_loss"], _delta(r["log_loss"], base["log_loss"]),
               r["log_loss_grid"], r["brier"], r["n_markets"],
               r.get("build_seconds", 0)))

    notes = []
    for name, r in rows:
        if name == "baseline":
            continue
        v = read_variant(name)
        d_near = r["near30_log_loss"] - base["near30_log_loss"]
        d_pool = r["log_loss"] - base["log_loss"]
        notes.append(
            "<h3>%s</h3><p class='note'><em>Expected:</em> %s<br><em>Observed:</em> "
            "near-strike last-30 s log-loss %+.5f, pooled %+.5f against baseline. "
            "Variance level by remaining time: %s.</p>"
            % (html.escape(name), html.escape(v["notes"].strip()), d_near, d_pool,
               html.escape(json.dumps({k: round(x, 3)
                                       for k, x in r["level_by_tte"].items()}))))

    OUT.write_text(
        "<title>Fair-value variants</title><style>%s</style>"
        "<h1>Fair-value variants</h1>"
        "<p class='sub'>Selection window 2026-08-14 to 2026-08-25, real 5 m markets "
        "and real strikes. The holdout 08-26 to 08-31 is not scored here. Sorted by "
        "near-strike last-30 s log-loss; lower is better throughout.</p>"
        "<div class='wrap'><table><thead>%s</thead><tbody>%s</tbody></table></div>"
        "<h2>Per variant</h2>%s"
        % (CSS, head, "".join(body), "".join(notes)), encoding="utf-8")
    print("wrote %s: %d variants" % (OUT, len(rows)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the whole shipped set**

Run: `cd "$INV" && python run_variants.py --all`
Expected: 16 variants built and scored, `runs/scores_select.json` written. Total
runtime should be well under an hour; the register bank is swept once.

- [ ] **Step 8: Generate the report**

Run: `cd "$INV" && python -m score.report_variants`
Expected: `wrote .../report_variants.html: 16 variants`

- [ ] **Step 9: Read the report and check each variant against its README expectation**

Open `report_variants.html`. For each variant, confirm the observed direction
matches the `notes` field. Where it does not, **do not adjust the variant** —
write the disagreement into the note, because a knob that does not do what it
says is the finding.

Particularly check: `temperature` is absent from every calibration column (it is
a `p_quoted` knob only); `no_alpha` moves pooled log-loss very little; `rho_off`
pushes `level_by_tte` above one across the board.

- [ ] **Step 10: Run the full test suite and commit**

Run: `cd "$INV" && python -m pytest -v`
Expected: everything PASS.

```bash
cd /c/Users/kaima/Github2/Gambling104
git add investigations/2026-9-9_business_clock_t_dist
git commit -m "feat: calibration scorecard on real markets, and the variant report"
```

---

## Open item for the harness team

`link` is contracted as a pure `link(z) -> p`, but the settlement tail's
`(ν, μ, σ_t)` are a function of business time left and are therefore per-tick —
the same way `f` is already bound to `strike` and `sigma[i]`. `link.py` ships
`bind(ep)` for this. Raise it before the harness freezes its block API.

Separately, `backtesting_5m/harness/paths.py` sets
`INVESTIGATIONS = backtesting_5m/investigations`, but this folder is at the repo
root under `investigations/`. Either the resolver takes an explicit path, or this
folder moves. Exports are already written to the harness's own `FAIR_DIR`, so
only block resolution is affected.
