# Stream Registry and Usable Module — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an investigation reference tick data — including streams recorded after the harness was written — without editing the harness.

**Architecture:** Streams become self-describing parquets carrying their own metadata; a registry resolves them investigation-first, catalog-second; the Episode exposes them through an accessor rather than accreting typed fields; blocks resolve against shared model directories; one public `backtest()` entry point; `harness` becomes an installed package so research in `../investigations` imports it directly.

**Tech Stack:** Python 3.13, numpy 2.2, pandas 2.2, pyarrow 19, polars 1.40 (soft fallback), pytest 8.4. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-09-stream-registry-and-module-design.md` — read it before Task 1.

## Global Constraints

- **Working directory** is `C:\Users\kaima\Github2\Gambling104\backtesting_5m`. Git root is `C:\Users\kaima\Github2\Gambling104`.
- **Run tests with** `python -m pytest harness/tests -q` from the working directory. **216 pass today**; every task must keep them passing.
- **A second Claude session shares this branch.** Never touch `Gambling104/investigations/2026-09-09_normal_qq_basic/` or `Gambling104/investigations/2026-9-9_business_clock_t_dist/`.
- **Never modify** `data/scripts/`, `data/book_5m_100ms.parquet`, `data/strikes_5m.parquet`, or any existing `data/spot_*.parquet`. New files under `data/` are fine.
- **Reserved stream columns:** `recv_ns` (int64, required, epoch ns, receipt on this host), `src_ns` (int64, required when the source provides one), `venue` (str, optional), `seq` (int64, optional). **Every other column is a value column.**
- **Alignment is always `recv_ns`, never `src_ns`.** A source stamp can lead receipt — `rtds_btc`'s `oracle_ms` leads by ~1.5 s — so aligning on it is lookahead.
- **Causality rule unchanged:** decision index `i` may only use observations knowable at `t_ms = i*100`; everything causal routes through `shift_to_decision_grid`.
- **Fee constants unchanged:** `BASE_FEE_RATE = 0.07`, `MAKER_REBATE_PHI = 0.20`, `TAKER_REBATE_RHO = 0.0833`. Tick 0.01. `N = 3000`, bucket 100 ms.
- **Migration is additive.** Old constants keep working until nothing imports them.
- Use `harness.io.read_parquet` for all reads (pyarrow 19 cannot read these Arrow-24 files; it falls back to polars).
- Test style follows `data/tests/test_build.py`: module docstring, plain pytest functions, sentence-style names, `tmp_path`, no classes.
- **Every commit message ends with:**
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01ED1yvqexsuevgcrrK16ymZ
  ```
  Written in full in Task 1; later tasks abbreviate as `<trailer>` — always write it out.

---

## File Structure

```
backtesting_5m/
  pyproject.toml                  NEW — makes `harness` installable
  harness/
    streams/                      NEW package
      __init__.py                 re-exports Stream, TimeKind, register, resolve
      spec.py                     Stream, TimeKind, STREAM_META_PREFIX, reserved cols
      writer.py                   write_stream()
      validate.py                 validate_stream()
      reader.py                   read + grid one stream onto (open_ts, t_ms)
      registry.py                 register(), resolve(), StreamNotRegistered
      catalog.py                  the standard streams
    core/
      episode.py                  MODIFY — t_ms array, stream() accessor, StreamView
      config.py                   MODIFY — Sample.require
      run.py                      MODIFY — drop counts in summary
      api.py                      NEW — backtest(), BacktestResult
    build/episodes.py             MODIFY — registry-driven joins, groupby-once
    paths.py                      MODIFY — constants become catalog wrappers
models/normal_qq/                 NEW — fair/vol/f/link, one canonical copy
investigations/                   research moves here from backtesting_5m/
```

---

## Task 1: Make `harness` an installed package

Everything else assumes `import harness` works from `../investigations`.

**Files:**
- Create: `pyproject.toml`
- Test: `harness/tests/test_packaging.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installed `harness` distribution; `harness.__version__`.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_packaging.py`:

```python
"""Pins that `harness` is importable as an installed package.

Investigations live at the repository root, two directories away from this
module. The sys.path.insert every runner used to carry was depth-dependent: it
resolved correctly from backtesting_5m/investigations/x/ and silently resolved
to the WRONG directory from investigations/x/ -- and the block resolver does not
error on that, it falls back to defaults and quietly evaluates a different
model. An installed package removes the failure mode.
"""
import subprocess
import sys

import harness


def test_harness_exposes_a_version():
    assert isinstance(harness.__version__, str)
    assert harness.__version__


def test_harness_imports_from_an_unrelated_working_directory(tmp_path):
    """The real test: no sys.path help, cwd nowhere near the module."""
    out = subprocess.run(
        [sys.executable, "-c", "import harness; print(harness.__version__)"],
        cwd=str(tmp_path), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == harness.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_packaging.py -q`
Expected: FAIL — `AttributeError: module 'harness' has no attribute '__version__'`

- [ ] **Step 3: Write minimal implementation**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "gambling104-harness"
version = "0.1.0"
description = "Backtesting harness for the BTC up/down 5 m book"
requires-python = ">=3.13"
dependencies = ["numpy>=2.2", "pandas>=2.2", "pyarrow>=19", "matplotlib>=3.10"]

[project.optional-dependencies]
fallback = ["polars>=1.40"]

[tool.setuptools.packages.find]
include = ["harness*"]

[tool.pytest.ini_options]
testpaths = ["harness/tests"]
```

Set the version in `harness/__init__.py`:

```python
"""Backtesting harness for the BTC up/down 5 m book."""
__version__ = "0.1.0"
```

Install it editable:

```bash
python -m pip install -e . --no-deps
```

`--no-deps` because numpy/pandas/pyarrow/matplotlib are already present and we do not want pip resolving them.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_packaging.py -q`
Expected: PASS, 2 passed

Then the suite: `python -m pytest harness/tests -q` — expect 218 passed.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/pyproject.toml backtesting_5m/harness/__init__.py backtesting_5m/harness/tests/test_packaging.py
git commit -m "feat(harness): make the module an installed package

Investigations live at the repository root. The sys.path.insert each runner
carried was depth-dependent and, from the wrong depth, silently resolved
blocks to defaults instead of erroring -- quietly evaluating a different model.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01ED1yvqexsuevgcrrK16ymZ"
```

---

## Task 2: The stream spec, writer and validator

**Files:**
- Create: `harness/streams/__init__.py`, `harness/streams/spec.py`, `harness/streams/writer.py`, `harness/streams/validate.py`
- Test: `harness/tests/test_stream_spec.py`

**Interfaces:**
- Consumes: `harness.io.read_parquet`.
- Produces:
  - `TimeKind` (`RECEIPT`, `SOURCE_STAMP`)
  - `Stream(name, time_col="recv_ns", time_kind=None, time_unit="ns", causal=None, max_age_ms=None, transport_offset_ms=None)`
  - `RESERVED = ("recv_ns", "src_ns", "venue", "seq")`, `META_PREFIX = "stream."`
  - `write_stream(df, root, *, name, asset, causal, recorder, date_col=None) -> list[str]`
  - `validate_stream(path) -> dict` — returns parsed metadata, raises `StreamInvalid` otherwise

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_stream_spec.py`:

```python
"""Pins the stream standard.

The rule that carries most of the value: alignment is ALWAYS `recv_ns`, never
`src_ns`. A source stamp can LEAD receipt -- rtds_btc's oracle_ms leads its own
px_first_recv_ns by ~1.5 s -- so aligning on it feeds a model prices before it
was told them. Making receipt the only alignable column means that bug cannot
be written.
"""
import numpy as np
import pandas as pd
import pytest

from harness.streams import (RESERVED, Stream, StreamInvalid, TimeKind,
                             validate_stream, write_stream)


def _frame(n=5, with_src=True):
    base = 1_786_665_600_000_000_000
    d = pd.DataFrame({
        "recv_ns": base + np.arange(n, dtype="int64") * 100_000_000,
        "mid": np.linspace(100.0, 101.0, n),
    })
    if with_src:
        d["src_ns"] = d["recv_ns"] - 1_500_000_000      # source LEADS receipt
    return d


def test_reserved_columns_are_the_documented_four():
    assert RESERVED == ("recv_ns", "src_ns", "venue", "seq")


def test_write_then_validate_round_trips_the_metadata(tmp_path):
    write_stream(_frame(), str(tmp_path), name="venue_l1", asset="BTC",
                 causal=True, recorder="london_recorder")
    meta = validate_stream(str(tmp_path))
    assert meta["name"] == "venue_l1"
    assert meta["asset"] == "BTC"
    assert meta["causal"] is True
    assert meta["recorder"] == "london_recorder"
    assert meta["schema"] == "1"


def test_value_columns_are_inferred_from_the_schema(tmp_path):
    """No metadata map -- the schema is authoritative and cannot drift."""
    write_stream(_frame(), str(tmp_path), name="s", asset="BTC",
                 causal=True, recorder="r")
    assert validate_stream(str(tmp_path))["values"] == ("mid",)


def test_a_frame_without_recv_ns_is_refused(tmp_path):
    bad = _frame().drop(columns=["recv_ns"])
    with pytest.raises(StreamInvalid, match="recv_ns"):
        write_stream(bad, str(tmp_path), name="s", asset="BTC",
                     causal=True, recorder="r")


def test_recv_ns_must_be_int64(tmp_path):
    bad = _frame()
    bad["recv_ns"] = bad["recv_ns"].astype("float64")
    with pytest.raises(StreamInvalid, match="int64"):
        write_stream(bad, str(tmp_path), name="s", asset="BTC",
                     causal=True, recorder="r")


def test_files_are_partitioned_by_date(tmp_path):
    import os
    write_stream(_frame(), str(tmp_path), name="s", asset="BTC",
                 causal=True, recorder="r")
    assert any(p.startswith("date=") for p in os.listdir(str(tmp_path)))


def test_validate_rejects_a_parquet_with_no_stream_metadata(tmp_path):
    import os
    os.makedirs(str(tmp_path / "date=2026-08-20"), exist_ok=True)
    _frame().to_parquet(str(tmp_path / "date=2026-08-20" / "part.parquet"))
    with pytest.raises(StreamInvalid, match="metadata"):
        validate_stream(str(tmp_path))


def test_causal_absent_from_metadata_is_refused_not_defaulted(tmp_path):
    """Whether a series may be carried forward is not safe to guess."""
    import os, pyarrow.parquet as pq, pyarrow as pa
    os.makedirs(str(tmp_path / "date=2026-08-20"), exist_ok=True)
    tbl = pa.Table.from_pandas(_frame(), preserve_index=False)
    tbl = tbl.replace_schema_metadata({b"stream.schema": b"1",
                                       b"stream.name": b"s"})
    pq.write_table(tbl, str(tmp_path / "date=2026-08-20" / "part.parquet"))
    with pytest.raises(StreamInvalid, match="causal"):
        validate_stream(str(tmp_path))


def test_source_stamp_kind_requires_a_transport_offset():
    with pytest.raises(StreamInvalid, match="transport_offset_ms"):
        Stream(name="x", time_col="oracle_ms", time_kind=TimeKind.SOURCE_STAMP)


def test_receipt_kind_needs_no_offset():
    s = Stream(name="x", time_col="px_first_recv_ns",
               time_kind=TimeKind.RECEIPT, causal=True)
    assert s.time_kind is TimeKind.RECEIPT
    assert s.transport_offset_ms is None


def test_time_kind_has_no_default():
    with pytest.raises(StreamInvalid, match="time_kind"):
        Stream(name="x", time_col="whatever", causal=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_stream_spec.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.streams'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/streams/spec.py`:

```python
"""What a stream is, and the one rule that makes lookahead unwritable.

RESERVED COLUMNS
    recv_ns   int64, REQUIRED. Epoch nanoseconds, when THIS HOST received it.
    src_ns    int64, required when the source provides one. The source's own
              stamp. Recorded, never used for alignment.
    venue     optional, for multi-venue streams.
    seq       optional, gap detection.

Everything else is a value column. There is deliberately no metadata map of
value columns: the parquet schema is authoritative and cannot drift from
itself.

ALIGNMENT IS ALWAYS `recv_ns`, NEVER `src_ns`.

A source stamp can LEAD receipt. `rtds_btc`'s `oracle_ms` leads its own
`px_first_recv_ns` by roughly 1.5 s, so aligning on it hands a model prices
before it was told them -- lookahead, silently, in the alpha column. That bug
was found and fixed by hand on 2026-09-09. Making receipt the only alignable
column means it cannot be written again.
"""
from dataclasses import dataclass
from enum import Enum

RESERVED = ("recv_ns", "src_ns", "venue", "seq")
META_PREFIX = "stream."
SCHEMA_VERSION = "1"


class StreamInvalid(ValueError):
    """A stream file or declaration does not meet the standard."""


class TimeKind(Enum):
    RECEIPT = "receipt"
    SOURCE_STAMP = "source_stamp"


@dataclass(frozen=True)
class Stream:
    """An adapter for a NON-conforming file. Conforming files need none.

    `time_kind` has no default on purpose. Declaring RECEIPT is free;
    declaring SOURCE_STAMP additionally requires a transport offset, which the
    run records as an assumption -- the same treatment the clock offset gets.
    A stream that cannot say which it is cannot be registered.
    """
    name: str
    time_col: str = "recv_ns"
    time_kind: TimeKind | None = None
    time_unit: str = "ns"
    causal: bool | None = None
    max_age_ms: float | None = None
    transport_offset_ms: float | None = None

    #: True for a panel already keyed on (open_ts, t_ms) rather than an
    #: absolute receipt time -- the book and spot panels. Such a file is
    #: already on the decision grid and must NOT be routed through
    #: `grid_stream`, whose time column is an absolute epoch: `t_ms` is
    #: milliseconds since the market open, so the subtraction would go hugely
    #: negative and drop every row silently.
    pre_gridded: bool = False

    def __post_init__(self):
        if self.time_kind is None:
            raise StreamInvalid(
                f"{self.name}: time_kind is required and has no default. Use "
                f"TimeKind.RECEIPT when the column records arrival on this "
                f"host, or TimeKind.SOURCE_STAMP with a transport_offset_ms.")
        if (self.time_kind is TimeKind.SOURCE_STAMP
                and self.transport_offset_ms is None):
            raise StreamInvalid(
                f"{self.name}: time_kind=SOURCE_STAMP requires "
                f"transport_offset_ms. A source stamp can lead receipt, so "
                f"aligning on it without an offset is lookahead.")
        if self.time_unit not in ("ns", "us", "ms", "s"):
            raise StreamInvalid(f"{self.name}: bad time_unit {self.time_unit!r}")
```

Create `harness/streams/writer.py`:

```python
"""Emit conforming stream files, so a recorder never hand-rolls metadata."""
import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from harness.streams.spec import META_PREFIX, SCHEMA_VERSION, StreamInvalid


def write_stream(df, root, *, name, asset, causal, recorder, date_col=None):
    """Write `df` as a conforming stream under `root`, partitioned by date.

    Returns the list of files written. `causal` is mandatory and is written to
    the metadata: whether a series may be carried forward onto a decision grid
    is not something a reader should have to guess.
    """
    if "recv_ns" not in df.columns:
        raise StreamInvalid(f"{name}: a stream must carry recv_ns")
    if df["recv_ns"].dtype != "int64":
        raise StreamInvalid(
            f"{name}: recv_ns must be int64 epoch nanoseconds, got "
            f"{df['recv_ns'].dtype}")

    day = (df[date_col] if date_col else
           pd.to_datetime(df["recv_ns"], unit="ns").dt.strftime("%Y-%m-%d"))

    meta = {
        f"{META_PREFIX}schema": SCHEMA_VERSION,
        f"{META_PREFIX}name": name,
        f"{META_PREFIX}asset": asset,
        f"{META_PREFIX}causal": "true" if causal else "false",
        f"{META_PREFIX}recorder": recorder,
        f"{META_PREFIX}span_start_ns": str(int(df["recv_ns"].min())),
        f"{META_PREFIX}span_end_ns": str(int(df["recv_ns"].max())),
        f"{META_PREFIX}created_utc": pd.Timestamp.utcnow().isoformat(),
    }

    written = []
    for d, part in df.groupby(day, sort=True):
        out_dir = os.path.join(root, f"date={d}")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "part.parquet")
        tbl = pa.Table.from_pandas(part, preserve_index=False)
        tbl = tbl.replace_schema_metadata(
            {k.encode(): str(v).encode() for k, v in meta.items()})
        pq.write_table(tbl, path)
        written.append(path)
    return written
```

Create `harness/streams/validate.py`:

```python
"""Read a stream's self-description, and refuse anything ambiguous.

Parquet key-value metadata lives in the footer and is parsed once per file at
open, never per row -- it costs nothing at read time. It is also robust across
writer versions: pyarrow 19 reads these files' metadata even where it cannot
read their data pages.
"""
import glob
import os

import pyarrow.parquet as pq

from harness.streams.spec import META_PREFIX, RESERVED, StreamInvalid


def _first_file(path):
    if os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, "**", "*.parquet"),
                                recursive=True))
        if not hits:
            raise StreamInvalid(f"{path}: no parquet files found")
        return hits[0]
    return path


def validate_stream(path):
    """Return the parsed stream metadata, or raise StreamInvalid."""
    f = _first_file(path)
    md = pq.ParquetFile(f).schema_arrow.metadata or {}
    kv = {k.decode(): v.decode() for k, v in md.items()
          if k.decode().startswith(META_PREFIX)}
    if not kv:
        raise StreamInvalid(
            f"{path}: no stream.* metadata. Non-conforming files need an "
            f"explicit Stream(...) adapter at registration.")

    out = {k[len(META_PREFIX):]: v for k, v in kv.items()}
    if "causal" not in out:
        raise StreamInvalid(
            f"{path}: metadata has no stream.causal. Whether a series may be "
            f"carried forward onto a decision grid is not safe to guess.")
    out["causal"] = out["causal"].lower() == "true"

    cols = [c for c in pq.ParquetFile(f).schema_arrow.names]
    if "recv_ns" not in cols:
        raise StreamInvalid(f"{path}: a stream must carry recv_ns")
    out["values"] = tuple(c for c in cols if c not in RESERVED)
    out["path"] = path
    return out
```

Create `harness/streams/__init__.py`:

```python
"""Self-describing tick streams."""
from harness.streams.spec import (META_PREFIX, RESERVED, SCHEMA_VERSION,
                                  Stream, StreamInvalid, TimeKind)
from harness.streams.validate import validate_stream
from harness.streams.writer import write_stream

__all__ = ["META_PREFIX", "RESERVED", "SCHEMA_VERSION", "Stream",
           "StreamInvalid", "TimeKind", "validate_stream", "write_stream"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_stream_spec.py -q`
Expected: PASS, 11 passed. Then `python -m pytest harness/tests -q` — expect 229.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness/streams backtesting_5m/harness/tests/test_stream_spec.py
git commit -m "feat(streams): the stream standard, writer and validator

Alignment is always recv_ns, never src_ns. A source stamp can lead receipt --
rtds_btc's oracle_ms leads its own px_first_recv_ns by ~1.5 s -- so aligning on
it is lookahead by construction.

<trailer>"
```

---

## Task 3: Read and grid one stream onto the decision grid

**Files:**
- Create: `harness/streams/reader.py`
- Test: `harness/tests/test_stream_reader.py`

**Interfaces:**
- Consumes: `Stream`, `validate_stream`, `harness.io.read_parquet`, `harness.core.episode.shift_to_decision_grid`, `harness.paths.{BUCKET_MS, N_BUCKET, H}`.
- Produces: `grid_stream(df, open_ts, value_cols, time_col="recv_ns", time_unit="ns", causal=True) -> dict[str, np.ndarray]` returning `{col: array}` plus `"age_ms"` and `"has"`.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_stream_reader.py`:

```python
"""Pins stream gridding.

A registered stream must inherit the SAME causality treatment as the built-in
columns -- it does not get to re-earn the lookahead guarantee, it gets it by
routing through the same shift.
"""
import numpy as np
import pandas as pd
import pytest

from harness.streams.reader import grid_stream

OPEN = 1_786_665_600


def _rows(offsets_s, values):
    return pd.DataFrame({
        "recv_ns": [(OPEN + o) * 1_000_000_000 for o in offsets_s],
        "px": values,
    })


def test_an_observation_is_not_visible_at_its_own_bucket():
    out = grid_stream(_rows([0.0], [7.0]), OPEN, ("px",))
    assert np.isnan(out["px"][0]), "bucket 0 leaked into decision index 0"
    assert out["px"][1] == 7.0


def test_age_is_zero_when_first_usable_then_grows():
    out = grid_stream(_rows([0.7], [7.0]), OPEN, ("px",))
    assert out["px"][8] == 7.0 and out["age_ms"][8] == 0.0
    assert out["px"][9] == 7.0 and out["age_ms"][9] == 100.0


def test_has_is_false_before_the_first_observation():
    out = grid_stream(_rows([0.4], [1.0]), OPEN, ("px",))
    assert not out["has"][:5].any()
    assert out["has"][5:].all()


def test_first_observation_in_a_bucket_wins():
    df = _rows([0.00, 0.05, 0.10], [1.0, 999.0, 2.0])
    out = grid_stream(df, OPEN, ("px",))
    assert out["px"][1] == 1.0
    assert out["px"][2] == 2.0


def test_observations_outside_the_window_are_dropped():
    df = _rows([-1.0, 0.5, 301.0], [1.0, 2.0, 3.0])
    out = grid_stream(df, OPEN, ("px",))
    assert np.isfinite(out["px"]).sum() > 0
    assert out["px"][-1] == 2.0


def test_a_non_causal_stream_is_not_shifted():
    """Ground-truth series plotted after the fact are not decision inputs."""
    out = grid_stream(_rows([0.0], [7.0]), OPEN, ("px",), causal=False)
    assert out["px"][0] == 7.0


def test_millisecond_time_units_are_honoured():
    df = pd.DataFrame({"recv_ns": [(OPEN * 1000) + 700], "px": [3.0]})
    out = grid_stream(df, OPEN, ("px",), time_unit="ms")
    assert out["px"][8] == 3.0


def test_bucket_boundary_survives_float_error():
    """0.1 s after open must land in bucket 100, not fall back to 0."""
    out = grid_stream(_rows([0.1, 0.0996], [1.0, 2.0]), OPEN, ("px",))
    assert out["px"][2] == 1.0, "exact 0.1 s must not fall a bucket early"
    assert out["px"][1] == 2.0, "99.6 ms must not be rounded up a bucket"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_stream_reader.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.streams.reader'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/streams/reader.py`:

```python
"""Put one stream's observations onto the decision grid.

Two rules carried from the panel build so every grid means the same thing:
buckets are [t, t+100) since the market open, and the FIRST observation in a
bucket wins.

The epsilon is in BUCKET units. An epoch second near 1.79e9 has a float64 ULP
of 2.4e-7 s, so (ts - open_ts) * 1000 lands up to ~2.4e-4 ms below a whole
millisecond: an observation exactly 0.1 s after the open computes as 99.9999 ms
and would floor into bucket 0 instead of 100. A 10 Hz sampler puts most
observations ON those exact multiples, so that is the common case. Do NOT
"fix" it by rounding to the nearest millisecond -- that pushes a true 99.6 ms
observation into bucket 100, an error 500x larger.
"""
import numpy as np

from harness.core.episode import shift_to_decision_grid
from harness.paths import BUCKET_MS, H, N_BUCKET

EPS = 1e-5

_TO_S = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def grid_stream(df, open_ts, value_cols, time_col="recv_ns",
                time_unit="ns", causal=True):
    """Grid one stream's rows for one market window.

    Returns {col: array(N_BUCKET)} plus "age_ms" and "has". When `causal`,
    every series routes through `shift_to_decision_grid`, the same function the
    built-in columns use -- a registered stream inherits the lookahead
    guarantee rather than re-earning it.
    """
    ts = df[time_col].to_numpy(dtype="float64") * _TO_S[time_unit]
    k = np.floor((ts - open_ts) * 1000.0 / BUCKET_MS + EPS).astype("int64")
    keep = (k >= 0) & (k < N_BUCKET)
    k = k[keep]

    order = np.argsort(k, kind="stable")            # first-in-bucket wins
    k_sorted = k[order]
    first = np.ones(len(k_sorted), dtype=bool)
    first[1:] = k_sorted[1:] != k_sorted[:-1]
    slots = k_sorted[first]

    out = {}
    present = np.zeros(N_BUCKET, dtype=bool)
    present[slots] = True

    age = None
    for col in value_cols:
        raw = np.full(N_BUCKET, np.nan)
        vals = df[col].to_numpy(dtype="float64")[keep][order][first]
        raw[slots] = vals
        if causal:
            carried, a = shift_to_decision_grid(raw, present)
            out[col] = carried
            age = a if age is None else age
        else:
            out[col] = raw
            age = np.where(present, 0.0, np.inf) if age is None else age

    out["age_ms"] = age
    out["has"] = np.isfinite(out[value_cols[0]]) if value_cols else present
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_stream_reader.py -q`
Expected: PASS, 8 passed. Full suite: 237.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness/streams/reader.py backtesting_5m/harness/tests/test_stream_reader.py
git commit -m "feat(streams): grid a stream onto the decision grid, causality included

<trailer>"
```

---

## Task 4: The registry and the catalog

**Files:**
- Create: `harness/streams/registry.py`, `harness/streams/catalog.py`
- Modify: `harness/streams/__init__.py`
- Test: `harness/tests/test_stream_registry.py`

**Interfaces:**
- Consumes: `Stream`, `TimeKind`, `validate_stream`.
- Produces:
  - `register(name, path, adapter=None) -> None`
  - `resolve(name) -> RegisteredStream(name, path, adapter, meta, causal, values)`
  - `registered() -> tuple[str, ...]`, `clear_registry()` (tests only)
  - `StreamNotRegistered(KeyError)`

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_stream_registry.py`:

```python
"""Pins stream registration and resolution.

Streams resolve investigation-first, catalog-second, mirroring how blocks
already resolve. The registered name is the LOCAL ALIAS and always wins; a
conforming file's stream.name metadata is informational.
"""
import numpy as np
import pandas as pd
import pytest

from harness.streams import (Stream, StreamInvalid, StreamNotRegistered,
                             TimeKind, clear_registry, register, registered,
                             resolve, write_stream)


@pytest.fixture(autouse=True)
def _clean():
    clear_registry()
    yield
    clear_registry()


def _write(tmp_path, name="venue_l1"):
    base = 1_786_665_600_000_000_000
    df = pd.DataFrame({"recv_ns": base + np.arange(3, dtype="int64") * 10**8,
                       "mid": [1.0, 2.0, 3.0]})
    write_stream(df, str(tmp_path), name=name, asset="BTC", causal=True,
                 recorder="london_recorder")
    return str(tmp_path)


def test_a_conforming_stream_needs_no_declaration(tmp_path):
    register("spot", _write(tmp_path))
    got = resolve("spot")
    assert got.causal is True
    assert got.values == ("mid",)


def test_the_registered_alias_wins_over_the_files_own_name(tmp_path):
    register("my_alias", _write(tmp_path, name="venue_l1"))
    got = resolve("my_alias")
    assert got.name == "my_alias"
    assert got.meta["name"] == "venue_l1"


def test_an_unregistered_name_raises_and_lists_what_is_registered(tmp_path):
    register("spot", _write(tmp_path))
    with pytest.raises(StreamNotRegistered, match="spot"):
        resolve("chainlink")


def test_registering_the_same_name_twice_shadows_investigation_first(tmp_path):
    a = _write(tmp_path / "a", name="one")
    b = _write(tmp_path / "b", name="two")
    register("spot", a)
    register("spot", b)                       # later call wins: investigation
    assert resolve("spot").meta["name"] == "two"


def test_a_legacy_file_needs_an_adapter(tmp_path):
    import os
    d = tmp_path / "legacy"
    os.makedirs(str(d))
    pd.DataFrame({"oracle_ms": [1], "px": [2.0]}).to_parquet(
        str(d / "p.parquet"))
    with pytest.raises(StreamInvalid, match="adapter"):
        register("chainlink", str(d))

    register("chainlink", str(d), adapter=Stream(
        name="chainlink", time_col="oracle_ms", time_kind=TimeKind.RECEIPT,
        time_unit="ms", causal=True))
    assert resolve("chainlink").causal is True


def test_registered_lists_names(tmp_path):
    register("spot", _write(tmp_path))
    assert registered() == ("spot",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_stream_registry.py -q`
Expected: FAIL — `ImportError: cannot import name 'register'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/streams/registry.py`:

```python
"""Which name points at which stream.

Resolution mirrors block resolution: the catalog holds the standard streams,
an investigation registers its own, and a later registration of the same name
shadows an earlier one -- so an investigation overrides the catalog simply by
registering after import.
"""
from dataclasses import dataclass

from harness.streams.spec import Stream, StreamInvalid
from harness.streams.validate import validate_stream


class StreamNotRegistered(KeyError):
    """A name was requested that nothing has registered."""


@dataclass(frozen=True)
class RegisteredStream:
    name: str
    path: str
    adapter: Stream | None
    meta: dict
    causal: bool
    values: tuple

    @property
    def time_col(self):
        return self.adapter.time_col if self.adapter else "recv_ns"

    @property
    def time_unit(self):
        return self.adapter.time_unit if self.adapter else "ns"

    @property
    def pre_gridded(self):
        """True for panels already keyed on (open_ts, t_ms).

        `grid_stream` treats its time column as an ABSOLUTE epoch time. The
        book and spot panels key on `t_ms`, milliseconds SINCE THE MARKET
        OPEN, so routing them through it would compute `ts - open_ts` as a
        hugely negative number and silently drop every observation. They are
        also already on the decision grid, so there is nothing to grid. Such a
        panel is read column-wise and never gridded.
        """
        return bool(self.adapter and self.adapter.pre_gridded)


_REGISTRY = {}


def register(name, path, adapter=None):
    """Register `path` under the local alias `name`.

    A conforming file needs no adapter. A non-conforming one needs an explicit
    `Stream(...)`, whose `time_kind` has no default -- see harness.streams.spec.
    """
    if adapter is None:
        try:
            meta = validate_stream(path)
        except StreamInvalid as exc:
            raise StreamInvalid(
                f"{name}: {exc}. Pass adapter=Stream(...) to register a "
                f"non-conforming file explicitly.") from exc
        causal, values = meta["causal"], meta["values"]
    else:
        meta = {"name": adapter.name}
        if adapter.causal is None:
            raise StreamInvalid(f"{name}: adapter must state causal=")
        causal, values = adapter.causal, ()

    _REGISTRY[name] = RegisteredStream(name, path, adapter, meta, causal,
                                       values)


def resolve(name):
    if name not in _REGISTRY:
        raise StreamNotRegistered(
            f"{name!r} is not registered. Registered: {registered()}")
    return _REGISTRY[name]


def registered():
    return tuple(sorted(_REGISTRY))


def clear_registry():
    """Tests only."""
    _REGISTRY.clear()
```

Create `harness/streams/catalog.py`:

```python
"""The standard streams, in one readable screen.

An investigation registers its own by calling `register()` before `backtest()`,
and shadows any of these by registering the same name.
"""
from harness import paths
from harness.streams.registry import register
from harness.streams.spec import Stream, TimeKind


def install():
    """Register the standard streams. Idempotent."""
    #: PRE-GRIDDED. The book and spot panels are already keyed on
    #: (open_ts, t_ms) and are already on the decision grid. They are
    #: registered for discovery and documentation; `load_episodes` reads them
    #: by its existing path and never routes them through `grid_stream`.
    register("book", paths.PANEL, adapter=Stream(
        name="book", time_col="t_ms", time_kind=TimeKind.RECEIPT,
        time_unit="ms", causal=True, pre_gridded=True))

    register("spot", paths.SPOT, adapter=Stream(
        name="spot", time_col="t_ms", time_kind=TimeKind.RECEIPT,
        time_unit="ms", causal=True, pre_gridded=True))

    #: `px_first_recv_ns`, NOT `oracle_ms`. The oracle's own stamp runs ~1.5 s
    #: ahead of arrival on this vantage, so aligning on it is lookahead.
    register("chainlink", paths.RTDS_BTC, adapter=Stream(
        name="chainlink", time_col="px_first_recv_ns",
        time_kind=TimeKind.RECEIPT, time_unit="ns", causal=True))
```

Extend `harness/streams/__init__.py` exports with `register`, `resolve`,
`registered`, `clear_registry`, `StreamNotRegistered`, `RegisteredStream`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_stream_registry.py -q`
Expected: PASS, 6 passed. Full suite: 243.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness/streams backtesting_5m/harness/tests/test_stream_registry.py
git commit -m "feat(streams): registry and catalog, investigation-first resolution

<trailer>"
```

---

## Task 5: `Episode.stream()` and the `t_ms` time base

**Files:**
- Modify: `harness/core/episode.py`
- Test: `harness/tests/test_episode_streams.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `Episode.t_ms: np.ndarray` — decision time of each index; today `i * 100`
  - `Episode.streams: dict[str, dict]` (populated by the loader)
  - `Episode.stream(name) -> StreamView` with attribute access to values plus `.age_ms`, `.has`
  - `StreamView.__getattr__` raising `AttributeError` listing available columns

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_episode_streams.py`:

```python
"""Pins the stream accessor and the time base.

Core fields stay statically typed; registered streams come through an explicit
accessor. A typo must fail at the first tick, not surface as NaN deep in a run.
"""
import numpy as np
import pytest

from harness.paths import N_BUCKET


def test_t_ms_is_the_decision_time_of_each_index(flat_episode):
    assert flat_episode.t_ms.shape == (N_BUCKET,)
    assert flat_episode.t_ms[0] == 0
    assert flat_episode.t_ms[1] == 100
    assert flat_episode.t_ms[-1] == (N_BUCKET - 1) * 100


def test_tte_is_consistent_with_the_time_base(flat_episode):
    i = 1500
    assert flat_episode.tte_s(i) == pytest.approx(
        300.0 - flat_episode.t_ms[i] / 1000.0)


def test_a_registered_stream_is_reachable_by_name(flat_episode):
    from dataclasses import replace
    ep = replace(flat_episode, streams={
        "chainlink": {"px": np.full(N_BUCKET, 5.0),
                      "age_ms": np.zeros(N_BUCKET),
                      "has": np.ones(N_BUCKET, dtype=bool)}})
    cl = ep.stream("chainlink")
    assert cl.px[10] == 5.0
    assert cl.age_ms[10] == 0.0
    assert cl.has[10]


def test_an_unregistered_stream_raises_and_lists_what_is_there(flat_episode):
    from dataclasses import replace
    ep = replace(flat_episode, streams={"chainlink": {"px": np.zeros(1),
                                                      "age_ms": np.zeros(1),
                                                      "has": np.zeros(1, bool)}})
    with pytest.raises(KeyError, match="chainlink"):
        ep.stream("chainlnk")


def test_a_mistyped_column_raises_listing_the_real_ones(flat_episode):
    from dataclasses import replace
    ep = replace(flat_episode, streams={"cl": {"px": np.zeros(1),
                                               "age_ms": np.zeros(1),
                                               "has": np.zeros(1, bool)}})
    with pytest.raises(AttributeError, match="px"):
        ep.stream("cl").pxx


def test_episodes_without_streams_still_work(flat_episode):
    assert flat_episode.streams == {}
    with pytest.raises(KeyError):
        flat_episode.stream("anything")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_episode_streams.py -q`
Expected: FAIL — `AttributeError: 'Episode' object has no attribute 't_ms'`

- [ ] **Step 3: Write minimal implementation**

In `harness/core/episode.py`, add above `Episode`:

```python
class StreamView:
    """One registered stream's decision-aligned arrays.

    Values are reached as attributes; `age_ms` and `has` always exist. A
    mistyped column raises immediately and names the real ones, so it fails at
    the first tick rather than surfacing as NaN deep in a sweep.
    """
    __slots__ = ("_name", "_cols")

    def __init__(self, name, cols):
        self._name = name
        self._cols = cols

    def __getattr__(self, item):
        try:
            return self._cols[item]
        except KeyError:
            raise AttributeError(
                f"stream {self._name!r} has no column {item!r}; "
                f"it has {tuple(sorted(self._cols))}") from None

    def __repr__(self):
        return f"StreamView({self._name!r}, {tuple(sorted(self._cols))})"
```

Add two fields to the `Episode` dataclass, after the warm-up block so existing
positional construction is unaffected:

```python
    #: Decision time of each index, milliseconds since the market open. On the
    #: 100 ms grid this is exactly i*100. It exists so blocks and the engine ask
    #: "what time is index i" rather than assuming buckets -- which is what
    #: makes event replay additive rather than a rewrite.
    t_ms: np.ndarray = field(
        default_factory=lambda: np.arange(N_BUCKET, dtype="int64") * BUCKET_MS)

    #: Registered streams: {name: {col: array, "age_ms": array, "has": array}}
    streams: dict = field(default_factory=dict)
```

And a method on `Episode`:

```python
    def stream(self, name):
        """The decision-aligned arrays of a registered stream."""
        if name not in self.streams:
            raise KeyError(
                f"{name!r} is not on this episode; it has "
                f"{tuple(sorted(self.streams))}")
        return StreamView(name, self.streams[name])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_episode_streams.py -q`
Expected: PASS, 6 passed. Full suite: 249, with every pre-existing test still green — the new fields have defaults, so nothing that constructs an `Episode` today changes.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness/core/episode.py backtesting_5m/harness/tests/test_episode_streams.py
git commit -m "feat(episode): a t_ms time base and a stream() accessor

t_ms exists so blocks ask what time an index is rather than assuming 100 ms
buckets, which is what makes event replay additive later.

<trailer>"
```

---

## Task 6: `Sample.require` — exclude, never score as zero

**Files:**
- Modify: `harness/core/config.py`, `harness/core/run.py`
- Test: `harness/tests/test_sample_require.py`

**Interfaces:**
- Consumes: `Episode.streams`, `Episode.has_spot`.
- Produces: `Sample.require: tuple = ()`; `run()`'s `summary["sample"]["dropped"] = {stream: count}`.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_sample_require.py`:

```python
"""Pins require-by-stream.

`require_spot` ignored Chainlink and silently scored 237 unpriceable markets as
$0.00, which entered the mean as real observations. Excluded must mean ABSENT
-- not present-with-NaN, and never zero.
"""
import numpy as np
import pytest

from harness.core.config import Sample
from harness.core.run import select_episodes


def _eps(flat_episode, n=4, with_cl=2):
    from dataclasses import replace
    out = []
    for k in range(n):
        streams = {}
        if k < with_cl:
            streams["chainlink"] = {
                "px": np.full(len(flat_episode), 1.0),
                "age_ms": np.zeros(len(flat_episode)),
                "has": np.ones(len(flat_episode), dtype=bool)}
        out.append(replace(flat_episode, market_id=f"m{k}",
                           open_ts=1786665600 + 300 * k, streams=streams))
    return out


def test_require_excludes_markets_missing_the_stream(flat_episode):
    kept, dropped = select_episodes(_eps(flat_episode), Sample(
        require=("chainlink",)))
    assert [e.market_id for e in kept] == ["m0", "m1"]
    assert dropped["chainlink"] == 2


def test_excluded_markets_are_absent_not_zeroed(flat_episode):
    kept, _ = select_episodes(_eps(flat_episode), Sample(require=("chainlink",)))
    assert all("chainlink" in e.streams for e in kept)


def test_no_require_keeps_everything(flat_episode):
    kept, dropped = select_episodes(_eps(flat_episode), Sample())
    assert len(kept) == 4 and dropped == {}


def test_require_spot_still_works_for_existing_callers(flat_episode):
    kept, _ = select_episodes(_eps(flat_episode), Sample(require_spot=True))
    assert len(kept) == 4          # flat_episode has spot
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_sample_require.py -q`
Expected: FAIL — `ImportError: cannot import name 'select_episodes'`

- [ ] **Step 3: Write minimal implementation**

Add to `Sample` in `harness/core/config.py`:

```python
    #: Exclude any market for which a named stream has no usable observation
    #: in the window. EXCLUDED MEANS ABSENT -- not present-with-NaN, and never
    #: scored as zero. `require_spot` is the older special case and is kept
    #: working; prefer require=("spot",).
    require: tuple = ()
```

In `harness/core/run.py`, rename the private `_select` to a public
`select_episodes` that also returns drop counts, and keep `_select` as a thin
alias so nothing else breaks:

```python
def select_episodes(episodes, sample):
    """(kept, dropped_counts). Drop counts are reported in summary.json so a
    require clause that halves the sample is visible rather than inferred."""
    dropped = {}
    out = []
    for ep in episodes:
        if sample.t0 is not None and ep.open_ts < sample.t0:
            continue
        if sample.t1 is not None and ep.open_ts >= sample.t1:
            continue
        if sample.days and ep.day not in sample.days:
            continue
        if sample.markets and ep.market_id not in sample.markets:
            continue
        if sample.require_spot and not ep.has_spot.any():
            dropped["spot"] = dropped.get("spot", 0) + 1
            continue
        missing = None
        for name in sample.require:
            if name == "spot":
                ok = bool(ep.has_spot.any())
            else:
                ok = name in ep.streams and bool(ep.streams[name]["has"].any())
            if not ok:
                missing = name
                break
        if missing:
            dropped[missing] = dropped.get(missing, 0) + 1
            continue
        out.append(ep)

    out.sort(key=lambda e: e.open_ts)
    if sample.max_markets is not None:
        out = out[: sample.max_markets]
    return out, dropped


def _select(episodes, sample):
    return select_episodes(episodes, sample)[0]
```

In `run()`, replace the `_select` call with `select_episodes` and add to the
summary:

```python
    selected, dropped = select_episodes(episodes, sample)
    ...
    summary["sample"] = {"n_selected": len(selected), "dropped": dropped}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_sample_require.py -q`
Expected: PASS, 4 passed. Full suite: 253.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness/core/config.py backtesting_5m/harness/core/run.py backtesting_5m/harness/tests/test_sample_require.py
git commit -m "feat(sample): require streams by name, and report what was dropped

require_spot ignored Chainlink and scored 237 unpriceable markets as \$0.00.
Excluded now means absent, and the per-stream drop count reaches summary.json.

<trailer>"
```

---

## Task 7: Registry-driven episode loading

**Files:**
- Modify: `harness/build/episodes.py`
- Test: `harness/tests/test_episodes_streams.py`

**Interfaces:**
- Consumes: `resolve`, `grid_stream`, `Episode.streams`.
- Produces: `load_episodes(..., streams=())` attaching each named stream; the same function fixed to group once rather than scanning per market.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_episodes_streams.py`:

```python
"""Pins registry-driven loading, and that it groups once.

The per-market boolean scan was O(markets x rows) over a 4.6 M-row frame.
"""
import time

import numpy as np
import pandas as pd
import pytest

from harness.build.episodes import load_episodes
from harness.streams import (TimeKind, Stream, clear_registry, register,
                             write_stream)


@pytest.fixture(autouse=True)
def _clean():
    clear_registry()
    yield
    clear_registry()


def _panel(tmp_path, n_markets=3):
    rows = []
    for m in range(n_markets):
        open_ts = 1786665600 + 300 * m
        for t in (0, 100, 200):
            rows.append({"market_id": f"m{m}", "open_ts": open_ts, "t_ms": t,
                         "bid": 0.49, "ask": 0.51, "mid": 0.50, "n_src": 2})
    p = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(p)
    s = tmp_path / "strikes.parquet"
    pd.DataFrame({"market_id": [f"m{m}" for m in range(n_markets + 1)],
                  "open_ts": [1786665600 + 300 * m
                              for m in range(n_markets + 1)],
                  "strike": [100.0 + m for m in range(n_markets + 1)]
                  }).to_parquet(s)
    return str(p), str(s)


def test_a_registered_stream_is_attached_to_each_episode(tmp_path):
    panel, strikes = _panel(tmp_path)
    root = tmp_path / "cl"
    base = 1786665600 * 10**9
    write_stream(pd.DataFrame({
        "recv_ns": base + np.arange(9, dtype="int64") * 10**8,
        "px": np.arange(9, dtype="float64")}),
        str(root), name="chainlink", asset="BTC", causal=True, recorder="r")
    register("chainlink", str(root))

    eps = load_episodes(panel_path=panel, strikes_path=strikes,
                        streams=("chainlink",))
    assert "chainlink" in eps[0].streams
    assert eps[0].stream("chainlink").px.shape == (3000,)


def test_streams_not_requested_are_not_attached(tmp_path):
    panel, strikes = _panel(tmp_path)
    eps = load_episodes(panel_path=panel, strikes_path=strikes)
    assert eps[0].streams == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_episodes_streams.py -q`
Expected: FAIL — `TypeError: load_episodes() got an unexpected keyword argument 'streams'`

- [ ] **Step 3: Write minimal implementation**

In `harness/build/episodes.py`, add the parameter and the join, and replace the
per-market scans with a single groupby:

```python
def load_episodes(panel_path=None, strikes_path=None, spot_path=None,
                  fair_path=None, rtds_path=None, days=None, markets=None,
                  max_markets=None, warmup=False, warmup_s=900,
                  fair_is_causal=False, streams=()):
    ...
    # Group ONCE. The previous per-market boolean scan was O(markets x rows)
    # over a 4.6 M-row frame -- a full pass per market.
    spot_by_open = (dict(tuple(spot.groupby("open_ts", sort=False)))
                    if spot is not None else {})

    stream_frames = {}
    for name in streams:
        reg = resolve(name)
        if reg.pre_gridded:
            # Already keyed on (open_ts, t_ms) and already on the decision
            # grid. Routing it through grid_stream would treat t_ms as an
            # absolute epoch and drop every row. Skip: load_episodes reads
            # these panels by its existing path.
            continue
        stream_frames[name] = (reg, read_parquet(reg.path))
    ...
    # inside the per-market loop, after build_episode:
    if stream_frames:
        window = {}
        for name, (reg, df) in stream_frames.items():
            lo = open_ts * 1_000_000_000
            hi = (open_ts + paths.H) * 1_000_000_000
            sub = df[(df[reg.time_col] >= lo) & (df[reg.time_col] < hi)]
            window[name] = grid_stream(
                sub, int(open_ts), reg.values or
                tuple(c for c in sub.columns if c not in RESERVED),
                time_col=reg.time_col, time_unit=reg.time_unit,
                causal=reg.causal)
        ep = replace(ep, streams=window)
    episodes.append(ep)
```

Import `resolve`, `grid_stream`, `RESERVED` and `dataclasses.replace` at the
top of the module.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_episodes_streams.py -q`
Expected: PASS, 2 passed. Full suite: 255.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness/build/episodes.py backtesting_5m/harness/tests/test_episodes_streams.py
git commit -m "feat(episodes): attach registered streams, and group once

The per-market boolean scan was O(markets x rows) over a 4.6 M-row frame.

<trailer>"
```

---

## Task 8: Model directories

**Files:**
- Modify: `harness/core/provenance.py`
- Create: `models/normal_qq/{fair,vol,f,link}.py` (copied from `investigations/2026-09-09-normal-qq-eval/`)
- Test: `harness/tests/test_model_dirs.py`

**Interfaces:**
- Consumes: `resolve_slots`.
- Produces: `resolve_slots(investigation_dir, model_dir=None)` — investigation → model → defaults.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_model_dirs.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_model_dirs.py -q`
Expected: FAIL — `TypeError: resolve_slots() got an unexpected keyword argument 'model_dir'`

- [ ] **Step 3: Write minimal implementation**

In `harness/core/provenance.py`:

```python
def resolve_slots(investigation_dir, model_dir=None):
    """slot -> path. Investigation wins, then the model, then defaults.

    A model directory holds one canonical block set that many investigations
    share; shadowing still works, so forking one block is a one-file act.
    Provenance is unchanged -- write_manifest still freezes the resolved files
    with their sha256s into the run folder.
    """
    resolved = {}
    for slot in SLOTS:
        local = os.path.join(investigation_dir, f"{slot}.py")
        if os.path.exists(local):
            resolved[slot] = local
            continue
        if model_dir:
            in_model = os.path.join(model_dir, f"{slot}.py")
            if os.path.exists(in_model):
                resolved[slot] = in_model
                continue
        resolved[slot] = os.path.join(DEFAULTS_DIR, f"{slot}.py")
    return resolved
```

Then create the canonical model:

```bash
mkdir -p ../models/normal_qq
cp investigations/2026-09-09-normal-qq-eval/{fair,vol,f,link}.py ../models/normal_qq/
```

Add `../models/normal_qq/README.md` recording that these came from
`investigations/2026-09-09-normal-qq-eval/`, that the concurrent session's copy
at `Gambling104/investigations/2026-09-09_normal_qq_basic/` is the upstream
original and is not touched, and that a run's manifest pins the exact versions
by sha256.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_model_dirs.py -q`
Expected: PASS, 3 passed. Full suite: 258.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness/core/provenance.py backtesting_5m/harness/tests/test_model_dirs.py models/
git commit -m "feat(provenance): model directories between investigation and defaults

<trailer>"
```

---

## Task 9: The `backtest()` entry point

**Files:**
- Create: `harness/core/api.py`
- Modify: `harness/__init__.py`
- Test: `harness/tests/test_api.py`

**Interfaces:**
- Consumes: `run`, `select_episodes`, `resolve_slots`, `catalog.install`.
- Produces: `backtest(model=None, streams=(), quote=..., execn=..., sample=..., output=..., episodes=None, investigation_dir=None) -> BacktestResult` with `.summary .ledger .markets .ticks .run_dir`; re-exported from `harness`.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_api.py`:

```python
"""Pins the public surface: one import, one call, a result object."""
import os

import pytest

import harness
from harness import ExecConfig, Output, QuoteParams, Sample, backtest


@pytest.fixture
def episodes(flat_episode):
    from dataclasses import replace
    out = []
    for k in range(6):
        ep = replace(flat_episode, market_id=f"m{k}",
                     open_ts=1786665600 + 300 * k,
                     day=f"2026-08-{14 + k % 3:02d}")
        ep.ask[:] = 0.20
        out.append(ep)
    return out


def test_the_public_names_are_importable_from_harness():
    for n in ("backtest", "Sample", "QuoteParams", "ExecConfig", "Output",
              "register", "Stream", "TimeKind"):
        assert hasattr(harness, n), n


def test_backtest_returns_a_result_object(tmp_path, episodes):
    r = backtest(investigation_dir=str(tmp_path), episodes=episodes,
                 quote=QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
                 execn=ExecConfig(), sample=Sample(),
                 output=Output(plots=False))
    assert os.path.exists(os.path.join(r.run_dir, "summary.json"))
    assert r.summary["headline"]["n_markets"] == 6
    assert len(r.markets) == 6
    assert r.ledger is not None


def test_the_result_carries_the_drop_counts(tmp_path, episodes):
    r = backtest(investigation_dir=str(tmp_path), episodes=episodes,
                 quote=QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
                 execn=ExecConfig(),
                 sample=Sample(require=("chainlink",)),
                 output=Output(plots=False))
    assert r.summary["sample"]["dropped"]["chainlink"] == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_api.py -q`
Expected: FAIL — `ImportError: cannot import name 'backtest' from 'harness'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/core/api.py`:

```python
"""One entry point.

An investigation should declare what it wants and call one function, rather
than importing from six modules and knowing which.
"""
from dataclasses import dataclass

import pandas as pd

from harness.core.run import run


@dataclass(frozen=True)
class BacktestResult:
    run_dir: str
    summary: dict
    ledger: pd.DataFrame
    markets: pd.DataFrame
    ticks: pd.DataFrame | None = None


def backtest(*, model=None, streams=(), quote, execn, sample, output,
             episodes, investigation_dir):
    """Run one backtest and return its artefacts.

    `model` names a shared model directory; blocks resolve investigation-first,
    then the model, then harness defaults.
    """
    out = run(investigation_dir, quote, execn, sample, output, episodes,
              model_dir=model, streams=streams)
    return BacktestResult(run_dir=out["run_dir"], summary=out["summary"],
                          ledger=out["ledger"], markets=out["markets"],
                          ticks=out.get("ticks"))
```

Thread `model_dir` and `streams` through `run()` — `model_dir` into
`resolve_slots`, `streams` recorded in the run config so the manifest shows
which streams a run used.

Extend `harness/__init__.py`:

```python
from harness.core.api import backtest, BacktestResult
from harness.core.config import ExecConfig, Output, QuoteParams, Sample
from harness.streams import Stream, TimeKind, register, registered, resolve

__all__ = ["backtest", "BacktestResult", "ExecConfig", "Output",
           "QuoteParams", "Sample", "Stream", "TimeKind", "register",
           "registered", "resolve", "__version__"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_api.py -q`
Expected: PASS, 3 passed. Full suite: 261.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness/core/api.py backtesting_5m/harness/__init__.py backtesting_5m/harness/core/run.py backtesting_5m/harness/tests/test_api.py
git commit -m "feat(api): one public backtest() entry point

<trailer>"
```

---

## Task 9b: `harness.report` — plotting leaves the engine

**Files:**
- Create: `harness/report/__init__.py`, `harness/report/load.py`, `harness/report/figures.py`
- Modify: `harness/core/run.py` (remove the plot call), `harness/core/config.py` (drop `Output.plots`)
- Delete: `harness/core/plots.py`
- Test: `harness/tests/test_report.py`

**Interfaces:**
- Consumes: `harness.io.read_parquet`.
- Produces:
  - `load_runs(mapping) -> pd.DataFrame` — tidy markets frame with a `run` column
  - `load_ledgers(mapping) -> pd.DataFrame` — same, for ledgers
  - `cumulative_pnl(markets, path, *, gross=True)` — one line per run, gross greyed behind net
  - `markout_distribution(ledgers, path)`, `calibration(ledgers, path)`, `pnl_by_tte(ledgers, path)`

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_report.py`:

```python
"""Pins that reporting is separate from running, and multi-run by default.

`run()` sees exactly one run, so a hyperparameter sweep -- the thing anyone
actually wants to look at -- cannot be plotted from inside it. The loader
therefore takes a mapping of label -> run directory, and a single run is the
one-element case rather than a separate path.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from harness.report import cumulative_pnl, load_runs


def _run_dir(tmp_path, name, pnl):
    d = tmp_path / name
    os.makedirs(str(d))
    pd.DataFrame({
        "market_id": [f"m{i}" for i in range(len(pnl))],
        "open_ts": 1786665600 + np.arange(len(pnl)) * 300,
        "day": ["2026-08-20"] * len(pnl),
        "seed": [0] * len(pnl),
        "pnl_net": pnl,
        "pnl_gross": [p + 0.5 for p in pnl],
        "shares": [10.0] * len(pnl),
        "n_fills": [1] * len(pnl),
    }).to_parquet(str(d / "markets.parquet"))
    with open(str(d / "summary.json"), "w") as fh:
        json.dump({"headline": {"n_markets": len(pnl)}}, fh)
    return str(d)


def test_load_runs_tags_each_row_with_its_run(tmp_path):
    m = load_runs({"a": _run_dir(tmp_path, "a", [1.0, 2.0]),
                   "b": _run_dir(tmp_path, "b", [3.0])})
    assert set(m["run"]) == {"a", "b"}
    assert len(m) == 3


def test_a_single_run_is_the_one_element_case(tmp_path):
    m = load_runs({"only": _run_dir(tmp_path, "only", [1.0])})
    assert list(m["run"]) == ["only"]


def test_cumulative_pnl_writes_one_figure_for_many_runs(tmp_path):
    m = load_runs({"a": _run_dir(tmp_path, "a", [1.0, 2.0]),
                   "b": _run_dir(tmp_path, "b", [-1.0, -2.0])})
    out = str(tmp_path / "cum.png")
    cumulative_pnl(m, out)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_run_no_longer_accepts_a_plots_flag():
    from harness.core.config import Output
    assert not hasattr(Output(), "plots"), (
        "plotting belongs in harness.report, not in the engine")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_report.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.report'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/report/load.py`:

```python
"""Read run folders into tidy frames, many at a time.

The mapping is label -> run directory, so a comparison is the default shape
and a single run is the one-element case.
"""
import json
import os

import pandas as pd

from harness.io import read_parquet


def _tidy(mapping, filename):
    frames = []
    for label, run_dir in mapping.items():
        path = os.path.join(run_dir, filename)
        if not os.path.exists(path):
            continue
        df = read_parquet(path)
        df.insert(0, "run", label)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_runs(mapping):
    """Per-market rows for each run, tagged with a `run` column."""
    return _tidy(mapping, "markets.parquet")


def load_ledgers(mapping):
    """Per-fill rows for each run, tagged with a `run` column."""
    return _tidy(mapping, "ledger.parquet")


def load_summaries(mapping):
    out = {}
    for label, run_dir in mapping.items():
        with open(os.path.join(run_dir, "summary.json")) as fh:
            out[label] = json.load(fh)
    return out
```

Create `harness/report/figures.py`:

```python
"""Generic figures over one or many runs.

Never called by the engine. Bespoke figures belong in the investigation that
wants them, composed from the same loader.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402


def cumulative_pnl(markets, path, *, gross=True, title="Cumulative PnL"):
    """One net line per run; the gross line greyed behind it.

    Gross versus net is the split between forecast error and fee drag, which
    on this venue is the difference between "the model is wrong" and "the
    threshold is wrong". It is worth seeing on every chart.
    """
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for label, g in markets.groupby("run", sort=True):
        m = g[g["seed"] == g["seed"].iloc[0]].sort_values("open_ts")
        net = m["pnl_net"].fillna(0.0).cumsum()
        if gross and "pnl_gross" in m:
            gr = m["pnl_gross"].fillna(0.0).cumsum()
            ax.plot(range(len(m)), gr, lw=1.0, color="0.75", zorder=1)
            drag = float(gr.iloc[-1] - net.iloc[-1])
            shares = float(m["shares"].sum())
            label = (f"{label}  (fee drag ${drag:,.0f}"
                     f" = {100 * drag / shares:.3f} c/share)") if shares else label
        ax.plot(range(len(m)), net, lw=1.3, zorder=2, label=label)

    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("market (chronological)")
    ax.set_ylabel("cumulative PnL, USD")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
```

Then move `markout_distribution`, `calibration` and `pnl_by_tte` into the same
module from `investigations/2026-09-09-normal-qq-eval/plot_report.py`, changing
only their signature to take the tidy frame and to iterate
`groupby("run")` the way `cumulative_pnl` does. Their bodies are otherwise
unchanged — they already do the right thing for a single run.

Create `harness/report/__init__.py` re-exporting all of the above.

Then remove plotting from the engine: delete `harness/core/plots.py`, drop the
`plots` field from `Output`, and remove the `if output.plots:` block from
`run()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_report.py -q`
Expected: PASS, 4 passed. Full suite: 265 — note the count *drops* where
`Output(plots=False)` assertions are removed from existing tests; update those
call sites rather than keeping a dead field.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness/report backtesting_5m/harness/core backtesting_5m/harness/tests/test_report.py
git rm backtesting_5m/harness/core/plots.py
git commit -m "refactor(report): plotting leaves the engine and becomes multi-run

run() sees one run, so it structurally cannot draw the comparison a
hyperparameter sweep needs. The loader takes label -> run dir, making
side-by-side the default shape.

<trailer>"
```

---

## Task 10: Move investigations to `../investigations` and port them

**Files:**
- Move: `backtesting_5m/investigations/*` → `investigations/*`
- Modify: each moved `run.py` and `plot_report.py`
- Test: `harness/tests/test_template_runs.py`

**Interfaces:**
- Consumes: `backtest`, `register`.
- Produces: `investigations/_template/run.py` using the public API.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_template_runs.py`:

```python
"""Pins that the template is importable and uses the public surface.

The template teaches the pattern; if it shows the old sys.path dance, every
new investigation copies it.
"""
import os
import re

TEMPLATE = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                        os.pardir, "investigations", "_template", "run.py")


def test_the_template_exists():
    assert os.path.exists(os.path.abspath(TEMPLATE))


def test_the_template_does_not_manipulate_sys_path():
    src = open(os.path.abspath(TEMPLATE)).read()
    assert "sys.path" not in src, (
        "the template must import harness as an installed package")


def test_the_template_uses_the_public_entry_point():
    src = open(os.path.abspath(TEMPLATE)).read()
    assert re.search(r"from harness import .*backtest", src)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_template_runs.py -q`
Expected: FAIL — the template is still at `backtesting_5m/investigations/`

- [ ] **Step 3: Write minimal implementation**

```bash
cd C:/Users/kaima/Github2/Gambling104
git mv backtesting_5m/investigations/2026-09-09-grid-excursion investigations/
git mv backtesting_5m/investigations/2026-09-09-normal-qq-eval investigations/
git mv backtesting_5m/investigations/2026-09-09-smoke investigations/
git mv backtesting_5m/investigations/2026-09-09-vol-fixed investigations/
git mv backtesting_5m/investigations/_template investigations/
```

Rewrite `investigations/_template/run.py`:

```python
"""Copy this folder, rename it, drop in the blocks you want to change."""
import os

from harness import ExecConfig, Output, QuoteParams, Sample, backtest
from harness.build.episodes import load_episodes
from harness.streams import catalog

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, os.pardir, os.pardir, "models", "normal_qq")

def one(e_p):
    return backtest(
        investigation_dir=HERE,
        model=MODEL,
        streams=("chainlink",),
        quote=QuoteParams(e_p=e_p, rpl_p=0.0005, max_pos=50.0, shares=10.0),
        execn=ExecConfig(),
        sample=Sample(require=("spot", "chainlink")),
        output=Output(seeds=(0, 1, 2)),
        episodes=EPISODES,
    )


if __name__ == "__main__":
    catalog.install()
    EPISODES = load_episodes(days=("2026-08-20",), streams=("chainlink",))

    # Running and reporting are separate steps. The harness writes artefacts;
    # the figures are drawn afterwards, across as many runs as you like.
    runs = {f"e_p={e}": one(e).run_dir for e in (0.01, 0.02, 0.03)}

    from harness.report import cumulative_pnl, load_runs
    cumulative_pnl(load_runs(runs), os.path.join(HERE, "cum_pnl.png"))
```

Then remove the `sys.path.insert` line from every moved runner and plot script,
and point each at `MODEL` rather than its local block copies. Delete the copied
`fair.py`/`vol.py`/`f.py`/`link.py` from `2026-09-09-normal-qq-eval` and
`2026-09-09-vol-fixed` **only after** confirming `models/normal_qq` is
byte-identical to them:

```bash
for f in fair vol f link; do
  diff investigations/2026-09-09-normal-qq-eval/$f.py models/normal_qq/$f.py || echo "DIFFERS: $f"
done
```

If any differ, stop and report rather than deleting — a difference means the
copies had already drifted and the canonical version is a decision, not a
formality.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_template_runs.py -q`
Expected: PASS, 3 passed. Full suite: 264.

Then confirm a real runner still works end to end:
Run: `python investigations/2026-09-09-smoke/run.py`
Expected: prints a run directory and a headline dict.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add -A investigations backtesting_5m/harness/tests/test_template_runs.py
git commit -m "refactor: research moves to ../investigations, on the public API

<trailer>"
```

---

## Task 11: Collapse the spot constants into catalog entries

**Files:**
- Modify: `harness/paths.py`, `harness/streams/catalog.py`
- Test: `harness/tests/test_catalog_panels.py`

**Interfaces:**
- Consumes: `register`.
- Produces: catalog names `spot_usd`, `spot_oracle_window`, `spot_london_usdt`, `spot_legacy_usdt`; `paths.SPOT*` constants remain as thin aliases.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_catalog_panels.py`:

```python
"""Pins that the spot panels are catalog entries, and that the USDT one says so.

There are four physical panels behind five constants. The London panel is
BTC/USDT UNCORRECTED -- roughly +$56 against the BTC/USD oracle these markets
settle on -- and is usable only because fair.py learns the basis. A consumer
assuming USD would be that much wrong.
"""
from harness.streams import catalog, clear_registry, registered, resolve
import pytest


@pytest.fixture(autouse=True)
def _clean():
    clear_registry()
    yield
    clear_registry()


def test_the_catalog_registers_every_spot_panel():
    catalog.install()
    for n in ("spot_usd", "spot_oracle_window", "spot_london_usdt"):
        assert n in registered()


def test_the_usdt_panel_is_named_so_it_cannot_be_mistaken_for_usd():
    catalog.install()
    assert "usdt" in "spot_london_usdt"
    assert "usd" == resolve("spot_usd").name[-3:]


def test_the_legacy_panel_is_registered_but_marked():
    catalog.install()
    assert "spot_legacy_usdt" in registered()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_catalog_panels.py -q`
Expected: FAIL — `AssertionError` on `spot_usd` not registered

- [ ] **Step 3: Write minimal implementation**

Extend `catalog.install()`:

```python
    #: BTC/USD. The default. Venue mid less the capture's usdt_basis; sits
    #: +$4.50 (sd 7.44) against the Chainlink oracle, versus +$43.17 raw.
    register("spot_usd", paths.SPOT, adapter=_spot_adapter("spot_usd"))

    #: BTC/USD, rebuilt over the oracle overlap 2026-08-17..08-21 so a
    #: basis-learning fair block has a Chainlink line for the whole window.
    register("spot_oracle_window", paths.SPOT_ORACLE_WINDOW,
             adapter=_spot_adapter("spot_oracle_window"))

    #: BTC/USDT, UNCORRECTED -- roughly +$56 against the BTC/USD oracle these
    #: markets settle on. panel_100ms carries no usdt_basis column, so no
    #: correction is possible here. Usable ONLY with a fair block that learns
    #: the basis against the oracle itself. A USD-assuming consumer is $56
    #: wrong; the name says usdt for that reason.
    register("spot_london_usdt", paths.SPOT_LONDON_USDT,
             adapter=_spot_adapter("spot_london_usdt"))

    #: Superseded BTC/USDT panel. Registered only so a historical run folder
    #: can be reproduced against the data it actually used. Do not build on it.
    register("spot_legacy_usdt", paths.SPOT_LEGACY_USDT,
             adapter=_spot_adapter("spot_legacy_usdt"))
```

with the shared adapter factory above `install()`:

```python
def _spot_adapter(name):
    """Spot panels predate the standard: they key on (open_ts, t_ms), not
    recv_ns, and are ALREADY on the decision grid.

    `pre_gridded=True` is load-bearing. `grid_stream` treats its time column as
    an absolute epoch; `t_ms` is milliseconds since the market open, so
    gridding one of these would compute a hugely negative offset and silently
    drop every observation. These entries exist for discovery and to put the
    currency warnings in one place -- `load_episodes` reads the panels by its
    existing path.
    """
    return Stream(name=name, time_col="t_ms", time_kind=TimeKind.RECEIPT,
                  time_unit="ms", causal=True, pre_gridded=True)
```

In `paths.py`, replace each panel constant's multi-line comment with a single
line pointing at its catalog entry — the constants become addresses, the
catalog becomes the documentation, and the currency warnings live in exactly
one place.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_catalog_panels.py -q`
Expected: PASS, 3 passed. Full suite: 267.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness/paths.py backtesting_5m/harness/streams/catalog.py backtesting_5m/harness/tests/test_catalog_panels.py
git commit -m "refactor(paths): spot panels become documented catalog entries

<trailer>"
```

---

## Task 12: Sweep the dead surface

**Files:**
- Modify: `harness/core/run.py` (wire `apply_daily_minimum`, or delete it)
- Delete: `data/spot_5m_100ms.parquet.sha256`
- Test: `harness/tests/test_no_dead_surface.py`

**Interfaces:**
- Consumes: `apply_daily_minimum`.
- Produces: either a wired daily-minimum option on `ExecConfig`, or its removal.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_no_dead_surface.py`:

```python
"""Guards against functions that are implemented, tested, and reachable by
nothing.

The whole-branch review found three: apply_daily_minimum, fingerprint_input and
ExecConfig.fees. Green tests over unreachable code is the failure mode per-task
review structurally cannot catch.
"""
import ast
import os

HARNESS = os.path.join(os.path.dirname(__file__), os.pardir)


def _calls_in_package():
    names = set()
    for root, _, files in os.walk(HARNESS):
        if "tests" in root or "__pycache__" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(root, f)).read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    if isinstance(fn, ast.Name):
                        names.add(fn.id)
                    elif isinstance(fn, ast.Attribute):
                        names.add(fn.attr)
    return names


def test_apply_daily_minimum_is_reachable_from_production_code():
    assert "apply_daily_minimum" in _calls_in_package(), (
        "apply_daily_minimum is implemented and tested but called by nothing; "
        "wire it into run() behind an ExecConfig flag, or delete it")


def test_fingerprint_input_is_reachable():
    assert "fingerprint_input" in _calls_in_package()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_no_dead_surface.py -q`
Expected: FAIL on `apply_daily_minimum`

- [ ] **Step 3: Write minimal implementation**

Add the field to `ExecConfig` in `harness/core/config.py`:

```python
    #: Apply the venue's $1.00/day per-stream minimum maker-rebate payout.
    #: Measured with perfect separation over 131 earn-days: smallest paid
    #: $1.0359, largest skipped $0.7209. Off by default because it changes the
    #: headline; when on, the caveats say so.
    apply_daily_minimum: bool = False
```

and in `run()`, immediately after the ledger frame is built and before the
summary is computed:

```python
    if execn.apply_daily_minimum:
        ledger = modules["fees"].apply_daily_minimum(ledger)
    summary["caveats"]["daily_rebate_minimum_applied"] = bool(
        execn.apply_daily_minimum)
```

Then delete the orphan sidecar:

```bash
rm -f data/spot_5m_100ms.parquet.sha256
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_no_dead_surface.py -q`
Expected: PASS, 2 passed. Full suite: 269.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add -A backtesting_5m/harness backtesting_5m/data
git commit -m "chore(harness): wire the daily rebate minimum, guard against dead surface

<trailer>"
```

---

## Task 13: Reconcile the original spec with the code

**Files:**
- Modify: `docs/superpowers/specs/2026-09-09-backtesting-harness-design.md`
- Test: none — documentation

**Interfaces:** none.

- [ ] **Step 1: List every drifted claim**

Read the spec against the code and confirm each of these is now false:

- `ExecConfig.mode = "maker" | "taker" | "both"` — removed; execution is one unified fee-aware policy
- snapping happens in `quote.py` — moved to `execution.py`, applied once after the fee adjustment
- the spot build **measures** a venue↔panel clock offset — superseded; it is a declared assumption, default 0.0, because the two streams carry different events
- `Sample.split` — deleted
- `eff_bid`/`eff_ask` in the ledger are order prices — they are theoretical valuations; the placed price is `price`
- the fill arms `optimistic` and `adverse_lag` differ only in `penetration` — optimistic now also zeroes cancel latency

- [ ] **Step 2: Rewrite the affected sections**

Correct each in place. Add a short "Superseded by" note at the top pointing at
`2026-09-09-stream-registry-and-module-design.md` for the parts this plan
replaces (paths, block copying, the investigation layout).

- [ ] **Step 3: Verify no claim in the spec contradicts the code**

Run: `python -m pytest harness/tests -q` — expect 269 passed, confirming the
code the spec now describes.

- [ ] **Step 4: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/docs/superpowers/specs/2026-09-09-backtesting-harness-design.md
git commit -m "docs: reconcile the original harness spec with the code

A stale spec is worse than no spec: it still described a mode flag that is
gone, snapping in the wrong module, and a measured clock offset that was
superseded by a declared assumption.

<trailer>"
```

---

## After the plan

Two things remain deliberately out of scope, both from the spec:

1. **Event replay.** The `t_ms` time base built in Task 5 is what makes it additive: supply a different `t_ms` array and convert index arithmetic to `searchsorted` in `latency.delay_idx`, `ledger.markout` and the loop. No change to stream declarations, blocks, or the accessor.
2. **The raw NDJSON → parquet converter.** `/z/recording_runs/london_recorder/stream=<name>__hour=<ts>.ndjson.gz` runs to 2026-08-26, five days past the derived files, and carries streams not yet derived (trades, depth diffs, klines, force orders). Task 2's `write_stream` is the writer it should target.
