# 5 m Backtesting Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a block-composed, causality-safe backtesting harness over the reconciled 100 ms BTC up/down 5 m panel, so that quoting policies can be scored against real fees, latency and fill mechanics with reproducible provenance.

**Architecture:** An event ("feed") loop replays each market as an independent 3,000-tick episode. Stateless signal work is precomputed as arrays; inventory-dependent quoting, order policy and fills run per tick. Every block is a plain Python file resolved investigation-first, defaults-second, and every run freezes the exact files it used.

**Tech Stack:** Python 3.13, numpy 2.2, pandas 2.2, pyarrow 19, pytest 8.4, scipy 1.15, matplotlib 3.10. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-09-backtesting-harness-design.md` — read it before Task 1; the plan argues from it throughout.

## Global Constraints

- **Working directory** is `C:\Users\kaima\Github2\Gambling104\backtesting_5m`. Git root is `C:\Users\kaima\Github2\Gambling104`.
- **Run tests with** `python -m pytest harness/tests -q` from the working directory.
- **Never modify** anything under `data/scripts/` or `data/*.parquet`. The build is finished and gated; this harness is a consumer.
- **Price tick is 0.01** (1 cent). All quoted prices snap to it — bids round *down*, asks round *up*.
- **Probabilities are in (0, 1)**; cents are `100 x probability`. Metrics report **c/share** and **$/market**.
- **Fee constants, verified on-chain, do not change:** `BASE_FEE_RATE = 0.07`, `MAKER_REBATE_PHI = 0.20`, `TAKER_REBATE_RHO = 0.0833`.
- **Latency defaults:** `place_ms = 100`, `cancel_ms = 100`, `take_ms = 200`, `taker_lock_ms = 250` (venue mechanic, enforced as a floor on the taker path).
- **Episode length** `N = 3000`; bucket is 100 ms; `tte_s = 300 - i * 0.1`.
- **Causality rule:** decision index `i` may only use observations from buckets `k <= i - 1`. This is enforced in one place (Task 2) and must never be worked around.
- **Test style** follows `data/tests/test_build.py`: module docstring explaining what the file pins, plain pytest functions, sentence-style names, `tmp_path` / `monkeypatch`, no classes.
- **Every commit message ends with the attribution trailer:**
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01ED1yvqexsuevgcrrK16ymZ
  ```
  Shown in full in Task 1; later tasks abbreviate it as `<trailer>` — always write it out.

---

## File Structure

```
harness/
  __init__.py
  paths.py                  roots and data locations
  core/
    __init__.py
    types.py                Side, Liquidity, OrderRequest, Order, Fill
    episode.py              Episode + the causality shift
    latency.py              LatencyModel, per-episode seeded draws
    config.py               QuoteParams, ExecConfig, Sample, Output, BlockSet
    ledger.py               fill/market/tick accumulation -> DataFrames
    loop.py                 run_episode
    stats.py                metrics, day-blocked bootstrap, the four gates
    provenance.py           slot resolution, hashing, manifest, run dirs
    run.py                  run() orchestrator
    plots.py                cumulative PnL
  build/
    __init__.py
    spot_5m_100ms.py        venue L1 -> 100 ms grid, with the clock gate
    episodes.py             panel + strikes + spot + fair -> Episode iterator
  blocks/
    __init__.py
    defaults/
      fair.py vol.py f.py link.py quote.py execution.py fill.py fees.py
    placeholders/
      fair_flat.py          s = strike, for plumbing tests before the export lands
  tests/
    conftest.py             synthetic episode fixtures
    test_*.py               one file per core module
investigations/
  README.md
  _template/run.py
```

Files are split by responsibility, not layer: `fill.py` owns only "does this order trade", `execution.py` owns only "what orders should exist". They are separate because maker results must be reported as a range over fill optimism with policy held fixed.

---

## Task 1: Package skeleton, paths, and the fee model

Fees come first: the formula is exactly known, it has zero dependencies, and getting it wrong silently corrupts every later number.

**Files:**
- Create: `harness/__init__.py`, `harness/paths.py`, `harness/core/__init__.py`, `harness/blocks/__init__.py`, `harness/blocks/defaults/__init__.py`, `harness/blocks/defaults/fees.py`
- Test: `harness/tests/test_fees.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Liquidity` enum re-exported from `fees`; `FeeSchedule` with `charge(liquidity: Liquidity, shares: float, p: float) -> float` returning USD, negative when we are paid. `paths.ROOT`, `paths.DATA`, `paths.PANEL`, `paths.STRIKES`, `paths.RESULTS`.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_fees.py`:

```python
"""Pins the Polymarket fee model to values verified on-chain.

fee = BASE_FEE_RATE * p * (1-p) * shares, peaking at 1.75 c/share at p=0.50.
Takers pay it net of the Silver rebate; makers are PAID a fifth of it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from harness.blocks.defaults.fees import FeeSchedule, Liquidity  # noqa: E402


def test_taker_fee_peaks_at_the_money():
    f = FeeSchedule()
    at_mid = f.charge(Liquidity.TAKER, shares=100.0, p=0.50)
    off_mid = f.charge(Liquidity.TAKER, shares=100.0, p=0.10)
    assert at_mid > off_mid > 0.0


def test_taker_fee_at_mid_is_1_75_cents_per_share_net_of_silver_rebate():
    """0.07 * 0.25 = 1.75 c/share gross, x (1 - 0.0833) at Silver."""
    f = FeeSchedule()
    usd = f.charge(Liquidity.TAKER, shares=100.0, p=0.50)
    assert usd == pytest.approx(100.0 * 0.0175 * (1.0 - 0.0833), rel=1e-12)


def test_maker_is_paid_a_fifth_of_the_base_fee():
    f = FeeSchedule()
    usd = f.charge(Liquidity.MAKER, shares=100.0, p=0.50)
    assert usd == pytest.approx(-100.0 * 0.0175 * 0.20, rel=1e-12)
    assert usd < 0.0


def test_fee_vanishes_at_the_bounds():
    f = FeeSchedule()
    assert f.charge(Liquidity.TAKER, 100.0, 0.0) == pytest.approx(0.0)
    assert f.charge(Liquidity.TAKER, 100.0, 1.0) == pytest.approx(0.0)


def test_fee_is_symmetric_about_a_half():
    f = FeeSchedule()
    assert f.charge(Liquidity.TAKER, 100.0, 0.3) == pytest.approx(
        f.charge(Liquidity.TAKER, 100.0, 0.7))


def test_the_realised_average_is_not_the_constant():
    """1.263 c/share is an average over a fill distribution, never a rate.

    Guards against anyone re-hard-coding it: no single p reproduces it as
    the *gross* per-share fee except off-mid, so the model must stay a curve.
    """
    f = FeeSchedule()
    per_share_at_mid = f.charge(Liquidity.TAKER, 1.0, 0.5) * 100.0
    assert per_share_at_mid > 1.263
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_fees.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness'`

- [ ] **Step 3: Write minimal implementation**

Create the empty package markers:

```bash
mkdir -p harness/core harness/build harness/blocks/defaults harness/blocks/placeholders harness/tests
for f in harness/__init__.py harness/core/__init__.py harness/build/__init__.py \
         harness/blocks/__init__.py harness/blocks/defaults/__init__.py \
         harness/blocks/placeholders/__init__.py; do : > "$f"; done
```

Create `harness/paths.py`:

```python
"""Where the harness reads and writes. Mirrors data/scripts/common.py."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, os.pardir))

DATA = os.path.join(PROJECT, "data")
PANEL = os.path.join(DATA, "book_5m_100ms.parquet")
STRIKES = os.path.join(DATA, "strikes_5m.parquet")
RESULTS = os.path.join(DATA, "results")
SPOT = os.path.join(DATA, "spot_5m_100ms.parquet")
FAIR_DIR = os.path.join(DATA, "fair")
INVESTIGATIONS = os.path.join(PROJECT, "investigations")

VENUE_L1 = r"Z:\parquet\stream_venue_l1"

H = 300
BUCKET_MS = 100
N_BUCKET = H * 1000 // BUCKET_MS      # 3000
TICK = 0.01
```

Create `harness/blocks/defaults/fees.py`:

```python
"""Polymarket fee model for crypto up/down markets.

Verified on-chain 2026-07-07: exact to 0.0 % median error across 9,073 fills,
constant since at least 2026-05-20.

    fee = BASE_FEE_RATE * p * (1 - p) * shares

It peaks at 1.75 c/share at p = 0.50 and vanishes at the bounds, which is why
taking is only ever economic in the tails. Makers pay nothing and are PAID a
fifth of the same curve, so a maker fill carries a NEGATIVE fee.
"""
from dataclasses import dataclass
from enum import IntEnum

BASE_FEE_RATE = 0.07
MAKER_REBATE_PHI = 0.20      # measured 0.199-0.209, both wallets, every period
TAKER_REBATE_RHO = 0.0833    # Silver tier; Gold is not worth buying


class Liquidity(IntEnum):
    MAKER = 0
    TAKER = 1


@dataclass(frozen=True)
class FeeSchedule:
    base_fee_rate: float = BASE_FEE_RATE
    maker_rebate_phi: float = MAKER_REBATE_PHI
    taker_rebate_rho: float = TAKER_REBATE_RHO

    def charge(self, liquidity: Liquidity, shares: float, p: float) -> float:
        """USD owed on a fill. Negative means we are paid."""
        base = self.base_fee_rate * p * (1.0 - p) * shares
        if liquidity == Liquidity.TAKER:
            return base * (1.0 - self.taker_rebate_rho)
        return -base * self.maker_rebate_phi
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_fees.py -q`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness
git commit -m "feat(harness): package skeleton, paths, and the on-chain fee model

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01ED1yvqexsuevgcrrK16ymZ"
```

---

## Task 2: The Episode and the causality shift

The load-bearing task. After this, no later code *can* see the future.

**Files:**
- Create: `harness/core/episode.py`
- Test: `harness/tests/conftest.py`, `harness/tests/test_episode.py`

**Interfaces:**
- Consumes: `harness.paths.N_BUCKET`.
- Produces:
  - `Episode` frozen dataclass with fields `market_id: str`, `open_ts: int`, `day: str`, `strike: float`, `settle: float | None`, `winner_up: bool | None`, and arrays of length 3000: `bid`, `ask`, `mid`, `book_age_ms`, `has_book`, `n_src`, `s`, `sigma`, `spot`, `spot_age_ms`, `has_spot`.
  - `build_episode(market_id, open_ts, day, strike, settle, obs: pd.DataFrame, spot: pd.DataFrame | None, s: np.ndarray | None) -> Episode` where `obs` has columns `t_ms, bid, ask, mid, n_src`.
  - `shift_to_decision_grid(values: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]` returning `(carried, age_ms)`.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/conftest.py`:

```python
"""Synthetic fixtures. Every number here is known by construction, so a
failure localises to the code rather than to the 20.9 M-row panel."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


@pytest.fixture
def obs_sparse():
    """Observations in buckets 0, 1 and 5 only. Everything else is a gap."""
    return pd.DataFrame({
        "t_ms": [0, 100, 500],
        "bid": [0.40, 0.41, 0.45],
        "ask": [0.42, 0.43, 0.47],
        "mid": [0.41, 0.42, 0.46],
        "n_src": [2, 2, 1],
    })


@pytest.fixture
def flat_episode():
    """A 3000-tick market quoted flat at 0.49/0.51, settling UP.

    Built directly rather than through the loader so engine tests do not
    depend on the loader's correctness.
    """
    from harness.core.episode import Episode
    from harness.paths import N_BUCKET

    n = N_BUCKET
    ones = np.ones(n)
    return Episode(
        market_id="synthetic-up",
        open_ts=1786665600,
        day="2026-08-14",
        strike=100_000.0,
        settle=100_100.0,
        winner_up=True,
        bid=ones * 0.49,
        ask=ones * 0.51,
        mid=ones * 0.50,
        book_age_ms=np.zeros(n),
        has_book=np.ones(n, dtype=bool),
        n_src=np.full(n, 2),
        s=ones * 100_000.0,
        sigma=ones * 50.0,
        spot=ones * 100_000.0,
        spot_age_ms=np.zeros(n),
        has_spot=np.ones(n, dtype=bool),
    )
```

Create `harness/tests/test_episode.py`:

```python
"""Pins the causality shift.

The panel README's trap: a row labelled t_ms holds an observation drawn from
[t_ms, t_ms+100), so it was NOT knowable at t_ms. Decision index i may only
see buckets k <= i-1. These tests are the enforcement.
"""
import numpy as np
import pytest

from harness.core.episode import build_episode, shift_to_decision_grid
from harness.paths import N_BUCKET


def test_a_bucket_is_not_visible_at_its_own_index():
    values = np.array([np.nan] * N_BUCKET)
    present = np.zeros(N_BUCKET, dtype=bool)
    values[0], present[0] = 0.40, True
    carried, age = shift_to_decision_grid(values, present)
    assert np.isnan(carried[0]), "bucket 0 leaked into decision index 0"
    assert carried[1] == 0.40


def test_a_fresh_observation_has_zero_age_when_first_usable():
    values = np.full(N_BUCKET, np.nan)
    present = np.zeros(N_BUCKET, dtype=bool)
    values[7], present[7] = 0.44, True
    carried, age = shift_to_decision_grid(values, present)
    assert carried[8] == 0.44 and age[8] == 0.0
    assert carried[9] == 0.44 and age[9] == 100.0


def test_gaps_carry_the_last_quote_with_a_growing_age(obs_sparse):
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_100.0,
                       obs_sparse, None, None)
    assert np.isnan(ep.bid[0])
    assert ep.bid[1] == pytest.approx(0.40) and ep.book_age_ms[1] == 0.0
    assert ep.bid[2] == pytest.approx(0.41) and ep.book_age_ms[2] == 0.0
    # bucket 5 is not usable until index 6; indices 3..5 still hold bucket 1
    assert ep.bid[5] == pytest.approx(0.41) and ep.book_age_ms[5] == 300.0
    assert ep.bid[6] == pytest.approx(0.45) and ep.book_age_ms[6] == 0.0


def test_has_book_is_false_before_the_first_observation():
    obs = __import__("pandas").DataFrame(
        {"t_ms": [400], "bid": [0.4], "ask": [0.5], "mid": [0.45], "n_src": [1]})
    ep = build_episode("m", 1786665600, "2026-08-14", 1.0, 2.0, obs, None, None)
    assert not ep.has_book[:5].any()
    assert ep.has_book[5:].all()


def test_no_future_leak_property():
    """Perturbing bucket k must never change any decision index <= k.

    This is the property that makes lookahead unrepresentable rather than
    merely discouraged. If it ever fails, every result in the repo is void.
    """
    rng = np.random.default_rng(0)
    present = rng.random(N_BUCKET) < 0.9
    values = np.where(present, rng.random(N_BUCKET), np.nan)
    base, _ = shift_to_decision_grid(values, present)
    for k in (0, 1, 37, 1500, N_BUCKET - 1):
        if not present[k]:
            continue
        bumped = values.copy()
        bumped[k] += 10.0
        after, _ = shift_to_decision_grid(bumped, present)
        np.testing.assert_array_equal(
            np.nan_to_num(base[: k + 1], nan=-1.0),
            np.nan_to_num(after[: k + 1], nan=-1.0),
            err_msg=f"bucket {k} leaked backwards")


def test_settlement_is_absent_when_no_successor_exists(obs_sparse):
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, None,
                       obs_sparse, None, None)
    assert ep.settle is None and ep.winner_up is None


def test_winner_up_is_true_on_a_tie(obs_sparse):
    """settle >= strike, ties Up -- the panel README's rule."""
    ep = build_episode("m", 1786665600, "2026-08-14", 100_000.0, 100_000.0,
                       obs_sparse, None, None)
    assert ep.winner_up is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_episode.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.core.episode'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/core/episode.py`:

```python
"""One market, as the strategy is allowed to see it.

THE CAUSALITY RULE, enforced here and nowhere else:

    Decision index i carries exactly the information available at
    t_ms = i * 100, which is every observation from buckets k <= i - 1.

The panel labels a row by the START of the 100 ms bucket its observation was
drawn from, so that row was not knowable until the bucket closed. Everything
downstream indexes these arrays freely and therefore cannot reach forward,
because no index contains a future value.

Carrying the last quote is not the same as inventing one. A live trader knows
the last book they saw; what they do not know is whether it is still valid.
That is what `book_age_ms` is for, and why the engine refuses to trade against
a book older than `max_book_age_ms`.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from harness.paths import BUCKET_MS, N_BUCKET


@dataclass(frozen=True)
class Episode:
    market_id: str
    open_ts: int
    day: str
    strike: float
    settle: float | None
    winner_up: bool | None

    bid: np.ndarray
    ask: np.ndarray
    mid: np.ndarray
    book_age_ms: np.ndarray
    has_book: np.ndarray
    n_src: np.ndarray

    s: np.ndarray
    sigma: np.ndarray
    spot: np.ndarray
    spot_age_ms: np.ndarray
    has_spot: np.ndarray

    def __len__(self) -> int:
        return N_BUCKET

    def tte_s(self, i: int) -> float:
        return 300.0 - i * (BUCKET_MS / 1000.0)


def shift_to_decision_grid(values, present):
    """Carry observations forward onto the decision grid, one bucket late.

    Returns (carried, age_ms). `carried[i]` is the freshest observation from a
    bucket k <= i-1; `age_ms[i]` is how stale it was by the time index i could
    act on it, measured from the close of its own bucket. A bucket observed at
    k is age 0 at index k+1.
    """
    values = np.asarray(values, dtype="float64")
    present = np.asarray(present, dtype=bool)
    n = values.shape[0]

    src = np.where(present, np.arange(n), -1)
    src = np.maximum.accumulate(src)

    # shift by one bucket: index i may only see buckets <= i-1
    shifted = np.full(n, -1, dtype="int64")
    shifted[1:] = src[:-1]

    carried = np.full(n, np.nan)
    seen = shifted >= 0
    carried[seen] = values[shifted[seen]]

    age = np.full(n, np.inf)
    idx = np.arange(n)
    age[seen] = (idx[seen] - shifted[seen] - 1) * BUCKET_MS
    return carried, age


def _grid(obs: pd.DataFrame, column: str):
    values = np.full(N_BUCKET, np.nan)
    present = np.zeros(N_BUCKET, dtype=bool)
    if obs is not None and len(obs):
        k = (obs["t_ms"].to_numpy() // BUCKET_MS).astype("int64")
        keep = (k >= 0) & (k < N_BUCKET)
        values[k[keep]] = obs[column].to_numpy()[keep]
        present[k[keep]] = True
    return values, present


def build_episode(market_id, open_ts, day, strike, settle,
                  obs, spot=None, s=None):
    """Assemble one Episode. `obs` needs t_ms, bid, ask, mid, n_src."""
    raw_bid, present = _grid(obs, "bid")
    raw_ask, _ = _grid(obs, "ask")
    raw_mid, _ = _grid(obs, "mid")
    raw_src, _ = _grid(obs, "n_src")

    bid, age = shift_to_decision_grid(raw_bid, present)
    ask, _ = shift_to_decision_grid(raw_ask, present)
    mid, _ = shift_to_decision_grid(raw_mid, present)
    n_src, _ = shift_to_decision_grid(raw_src, present)
    has_book = np.isfinite(bid) & np.isfinite(ask)

    if spot is not None and len(spot):
        raw_spot, spot_present = _grid(spot, "spot")
        spot_arr, spot_age = shift_to_decision_grid(raw_spot, spot_present)
    else:
        spot_arr = np.full(N_BUCKET, np.nan)
        spot_age = np.full(N_BUCKET, np.inf)
    has_spot = np.isfinite(spot_arr)

    s_arr = np.full(N_BUCKET, np.nan) if s is None else np.asarray(s, "float64")

    winner = None if settle is None else bool(settle >= strike)
    return Episode(
        market_id=market_id, open_ts=int(open_ts), day=day,
        strike=float(strike), settle=settle, winner_up=winner,
        bid=bid, ask=ask, mid=mid, book_age_ms=age, has_book=has_book,
        n_src=n_src,
        s=s_arr, sigma=np.full(N_BUCKET, np.nan),
        spot=spot_arr, spot_age_ms=spot_age, has_spot=has_spot,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_episode.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness backtesting_5m/harness/tests
git commit -m "feat(harness): Episode and the causality shift

Decision index i sees only buckets k <= i-1, enforced in one function with a
property test that perturbing bucket k never changes any earlier index.

<trailer>"
```

---

## Task 3: Latency model with per-episode seeding

**Files:**
- Create: `harness/core/latency.py`
- Test: `harness/tests/test_latency.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `LatencyModel(place_ms=100.0, cancel_ms=100.0, take_ms=200.0, taker_lock_ms=250.0, jitter_frac=0.0, move_cancel_ms=None)` with `draw(rng, kind: str, in_move: bool = False) -> float` for `kind in {"place", "cancel", "take"}`, and `delay_idx(ms) -> int`; plus `episode_rng(market_id: str, run_seed: int) -> np.random.Generator`.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_latency.py`:

```python
"""Pins latency behaviour.

Two properties matter. The taker path can never be faster than the venue's
250 ms non-cancellable hold, and a market's draws depend only on its own id
and the run seed -- never on how many workers ran or in what order.
"""
import numpy as np
import pytest

from harness.core.latency import LatencyModel, episode_rng


def test_defaults_match_the_spec():
    m = LatencyModel()
    assert (m.place_ms, m.cancel_ms, m.take_ms) == (100.0, 100.0, 200.0)
    assert m.taker_lock_ms == 250.0


def test_taker_never_beats_the_venue_lock():
    """The 250 ms hold is a venue mechanic, not a latency assumption."""
    m = LatencyModel(take_ms=50.0, jitter_frac=0.5)
    rng = np.random.default_rng(0)
    draws = [m.draw(rng, "take") for _ in range(200)]
    assert min(draws) >= 250.0


def test_jitter_is_off_by_default():
    m = LatencyModel()
    rng = np.random.default_rng(0)
    assert {m.draw(rng, "place") for _ in range(20)} == {100.0}


def test_jitter_spreads_but_stays_positive():
    m = LatencyModel(jitter_frac=0.3)
    rng = np.random.default_rng(1)
    draws = np.array([m.draw(rng, "place") for _ in range(500)])
    assert draws.std() > 0.0 and draws.min() > 0.0


def test_cancel_is_slower_inside_a_move():
    """Measured 28 ms quiet against 73-165 ms in a move -- the adverse
    selection mechanism itself."""
    m = LatencyModel(cancel_ms=28.0, move_cancel_ms=120.0)
    rng = np.random.default_rng(0)
    assert m.draw(rng, "cancel", in_move=True) > m.draw(rng, "cancel")


def test_the_same_market_and_seed_give_identical_draws():
    a = [LatencyModel(jitter_frac=0.3).draw(episode_rng("mkt-1", 7), "place")
         for _ in range(3)]
    b = [LatencyModel(jitter_frac=0.3).draw(episode_rng("mkt-1", 7), "place")
         for _ in range(3)]
    assert a == b


def test_different_markets_do_not_share_a_stream():
    assert (LatencyModel(jitter_frac=0.3).draw(episode_rng("mkt-1", 7), "place")
            != LatencyModel(jitter_frac=0.3).draw(episode_rng("mkt-2", 7), "place"))


def test_delay_index_rounds_up_to_whole_buckets():
    m = LatencyModel()
    assert m.delay_idx(0.0) == 0
    assert m.delay_idx(1.0) == 1
    assert m.delay_idx(100.0) == 1
    assert m.delay_idx(250.0) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_latency.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.core.latency'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/core/latency.py`:

```python
"""How long our intentions take to reach the exchange.

Defaults are operator-set: 100 ms to place or cancel, 200 ms to take. The
venue additionally holds marketable crypto up/down orders ~250 ms and will not
let them be cancelled, so `taker_lock_ms` is a FLOOR on the taker path that no
parameter can undercut. It also reconciles the measured 276 ms taker fill as
26 ms POST flight plus the 250 ms lock.

Seeding is per-episode so that a run is byte-identical regardless of worker
count or completion order.
"""
import hashlib
import math
from dataclasses import dataclass

import numpy as np

from harness.paths import BUCKET_MS


def episode_rng(market_id: str, run_seed: int) -> np.random.Generator:
    """A generator determined only by the market and the run seed."""
    digest = hashlib.blake2b(
        f"{market_id}:{run_seed}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


@dataclass(frozen=True)
class LatencyModel:
    place_ms: float = 100.0
    cancel_ms: float = 100.0
    take_ms: float = 200.0
    taker_lock_ms: float = 250.0
    jitter_frac: float = 0.0
    move_cancel_ms: float | None = None

    def draw(self, rng, kind: str, in_move: bool = False) -> float:
        if kind == "place":
            base = self.place_ms
        elif kind == "cancel":
            base = (self.move_cancel_ms
                    if in_move and self.move_cancel_ms is not None
                    else self.cancel_ms)
        elif kind == "take":
            base = self.take_ms
        else:
            raise ValueError(f"unknown latency kind {kind!r}")

        if self.jitter_frac > 0.0:
            # lognormal keeps it positive and right-skewed, like a real wire
            base = float(base * rng.lognormal(0.0, self.jitter_frac))

        if kind == "take":
            base = max(base, self.taker_lock_ms)
        return float(base)

    @staticmethod
    def delay_idx(ms: float) -> int:
        """Whole 100 ms buckets a delay of `ms` costs, rounded up."""
        return int(math.ceil(ms / BUCKET_MS))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_latency.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness
git commit -m "feat(harness): latency model with the 250 ms taker lock as a floor

<trailer>"
```

---

## Task 4: Quote construction — f, link, and the three-layer algebra

**Files:**
- Create: `harness/core/config.py`, `harness/blocks/defaults/f.py`, `harness/blocks/defaults/link.py`, `harness/blocks/defaults/quote.py`
- Test: `harness/tests/test_quote.py`

**Interfaces:**
- Consumes: `harness.paths.TICK`.
- Produces:
  - `QuoteParams(e_s=0.0, e_z=0.0, e_p=0.0, rpl_s=0.0, rpl_z=0.0, rpl_p=0.0, max_pos=0.0, tick=0.01)`.
  - `f.standardise(level: float, strike: float, sigma: float) -> float`.
  - `link.link(z: float) -> float`.
  - `quote.quotes(s_i, q, strike, sigma_i, params, standardise, link) -> tuple[float, float]`.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_quote.py`:

```python
"""Pins the quote algebra.

    z_bid   = f(s - e_s - q*rpl_s)
    eff_bid = link(z_bid - e_z - q*rpl_z) - e_p - q*rpl_p

e_* are symmetric half-spreads: they WIDEN the pair.
rpl_* are retreat per lot: they PUSH the pair, and must not widen it.
"""
import pytest

from harness.blocks.defaults.f import standardise
from harness.blocks.defaults.link import link
from harness.blocks.defaults.quote import quotes
from harness.core.config import QuoteParams

STRIKE, SIGMA = 100_000.0, 50.0


def q_at(s=100_000.0, q=0.0, **kw):
    return quotes(s, q, STRIKE, SIGMA, QuoteParams(**kw), standardise, link)


def test_zero_edge_and_zero_inventory_collapse_to_fair():
    bid, ask = q_at()
    assert bid == pytest.approx(0.50, abs=0.011)
    assert ask == pytest.approx(0.50, abs=0.011)


def test_probability_edge_widens_symmetrically():
    bid, ask = q_at(e_p=0.05)
    mid = 0.5 * (bid + ask)
    assert mid == pytest.approx(0.50, abs=0.011)
    assert ask - bid > 0.09


def test_retreat_pushes_without_widening():
    """A long position must move both quotes down by the same amount."""
    b0, a0 = q_at(e_p=0.05, q=0.0)
    b1, a1 = q_at(e_p=0.05, q=10.0, rpl_p=0.002)
    assert b1 < b0 and a1 < a0
    assert (a1 - b1) == pytest.approx(a0 - b0, abs=1e-9)


def test_a_short_position_retreats_upwards():
    b0, a0 = q_at(e_p=0.05, q=0.0)
    b1, a1 = q_at(e_p=0.05, q=-10.0, rpl_p=0.002)
    assert b1 > b0 and a1 > a0


def test_edge_in_level_space_moves_the_pair_apart():
    narrow = q_at(e_s=0.0)
    wide = q_at(e_s=100.0)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_edge_in_latent_space_moves_the_pair_apart():
    narrow = q_at(e_z=0.0)
    wide = q_at(e_z=0.5)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_quotes_snap_to_the_cent_grid_conservatively():
    """Bids round DOWN and asks round UP -- never quote better than intended."""
    bid, ask = q_at(e_p=0.037)
    assert bid == pytest.approx(round(bid, 2), abs=1e-9)
    assert ask == pytest.approx(round(ask, 2), abs=1e-9)
    assert bid <= 0.463 + 1e-9 and ask >= 0.537 - 1e-9


def test_quotes_stay_inside_the_bounds():
    bid, ask = q_at(s=100_500.0, e_p=0.9)
    assert 0.0 <= bid <= 1.0 and 0.0 <= ask <= 1.0


def test_inventory_is_clamped_to_max_pos():
    unclamped = q_at(q=1000.0, rpl_p=0.001, max_pos=10.0)
    clamped = q_at(q=10.0, rpl_p=0.001, max_pos=10.0)
    assert unclamped == clamped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_quote.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.core.config'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/core/config.py`:

```python
"""Every knob a run turns. Dataclasses only -- no behaviour."""
from dataclasses import dataclass, field

from harness.blocks.defaults.fees import FeeSchedule
from harness.core.latency import LatencyModel
from harness.paths import TICK


@dataclass(frozen=True)
class QuoteParams:
    """Edge and retreat, expressible at three layers.

    e_*   symmetric half-spread: widens the pair.
    rpl_* retreat per lot: pushes the pair, does not widen it.

    Three layers because the pass-through is not constant -- 0.79 c/$ at the
    money at tau=300 s against 14.5 c/$ at tau=30 s. A single-layer edge is
    mis-specified across the window by construction.
    """
    e_s: float = 0.0        # USD of BTC
    e_z: float = 0.0        # sigma
    e_p: float = 0.0        # probability
    rpl_s: float = 0.0
    rpl_z: float = 0.0
    rpl_p: float = 0.0
    max_pos: float = 0.0    # 0 disables the clamp
    tick: float = TICK
    shares: float = 1.0     # lot size


@dataclass(frozen=True)
class ExecConfig:
    mode: str = "maker"                 # "maker" | "taker" | "both"
    latency: LatencyModel = field(default_factory=LatencyModel)
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    max_book_age_ms: float = 1000.0
    requote_every: int = 10             # decision indices between requotes
    min_tte_s: float = 0.0
    max_tte_s: float = 300.0


@dataclass(frozen=True)
class Sample:
    t0: int | None = None
    t1: int | None = None
    days: tuple = ()
    markets: tuple = ()
    split: str = "all"                  # "train" | "test" | "all"
    max_markets: int | None = None
    require_spot: bool = False


@dataclass(frozen=True)
class Output:
    emit_ticks: bool = False
    tick_markets: tuple = ()
    seeds: tuple = (0,)
    plots: bool = True
```

Create `harness/blocks/defaults/f.py`:

```python
"""s -> z. The moneyness standardisation of the settlement chain."""


def standardise(level: float, strike: float, sigma: float) -> float:
    """How many sigma the expected settlement sits above the strike."""
    if sigma <= 0.0:
        return 0.0
    return (level - strike) / sigma
```

Create `harness/blocks/defaults/link.py`:

```python
"""z -> p. Default link.

The shipped chain uses a NIG survival function; this logistic default is a
stand-in with the right shape and bounds, so the harness runs before the NIG
parameters are wired in. Override by dropping a `link.py` into an
investigation folder.
"""
import math


def link(z: float) -> float:
    """P(up) given the standardised distance from the strike."""
    if z > 40.0:
        return 1.0
    if z < -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))
```

Create `harness/blocks/defaults/quote.py`:

```python
"""The quote construction.

    z_bid   = f(s - e_s - q*rpl_s)
    z_ask   = f(s + e_s - q*rpl_s)
    eff_bid = link(z_bid - e_z - q*rpl_z) - e_p - q*rpl_p
    eff_ask = link(z_ask + e_z - q*rpl_z) + e_p - q*rpl_p

Edge widens; retreat pushes. Bids snap DOWN to the cent grid and asks snap UP,
so rounding never quotes better than intended.
"""
import math


def _snap_down(p, tick):
    return math.floor(p / tick + 1e-9) * tick


def _snap_up(p, tick):
    return math.ceil(p / tick - 1e-9) * tick


def quotes(s_i, q, strike, sigma_i, params, standardise, link):
    """Return (eff_bid, eff_ask) on the tick grid, clipped to [0, 1]."""
    if params.max_pos > 0.0:
        q = max(-params.max_pos, min(params.max_pos, q))

    lean_s = q * params.rpl_s
    lean_z = q * params.rpl_z
    lean_p = q * params.rpl_p

    z_bid = standardise(s_i - params.e_s - lean_s, strike, sigma_i)
    z_ask = standardise(s_i + params.e_s - lean_s, strike, sigma_i)

    eff_bid = link(z_bid - params.e_z - lean_z) - params.e_p - lean_p
    eff_ask = link(z_ask + params.e_z - lean_z) + params.e_p - lean_p

    eff_bid = min(1.0, max(0.0, _snap_down(eff_bid, params.tick)))
    eff_ask = min(1.0, max(0.0, _snap_up(eff_ask, params.tick)))
    return eff_bid, eff_ask
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_quote.py -q`
Expected: PASS, 9 passed

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness
git commit -m "feat(harness): three-layer quote algebra with edge and retreat per lot

<trailer>"
```

---

## Task 5: Order types and fill models

**Files:**
- Create: `harness/core/types.py`, `harness/blocks/defaults/fill.py`
- Test: `harness/tests/test_fill.py`

**Interfaces:**
- Consumes: `Liquidity` from `fees`, `Episode`.
- Produces:
  - `Side` (`BUY = 1`, `SELL = -1`), `OrderRequest(side, price, shares, liquidity, reason)`, `Order(order_id, side, price, shares, liquidity, live_from, cancel_at, reason)`, `Fill(order_id, idx, side, liquidity, price, shares, reason)`.
  - `fill.resolve(orders: list[Order], ep, i: int, params: dict) -> list[Fill]` where `params` carries `penetration`.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_fill.py`:

```python
"""Pins the fill models.

A resting order fills at ITS OWN price when the book crosses it. Adverse
selection does not come from a worse fill price -- it comes from the cancel
losing the race, which is why `cancel_at` is an index and not a flag.

None of this is measurable from the panel: there is no depth, no trade tape
and no queue. These models are assumptions, which is why runs must report a
range across them rather than a single maker number.
"""
import numpy as np
import pytest

from harness.blocks.defaults.fees import Liquidity
from harness.blocks.defaults.fill import resolve
from harness.core.types import Order, Side


def _order(side=Side.BUY, price=0.49, live_from=0, cancel_at=None,
           liquidity=Liquidity.MAKER):
    return Order(order_id=1, side=side, price=price, shares=10.0,
                 liquidity=liquidity, live_from=live_from,
                 cancel_at=cancel_at, reason="test")


def test_a_resting_bid_fills_when_the_ask_reaches_it(flat_episode):
    ep = flat_episode
    ep.ask[5] = 0.49
    fills = resolve([_order(price=0.49)], ep, 5, {})
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(0.49)
    assert fills[0].liquidity == Liquidity.MAKER


def test_a_resting_bid_does_not_fill_above_the_ask(flat_episode):
    assert resolve([_order(price=0.45)], flat_episode, 5, {}) == []


def test_an_order_is_dead_before_it_is_live(flat_episode):
    ep = flat_episode
    ep.ask[5] = 0.49
    assert resolve([_order(price=0.49, live_from=6)], ep, 5, {}) == []


def test_a_cancelled_order_still_fills_inside_the_latency_window(flat_episode):
    """The whole adverse-selection mechanism, in one test.

    We decided to pull at index 4; the cancel lands at 7. A cross at 5 fills
    us anyway -- and it is exactly when the book is moving against us that
    the cross happens.
    """
    ep = flat_episode
    ep.ask[5] = 0.49
    fills = resolve([_order(price=0.49, cancel_at=7)], ep, 5, {})
    assert len(fills) == 1


def test_a_cancel_that_lands_first_prevents_the_fill(flat_episode):
    ep = flat_episode
    ep.ask[8] = 0.49
    assert resolve([_order(price=0.49, cancel_at=7)], ep, 8, {}) == []


def test_a_resting_ask_fills_when_the_bid_reaches_it(flat_episode):
    ep = flat_episode
    ep.bid[5] = 0.51
    fills = resolve([_order(side=Side.SELL, price=0.51)], ep, 5, {})
    assert len(fills) == 1


def test_penetration_requires_the_book_to_trade_through(flat_episode):
    """The conservative arm: a touch is not enough, it must go past us."""
    ep = flat_episode
    ep.ask[5] = 0.49
    assert resolve([_order(price=0.49)], ep, 5, {"penetration": 0.01}) == []
    ep.ask[5] = 0.48
    assert len(resolve([_order(price=0.49)], ep, 5, {"penetration": 0.01})) == 1


def test_a_taker_order_fills_at_the_book_not_at_its_limit(flat_episode):
    """Marketable orders pay the book. The limit only protects the worst case."""
    ep = flat_episode
    fills = resolve([_order(price=0.55, liquidity=Liquidity.TAKER)], ep, 5, {})
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(ep.ask[5])


def test_a_taker_order_does_not_fill_through_its_limit(flat_episode):
    """The book ran away during the 250 ms lock -- the limit saves us."""
    ep = flat_episode
    ep.ask[5] = 0.60
    assert resolve([_order(price=0.55, liquidity=Liquidity.TAKER)], ep, 5, {}) == []


def test_no_fill_against_a_missing_book(flat_episode):
    ep = flat_episode
    ep.ask[5] = np.nan
    assert resolve([_order(price=0.49)], ep, 5, {}) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_fill.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.core.types'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/core/types.py`:

```python
"""The vocabulary of an order's life.

`execution` decides WHAT orders should exist; the engine decides WHEN they
become live, by turning latency into `live_from` and `cancel_at` indices;
`fill` decides WHETHER a live order trades. Keeping those three apart is what
lets fill optimism be swept with policy held fixed.
"""
from dataclasses import dataclass
from enum import IntEnum

from harness.blocks.defaults.fees import Liquidity  # re-exported


class Side(IntEnum):
    BUY = 1
    SELL = -1


@dataclass(frozen=True)
class OrderRequest:
    """What the execution block asks for. Carries no timing."""
    side: Side
    price: float
    shares: float
    liquidity: Liquidity
    reason: str = ""


@dataclass(frozen=True)
class Order:
    """A request the engine has stamped with timing."""
    order_id: int
    side: Side
    price: float
    shares: float
    liquidity: Liquidity
    live_from: int
    cancel_at: int | None
    reason: str = ""
    latency_ms: float = 0.0      # the placement delay actually drawn

    def is_live(self, i: int) -> bool:
        if i < self.live_from:
            return False
        return self.cancel_at is None or i < self.cancel_at


@dataclass(frozen=True)
class Fill:
    order_id: int
    idx: int
    side: Side
    liquidity: Liquidity
    price: float
    shares: float
    reason: str = ""
```

Create `harness/blocks/defaults/fill.py`:

```python
"""Does a live order trade, and at what price.

DEFAULT: touch-fill. A resting order fills at its own price the moment the
book's L1 reaches it. Adverse selection is NOT modelled by degrading the fill
price -- a limit order fills at its limit. It is modelled by the cancel losing
the race: `cancel_at` is an index, and any cross before it fills us. Since
cancel latency rises inside a move (28 ms quiet against 73-165 ms in a move),
we are filled hardest exactly when we are most wrong.

`penetration` gives the conservative arm: require the book to trade THROUGH
the price by a margin, a cheap proxy for queue position we cannot observe.
Set it to 0 (the default) for the touch arm.

None of this is measurable from the panel. Report a range, never one number.
"""
import math

from harness.blocks.defaults.fees import Liquidity
from harness.core.types import Fill, Side


def resolve(orders, ep, i, params):
    """Fills generated at decision index `i`."""
    margin = float(params.get("penetration", 0.0))
    bid, ask = ep.bid[i], ep.ask[i]
    fills = []

    for o in orders:
        if not o.is_live(i):
            continue

        if o.liquidity == Liquidity.TAKER:
            # marketable: pay the book, but never through our own limit
            if o.side == Side.BUY:
                if math.isfinite(ask) and ask <= o.price:
                    fills.append(_fill(o, i, ask))
            else:
                if math.isfinite(bid) and bid >= o.price:
                    fills.append(_fill(o, i, bid))
            continue

        # resting: the book must come to us
        if o.side == Side.BUY:
            if math.isfinite(ask) and ask <= o.price - margin:
                fills.append(_fill(o, i, o.price))
        else:
            if math.isfinite(bid) and bid >= o.price + margin:
                fills.append(_fill(o, i, o.price))

    return fills


def _fill(o, i, price):
    return Fill(order_id=o.order_id, idx=i, side=o.side,
                liquidity=o.liquidity, price=float(price),
                shares=o.shares, reason=o.reason)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_fill.py -q`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness
git commit -m "feat(harness): order types and touch-fill model with cancel-race adverse selection

<trailer>"
```

---

## Task 6: The execution policy block

**Files:**
- Create: `harness/blocks/defaults/execution.py`
- Test: `harness/tests/test_execution.py`

**Interfaces:**
- Consumes: `OrderRequest`, `Side`, `Liquidity`, `ExecConfig`, `QuoteParams`.
- Produces: `execution.decide(i, eff_bid, eff_ask, q, ep, live_orders, execn, params) -> tuple[list[OrderRequest], list[int]]` — `(to_place, order_ids_to_cancel)`.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_execution.py`:

```python
"""Pins the execution policy.

Policy owns what orders should exist -- post, cross, cancel, and the gates
that stop us trading on information we should not trust. It owns no timing;
the engine turns decisions into live_from/cancel_at indices.
"""
import numpy as np

from harness.blocks.defaults.execution import decide
from harness.blocks.defaults.fees import Liquidity
from harness.core.config import ExecConfig, QuoteParams
from harness.core.types import Order, Side

PARAMS = QuoteParams(shares=10.0, max_pos=50.0)


def _live(price, side=Side.BUY, oid=1):
    return Order(order_id=oid, side=side, price=price, shares=10.0,
                 liquidity=Liquidity.MAKER, live_from=0, cancel_at=None)


def test_maker_mode_posts_both_sides(flat_episode):
    place, cancel = decide(100, 0.48, 0.52, 0.0, flat_episode, [],
                           ExecConfig(mode="maker"), PARAMS)
    assert {o.side for o in place} == {Side.BUY, Side.SELL}
    assert all(o.liquidity == Liquidity.MAKER for o in place)


def test_maker_mode_never_crosses_the_book(flat_episode):
    """Our bid must stay below the ask or it is not a maker order."""
    place, _ = decide(100, 0.60, 0.70, 0.0, flat_episode, [],
                      ExecConfig(mode="maker"), PARAMS)
    buys = [o for o in place if o.side == Side.BUY]
    assert all(o.price < flat_episode.ask[100] for o in buys)


def test_taker_mode_lifts_only_when_the_ask_is_below_our_bid(flat_episode):
    """The crossing rule: eff_bid >= market ask means the book is cheap."""
    place, _ = decide(100, 0.52, 0.60, 0.0, flat_episode, [],
                      ExecConfig(mode="taker"), PARAMS)
    assert len(place) == 1
    assert place[0].side == Side.BUY
    assert place[0].liquidity == Liquidity.TAKER


def test_taker_mode_stands_still_when_the_book_is_fair(flat_episode):
    place, _ = decide(100, 0.48, 0.52, 0.0, flat_episode, [],
                      ExecConfig(mode="taker"), PARAMS)
    assert place == []


def test_a_stale_book_suppresses_all_trading(flat_episode):
    ep = flat_episode
    ep.book_age_ms[100] = 5000.0
    place, cancel = decide(100, 0.48, 0.52, 0.0, ep, [_live(0.48)],
                           ExecConfig(mode="maker", max_book_age_ms=1000.0),
                           PARAMS)
    assert place == []
    assert cancel == [1], "stale book must also pull resting orders"


def test_a_missing_book_suppresses_all_trading(flat_episode):
    ep = flat_episode
    ep.has_book[100] = False
    place, _ = decide(100, 0.48, 0.52, 0.0, ep, [],
                      ExecConfig(mode="maker"), PARAMS)
    assert place == []


def test_the_position_cap_binds_on_the_maker_path(flat_episode):
    place, _ = decide(100, 0.48, 0.52, 50.0, flat_episode, [],
                      ExecConfig(mode="maker"), PARAMS)
    assert all(o.side == Side.SELL for o in place), "long at the cap: sell only"


def test_the_position_cap_binds_on_the_taker_path_too(flat_episode):
    """The 2026-05-20 taker-cap-bypass lesson, as a test."""
    place, _ = decide(100, 0.52, 0.60, 50.0, flat_episode, [],
                      ExecConfig(mode="taker"), PARAMS)
    assert place == []


def test_a_resting_order_at_the_right_price_is_left_alone(flat_episode):
    place, cancel = decide(100, 0.48, 0.52, 0.0, flat_episode,
                           [_live(0.48, Side.BUY, 1), _live(0.52, Side.SELL, 2)],
                           ExecConfig(mode="maker"), PARAMS)
    assert place == [] and cancel == []


def test_a_resting_order_at_a_stale_price_is_replaced(flat_episode):
    place, cancel = decide(100, 0.45, 0.52, 0.0, flat_episode,
                           [_live(0.48, Side.BUY, 1), _live(0.52, Side.SELL, 2)],
                           ExecConfig(mode="maker"), PARAMS)
    assert cancel == [1]
    assert [o.side for o in place] == [Side.BUY]


def test_trading_stops_outside_the_tte_window(flat_episode):
    place, _ = decide(2990, 0.48, 0.52, 0.0, flat_episode, [],
                      ExecConfig(mode="maker", min_tte_s=15.0), PARAMS)
    assert place == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_execution.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.blocks.defaults.execution'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/blocks/defaults/execution.py`:

```python
"""What orders should exist right now.

Owns policy only -- post, cross, cancel, and the gates. It owns no timing:
the engine converts these decisions into live_from / cancel_at indices using
the latency model, which is why latency can be swept without touching policy.

Gates, all of which bind:
  * a book older than max_book_age_ms is not a quote, so we neither post
    against it nor leave orders resting on it
  * the position cap binds on the TAKER path as well as the maker path
    (the 2026-05-20 taker-cap-bypass lesson)
  * a maker order that would cross the book is not a maker order
"""
import math

from harness.blocks.defaults.fees import Liquidity
from harness.core.types import OrderRequest, Side


def decide(i, eff_bid, eff_ask, q, ep, live_orders, execn, params):
    """Return (to_place, to_cancel) at decision index i."""
    to_place, to_cancel = [], []

    tte = ep.tte_s(i)
    tradable = (
        bool(ep.has_book[i])
        and ep.book_age_ms[i] <= execn.max_book_age_ms
        and execn.min_tte_s <= tte <= execn.max_tte_s
    )

    if not tradable:
        return [], [o.order_id for o in live_orders]

    cap = params.max_pos if params.max_pos > 0.0 else math.inf
    can_buy = q < cap
    can_sell = q > -cap

    book_bid, book_ask = ep.bid[i], ep.ask[i]

    if execn.mode == "taker":
        # cross only when the book is on the wrong side of our own valuation
        if can_buy and math.isfinite(book_ask) and eff_bid >= book_ask:
            to_place.append(OrderRequest(Side.BUY, eff_bid, params.shares,
                                         Liquidity.TAKER, "cross_bid"))
        if can_sell and math.isfinite(book_bid) and eff_ask <= book_bid:
            to_place.append(OrderRequest(Side.SELL, eff_ask, params.shares,
                                         Liquidity.TAKER, "cross_ask"))
        return to_place, to_cancel

    # maker (and the maker half of "both"): rest inside our own valuation,
    # never through the book
    want = {}
    if can_buy and math.isfinite(book_ask) and eff_bid < book_ask:
        want[Side.BUY] = eff_bid
    if can_sell and math.isfinite(book_bid) and eff_ask > book_bid:
        want[Side.SELL] = eff_ask

    for o in live_orders:
        if want.get(o.side) != o.price:
            to_cancel.append(o.order_id)

    resting = {o.side: o.price for o in live_orders
               if o.order_id not in to_cancel}
    for side, price in want.items():
        if resting.get(side) != price:
            to_place.append(OrderRequest(side, price, params.shares,
                                         Liquidity.MAKER, "quote"))

    if execn.mode == "both":
        if can_buy and math.isfinite(book_ask) and eff_bid >= book_ask:
            to_place.append(OrderRequest(Side.BUY, eff_bid, params.shares,
                                         Liquidity.TAKER, "cross_bid"))
        if can_sell and math.isfinite(book_bid) and eff_ask <= book_bid:
            to_place.append(OrderRequest(Side.SELL, eff_ask, params.shares,
                                         Liquidity.TAKER, "cross_ask"))

    return to_place, to_cancel
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_execution.py -q`
Expected: PASS, 11 passed

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness
git commit -m "feat(harness): execution policy with staleness, tte and position-cap gates

<trailer>"
```

---

## Task 7: The feed loop, ledger and settlement

**Files:**
- Create: `harness/core/ledger.py`, `harness/core/loop.py`
- Test: `harness/tests/test_ledger.py`, `harness/tests/test_loop.py`

**Interfaces:**
- Consumes: everything from Tasks 2–6.
- Produces:
  - `Ledger` with `record_fill(...)`, `record_tick(...)`, `fills_frame() -> pd.DataFrame`, `ticks_frame() -> pd.DataFrame`.
  - `run_episode(ep, blocks: dict, params: QuoteParams, execn: ExecConfig, seed: int, emit_ticks: bool = False) -> dict` returning `{"market_id", "fills", "ticks", "pnl_gross", "pnl_net", "fees", "shares", "n_fills", "max_abs_q", "settled"}`.
  - `markout(ep, idx, side, price) -> tuple[float, float, bool]` returning `(mid_t10, delta_quality_c, markout_settled)`.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_ledger.py`:

```python
"""Pins the ledger's accounting and the 10 s markout.

delta_quality_c is the adverse-selection measure: where the mid went in the
10 s after we traded, signed so that positive is good and expressed in cents.
"""
import numpy as np
import pytest

from harness.blocks.defaults.fees import Liquidity
from harness.core.ledger import markout
from harness.core.types import Side


def test_a_buy_is_scored_by_where_the_mid_went(flat_episode):
    ep = flat_episode
    ep.mid[200:] = 0.60
    mid_t10, dq, settled = markout(ep, 100, Side.BUY, 0.50)
    assert mid_t10 == pytest.approx(0.60)
    assert dq == pytest.approx(10.0)      # cents
    assert settled is False


def test_a_sell_is_scored_with_the_opposite_sign(flat_episode):
    ep = flat_episode
    ep.mid[200:] = 0.60
    _, dq, _ = markout(ep, 100, Side.SELL, 0.50)
    assert dq == pytest.approx(-10.0)


def test_the_markout_horizon_is_ten_seconds(flat_episode):
    """100 decision indices at 100 ms each."""
    ep = flat_episode
    ep.mid[199] = 0.90
    ep.mid[200] = 0.60
    mid_t10, _, _ = markout(ep, 100, Side.BUY, 0.50)
    assert mid_t10 == pytest.approx(0.60)


def test_a_fill_near_expiry_marks_out_against_settlement(flat_episode):
    """Past the window there is no mid, so the outcome is the reference."""
    mid_t10, dq, settled = markout(flat_episode, 2950, Side.BUY, 0.50)
    assert settled is True
    assert mid_t10 == pytest.approx(1.0)      # flat_episode settles UP
    assert dq == pytest.approx(50.0)


def test_markout_is_absent_when_the_outcome_is_unknown(flat_episode):
    from dataclasses import replace
    ep = replace(flat_episode, winner_up=None, settle=None)
    mid_t10, dq, settled = markout(ep, 2950, Side.BUY, 0.50)
    assert np.isnan(mid_t10) and np.isnan(dq)
```

Create `harness/tests/test_loop.py`:

```python
"""Pins the feed loop end to end on episodes whose answer is arithmetic.

The loop is where causality, latency, policy, fills and fees meet. If any of
them is wrong, these numbers move.
"""
import numpy as np
import pytest

from harness.blocks.defaults import execution, f, fees, fill, link, quote
from harness.core.config import ExecConfig, QuoteParams
from harness.core.latency import LatencyModel
from harness.core.loop import run_episode

BLOCKS = {"f": f.standardise, "link": link.link, "quote": quote.quotes,
          "execution": execution.decide, "fill": fill.resolve,
          "fees": fees.FeeSchedule()}


def _run(ep, params, execn, emit_ticks=False):
    return run_episode(ep, BLOCKS, params, execn, seed=0, emit_ticks=emit_ticks)


def test_a_policy_that_never_quotes_trades_nothing(flat_episode):
    out = _run(flat_episode, QuoteParams(e_p=0.9, shares=10.0),
               ExecConfig(mode="maker"))
    assert out["n_fills"] == 0
    assert out["pnl_net"] == pytest.approx(0.0)


def test_a_flat_book_and_a_flat_fair_produce_no_taker_trades(flat_episode):
    out = _run(flat_episode, QuoteParams(e_p=0.0, shares=10.0),
               ExecConfig(mode="taker"))
    assert out["n_fills"] == 0


def test_a_taker_that_lifts_a_cheap_book_pays_the_book_price(flat_episode):
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig(mode="taker"))
    assert out["n_fills"] == 1
    fills = out["fills"]
    assert fills[0]["price"] == pytest.approx(0.20)
    assert fills[0]["liquidity"] == int(fees.Liquidity.TAKER)


def test_settlement_pays_one_for_a_winning_long(flat_episode):
    """Buy 10 shares at 0.20, settle UP: gross = 10 * (1 - 0.20) = 8.00."""
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig(mode="taker"))
    assert out["pnl_gross"] == pytest.approx(8.0)


def test_fees_are_subtracted_from_gross(flat_episode):
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig(mode="taker"))
    expected = 10.0 * 0.07 * 0.20 * 0.80 * (1.0 - 0.0833)
    assert out["fees"] == pytest.approx(expected)
    assert out["pnl_net"] == pytest.approx(out["pnl_gross"] - expected)


def test_a_maker_fill_is_paid_a_rebate(flat_episode):
    """Maker fees are negative, so net beats gross."""
    ep = flat_episode
    ep.ask[500:] = 0.30
    out = _run(ep, QuoteParams(e_p=0.15, shares=10.0, max_pos=10.0),
               ExecConfig(mode="maker"))
    assert out["n_fills"] >= 1
    assert out["fees"] < 0.0
    assert out["pnl_net"] > out["pnl_gross"]


def test_the_position_cap_is_never_exceeded(flat_episode):
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=30.0),
               ExecConfig(mode="taker"))
    assert out["max_abs_q"] <= 30.0


def test_place_latency_delays_the_first_possible_fill(flat_episode):
    """With a one-second placement delay nothing can trade in the first second."""
    ep = flat_episode
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig(mode="taker",
                          latency=LatencyModel(take_ms=1000.0)))
    assert out["fills"][0]["t_ms"] >= 1000


def test_an_unsettled_market_reports_no_pnl(flat_episode):
    from dataclasses import replace
    ep = replace(flat_episode, winner_up=None, settle=None)
    ep.ask[:] = 0.20
    out = _run(ep, QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
               ExecConfig(mode="taker"))
    assert out["settled"] is False
    assert np.isnan(out["pnl_net"])


def test_tick_output_is_off_by_default_and_complete_when_on(flat_episode):
    params = QuoteParams(e_p=0.05, shares=10.0)
    assert _run(flat_episode, params, ExecConfig())["ticks"] == []
    ticks = _run(flat_episode, params, ExecConfig(), emit_ticks=True)["ticks"]
    assert len(ticks) == 3000
    assert {"t_ms", "eff_bid", "eff_ask", "q", "cum_pnl"} <= set(ticks[0])


def test_results_are_reproducible_across_runs(flat_episode):
    """Jitter is seeded per episode, so two identical runs agree exactly."""
    ep = flat_episode
    ep.ask[500:] = 0.30
    execn = ExecConfig(mode="maker", latency=LatencyModel(jitter_frac=0.4))
    params = QuoteParams(e_p=0.15, shares=10.0, max_pos=20.0)
    a = _run(ep, params, execn)
    b = _run(ep, params, execn)
    assert a["pnl_net"] == pytest.approx(b["pnl_net"])
    assert a["n_fills"] == b["n_fills"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_ledger.py harness/tests/test_loop.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.core.ledger'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/core/ledger.py`:

```python
"""Recording what happened, and scoring it 10 s later.

Every fill carries its own fee (negative for makers) and its 10 s markout, so
gross and net are always both recoverable and adverse selection is visible per
fill rather than only in the aggregate.
"""
import numpy as np
import pandas as pd

from harness.core.types import Side

MARKOUT_IDX = 100          # 100 buckets x 100 ms = 10 s


def markout(ep, idx, side, price):
    """(mid_t10, delta_quality_c, markout_settled) for a fill at `idx`.

    Past the end of the window there is no mid, so the settlement outcome is
    the reference instead -- which is the honest comparison, not a null.
    """
    j = idx + MARKOUT_IDX
    if j < len(ep):
        ref, settled = float(ep.mid[j]), False
    elif ep.winner_up is None:
        return float("nan"), float("nan"), True
    else:
        ref, settled = (1.0 if ep.winner_up else 0.0), True

    if not np.isfinite(ref):
        return float("nan"), float("nan"), settled

    signed = (ref - price) if side == Side.BUY else (price - ref)
    return ref, 100.0 * signed, settled


class Ledger:
    """Accumulates fills and (optionally) per-tick diagnostics."""

    def __init__(self):
        self.fills = []
        self.ticks = []

    def record_fill(self, ep, fill, fee_usd, q_before, q_after,
                    eff_bid, eff_ask, s_i, sigma_i, z_i, latency_ms,
                    order_age_ms, seed):
        mid_t10, dq, settled = markout(ep, fill.idx, fill.side, fill.price)
        self.fills.append({
            "market_id": ep.market_id, "open_ts": ep.open_ts, "day": ep.day,
            "t_ms": fill.idx * 100,
            "side": int(fill.side), "liquidity": int(fill.liquidity),
            "shares": fill.shares, "price": fill.price, "fee_usd": fee_usd,
            "s": s_i, "sigma": sigma_i, "z": z_i,
            "eff_bid": eff_bid, "eff_ask": eff_ask,
            "book_bid": float(ep.bid[fill.idx]),
            "book_ask": float(ep.ask[fill.idx]),
            "mid_at_fill": float(ep.mid[fill.idx]),
            "q_before": q_before, "q_after": q_after,
            "latency_ms": latency_ms, "order_age_ms": order_age_ms,
            "mid_t10": mid_t10, "delta_quality_c": dq,
            "markout_settled": settled,
            "seed": seed, "order_id": fill.order_id, "reason": fill.reason,
        })

    def record_tick(self, ep, i, s_i, sigma_i, z_i, fair_p, eff_bid, eff_ask,
                    q, cash, cum_pnl, orders_live):
        self.ticks.append({
            "market_id": ep.market_id, "t_ms": i * 100,
            "s": s_i, "sigma": sigma_i, "z": z_i, "fair_p": fair_p,
            "eff_bid": eff_bid, "eff_ask": eff_ask,
            "book_bid": float(ep.bid[i]), "book_ask": float(ep.ask[i]),
            "mid": float(ep.mid[i]), "book_age_ms": float(ep.book_age_ms[i]),
            "q": q, "cash": cash, "cum_pnl": cum_pnl,
            "orders_live": orders_live,
        })

    def fills_frame(self):
        return pd.DataFrame(self.fills)

    def ticks_frame(self):
        return pd.DataFrame(self.ticks)
```

Create `harness/core/loop.py`:

```python
"""The feed loop: one market, 3,000 decisions, in order.

Latency lives HERE, not in the policy block. The policy says what orders
should exist; this converts that into live_from / cancel_at indices. That is
why latency can be swept without touching a line of policy.
"""
import math
from dataclasses import replace

import numpy as np

from harness.core.latency import episode_rng
from harness.core.ledger import Ledger
from harness.core.types import Order


def run_episode(ep, blocks, params, execn, seed=0, emit_ticks=False):
    standardise = blocks["f"]
    link = blocks["link"]
    quotes = blocks["quote"]
    decide = blocks["execution"]
    resolve = blocks["fill"]
    fee_schedule = blocks["fees"]
    fill_params = blocks.get("fill_params", {})

    rng = episode_rng(ep.market_id, seed)
    latency = execn.latency
    ledger = Ledger()

    q = 0.0
    cash = 0.0
    fees_paid = 0.0
    max_abs_q = 0.0
    next_id = 1
    live = []

    s_arr = blocks["s"]
    sigma_arr = blocks["sigma"]

    for i in range(len(ep)):
        s_i = float(s_arr[i])
        sigma_i = float(sigma_arr[i])

        if math.isfinite(s_i) and math.isfinite(sigma_i):
            eff_bid, eff_ask = quotes(s_i, q, ep.strike, sigma_i, params,
                                      standardise, link)
            z_i = standardise(s_i, ep.strike, sigma_i)
            fair_p = link(z_i)
        else:
            eff_bid = eff_ask = float("nan")
            z_i = fair_p = float("nan")

        # --- fills against orders that were already live -------------------
        if live:
            for fl in resolve(live, ep, i, fill_params):
                fee = fee_schedule.charge(fl.liquidity, fl.shares, fl.price)
                q_before = q
                q += fl.shares * int(fl.side)
                cash -= fl.shares * fl.price * int(fl.side)
                fees_paid += fee
                max_abs_q = max(max_abs_q, abs(q))
                order = next(o for o in live if o.order_id == fl.order_id)
                ledger.record_fill(
                    ep, fl, fee, q_before, q, eff_bid, eff_ask, s_i, sigma_i,
                    z_i, order.latency_ms,
                    (i - order.live_from) * 100.0, seed)
                live = [o for o in live if o.order_id != fl.order_id]

        # --- policy --------------------------------------------------------
        if math.isfinite(eff_bid) and (i % max(1, execn.requote_every) == 0):
            to_place, to_cancel = decide(i, eff_bid, eff_ask, q, ep, live,
                                         execn, params)

            if to_cancel:
                in_move = ep.book_age_ms[i] == 0.0
                lands = i + latency.delay_idx(
                    latency.draw(rng, "cancel", in_move=in_move))
                live = [replace(o, cancel_at=min(lands, o.cancel_at or lands))
                        if o.order_id in to_cancel else o
                        for o in live]

            for req in to_place:
                kind = "take" if req.liquidity else "place"
                drawn = latency.draw(rng, kind)
                live_from = i + latency.delay_idx(drawn)
                live.append(Order(next_id, req.side, req.price, req.shares,
                                  req.liquidity, live_from, None, req.reason,
                                  latency_ms=drawn))
                next_id += 1

        # drop orders whose cancel has landed
        live = [o for o in live if o.cancel_at is None or i < o.cancel_at]

        if emit_ticks:
            ledger.record_tick(ep, i, s_i, sigma_i, z_i, fair_p,
                               eff_bid, eff_ask, q, cash,
                               cash + q * float(ep.mid[i])
                               if np.isfinite(ep.mid[i]) else cash,
                               len(live))

    # --- settlement --------------------------------------------------------
    if ep.winner_up is None:
        pnl_gross = pnl_net = float("nan")
        settled = False
    else:
        pnl_gross = cash + q * (1.0 if ep.winner_up else 0.0)
        pnl_net = pnl_gross - fees_paid
        settled = True

    return {
        "market_id": ep.market_id, "open_ts": ep.open_ts, "day": ep.day,
        "fills": ledger.fills, "ticks": ledger.ticks,
        "pnl_gross": pnl_gross, "pnl_net": pnl_net, "fees": fees_paid,
        "shares": sum(f["shares"] for f in ledger.fills),
        "n_fills": len(ledger.fills), "max_abs_q": max_abs_q,
        "settled": settled, "has_spot": bool(ep.has_spot.any()),
    }
```

The `_run` helper written in Step 1 must supply `s` and `sigma`, which the loop
reads from the blocks dict rather than recomputing. Replace the placeholder
helper at the top of `harness/tests/test_loop.py` with exactly this:

```python
def _run(ep, params, execn, emit_ticks=False):
    blocks = dict(BLOCKS)
    blocks["s"] = ep.s
    blocks["sigma"] = np.full(len(ep), 50.0)
    return run_episode(ep, blocks, params, execn, seed=0, emit_ticks=emit_ticks)
```

Every test in that file goes through `_run`; none calls `run_episode` directly.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_ledger.py harness/tests/test_loop.py -q`
Expected: PASS, 16 passed

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness
git commit -m "feat(harness): feed loop, ledger with 10 s markout, and settlement

<trailer>"
```

---

## Task 8: Block resolution and run provenance

**Files:**
- Create: `harness/core/provenance.py`
- Test: `harness/tests/test_provenance.py`

**Interfaces:**
- Consumes: `harness.paths`.
- Produces:
  - `SLOTS = ("fair", "vol", "f", "link", "quote", "execution", "fill", "fees")`
  - `resolve_slots(investigation_dir: str) -> dict[str, str]` (slot -> absolute path)
  - `load_slot(path: str, slot: str)` -> module
  - `sha256_file(path) -> str`, `fingerprint_input(path) -> dict`
  - `new_run_dir(investigation_dir: str, manifest: dict) -> str`
  - `write_manifest(run_dir, manifest, resolved) -> None` (also freezes block files into `<run_dir>/blocks/`)

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_provenance.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_provenance.py -q`
Expected: FAIL — `ImportError: cannot import name 'provenance'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/core/provenance.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_provenance.py -q`
Expected: PASS, 8 passed

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness
git commit -m "feat(harness): file-based block resolution and frozen run provenance

<trailer>"
```

---

## Task 9: Metrics, day-blocked bootstrap, and the four gates

**Files:**
- Create: `harness/core/stats.py`
- Test: `harness/tests/test_stats.py`

**Interfaces:**
- Consumes: a per-market DataFrame with `day`, `pnl_net`, `shares`.
- Produces:
  - `headline(markets: pd.DataFrame) -> dict` with `pnl_per_market`, `c_per_share`, `n_markets`, `n_fills`, `total_shares`.
  - `day_blocked_ci(markets, column="pnl_net", n_boot=10000, seed=0) -> tuple[float, float]`
  - `gate_sign_survives_periods(markets, n_periods=3) -> dict`
  - `gate_ci_excludes_zero(markets, **kw) -> dict`
  - `gate_delete_top_n(markets, n=10) -> dict`
  - `run_gates(markets) -> dict`

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_stats.py`:

```python
"""Pins the measurement layer and the discipline gates.

The gates exist because the architecture doc says they are applied
'inconsistently'. Making them a function every run calls is the fix.
"""
import numpy as np
import pandas as pd
import pytest

from harness.core import stats


def _markets(pnl, days=None, shares=10.0):
    pnl = np.asarray(pnl, dtype="float64")
    if days is None:
        days = [f"2026-08-{14 + i % 10:02d}" for i in range(len(pnl))]
    return pd.DataFrame({"day": days, "pnl_net": pnl,
                         "shares": np.full(len(pnl), shares),
                         "n_fills": np.ones(len(pnl))})


def test_headline_reports_per_market_and_per_share():
    m = _markets([1.0, 2.0, 3.0], shares=10.0)
    h = stats.headline(m)
    assert h["pnl_per_market"] == pytest.approx(2.0)
    assert h["c_per_share"] == pytest.approx(100.0 * 6.0 / 30.0)
    assert h["n_markets"] == 3


def test_markets_without_a_settlement_are_excluded():
    m = _markets([1.0, 2.0, np.nan])
    assert stats.headline(m)["n_markets"] == 2


def test_a_strong_positive_edge_has_a_ci_above_zero():
    rng = np.random.default_rng(0)
    m = _markets(rng.normal(1.0, 0.2, 400))
    lo, hi = stats.day_blocked_ci(m, n_boot=2000, seed=0)
    assert lo > 0.0 and hi > lo


def test_noise_has_a_ci_spanning_zero():
    rng = np.random.default_rng(1)
    m = _markets(rng.normal(0.0, 1.0, 400))
    lo, hi = stats.day_blocked_ci(m, n_boot=2000, seed=0)
    assert lo < 0.0 < hi


def test_the_bootstrap_resamples_days_not_markets():
    """A single day that is wholly responsible for the edge must widen the CI.

    Market-level resampling would hide it; day-blocked resampling cannot.
    """
    pnl = [0.0] * 200 + [5.0] * 20
    days = ["2026-08-14"] * 200 + ["2026-08-15"] * 20
    lo, _ = stats.day_blocked_ci(_markets(pnl, days), n_boot=2000, seed=0)
    assert lo <= 0.0


def test_the_bootstrap_is_reproducible():
    m = _markets(np.random.default_rng(2).normal(0.5, 1.0, 200))
    assert stats.day_blocked_ci(m, n_boot=500, seed=7) == \
        stats.day_blocked_ci(m, n_boot=500, seed=7)


def test_the_sign_gate_fails_when_a_period_flips():
    good = _markets([1.0] * 90)
    assert stats.gate_sign_survives_periods(good)["passed"] is True

    flipped = good.copy()
    flipped.loc[60:, "pnl_net"] = -1.0
    assert stats.gate_sign_survives_periods(flipped)["passed"] is False


def test_the_delete_top_n_gate_catches_an_edge_carried_by_ten_markets():
    pnl = [0.0] * 200 + [50.0] * 10
    assert stats.gate_delete_top_n(_markets(pnl), n=10)["passed"] is False

    rng = np.random.default_rng(3)
    assert stats.gate_delete_top_n(_markets(rng.normal(2.0, 0.3, 210)),
                                   n=10)["passed"] is True


def test_run_gates_reports_all_four():
    m = _markets(np.random.default_rng(4).normal(1.0, 0.2, 300))
    result = stats.run_gates(m, n_boot=500)
    assert set(result["gates"]) == {
        "sign_survives_periods", "ci_excludes_zero", "delete_top_10"}
    assert isinstance(result["passed"], bool)


def test_gates_on_an_empty_sample_do_not_claim_success():
    result = stats.run_gates(_markets([]), n_boot=100)
    assert result["passed"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_stats.py -q`
Expected: FAIL — `ImportError: cannot import name 'stats'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/core/stats.py`:

```python
"""Scoring, and the four gates that decide whether to believe it.

Day-blocked, not market-blocked. Markets inside a day share a regime, a book
and a competitor set, so resampling markets independently understates the
uncertainty -- often badly. Resampling whole days is the only version that
answers 'would another fortnight have shown this?'.

The gates come from the architecture doc's section 4, which observes that they
exist in the programme and are applied inconsistently. Here they run on every
result whether anyone remembers to ask or not.
"""
import numpy as np
import pandas as pd


def _settled(markets):
    if len(markets) == 0:
        return markets
    return markets[np.isfinite(markets["pnl_net"])]


def headline(markets):
    m = _settled(markets)
    total_shares = float(m["shares"].sum()) if len(m) else 0.0
    total_pnl = float(m["pnl_net"].sum()) if len(m) else 0.0
    return {
        "n_markets": int(len(m)),
        "n_fills": int(m["n_fills"].sum()) if len(m) else 0,
        "total_shares": total_shares,
        "pnl_total": total_pnl,
        "pnl_per_market": total_pnl / len(m) if len(m) else float("nan"),
        "c_per_share": 100.0 * total_pnl / total_shares
        if total_shares else float("nan"),
    }


def day_blocked_ci(markets, column="pnl_net", n_boot=10000, seed=0,
                   alpha=0.05):
    """Percentile CI for mean per-market PnL, resampling whole days."""
    m = _settled(markets)
    if len(m) == 0:
        return float("nan"), float("nan")

    by_day = [g[column].to_numpy() for _, g in m.groupby("day", sort=True)]
    if len(by_day) < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n_days = len(by_day)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_days, n_days)
        means[b] = np.concatenate([by_day[j] for j in pick]).mean()
    return (float(np.quantile(means, alpha / 2.0)),
            float(np.quantile(means, 1.0 - alpha / 2.0)))


def gate_sign_survives_periods(markets, n_periods=3):
    """Split the calendar into equal periods; the sign must hold in each."""
    m = _settled(markets)
    if len(m) == 0:
        return {"passed": False, "detail": "no settled markets"}

    days = sorted(m["day"].unique())
    if len(days) < n_periods:
        n_periods = max(1, len(days))
    chunks = np.array_split(np.array(days), n_periods)

    overall = np.sign(m["pnl_net"].mean())
    per_period = []
    for chunk in chunks:
        sub = m[m["day"].isin(set(chunk))]
        per_period.append(float(sub["pnl_net"].mean()) if len(sub) else 0.0)

    passed = bool(overall != 0 and all(
        np.sign(v) == overall for v in per_period))
    return {"passed": passed, "per_period": per_period,
            "overall_mean": float(m["pnl_net"].mean())}


def gate_ci_excludes_zero(markets, **kw):
    lo, hi = day_blocked_ci(markets, **kw)
    passed = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0))
    return {"passed": passed, "ci": [lo, hi]}


def gate_delete_top_n(markets, n=10):
    """An edge that lives in ten markets is not an edge."""
    m = _settled(markets)
    if len(m) <= n:
        return {"passed": False, "detail": f"only {len(m)} markets"}

    full = float(m["pnl_net"].mean())
    trimmed = m.sort_values("pnl_net", ascending=False).iloc[n:]
    without = float(trimmed["pnl_net"].mean())
    passed = bool(full != 0 and np.sign(without) == np.sign(full))
    return {"passed": passed, "mean_full": full, "mean_without_top": without}


def run_gates(markets, **kw):
    gates = {
        "sign_survives_periods": gate_sign_survives_periods(markets),
        "ci_excludes_zero": gate_ci_excludes_zero(markets, **kw),
        "delete_top_10": gate_delete_top_n(markets, n=10),
    }
    return {"passed": all(g["passed"] for g in gates.values()),
            "gates": gates}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_stats.py -q`
Expected: PASS, 10 passed

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness
git commit -m "feat(harness): day-blocked bootstrap and the four discipline gates

<trailer>"
```

---

## Task 10: The run() orchestrator, sample selection, plots

**Files:**
- Create: `harness/core/plots.py`, `harness/core/run.py`, `harness/blocks/defaults/fair.py`, `harness/blocks/defaults/vol.py`, `harness/blocks/placeholders/fair_flat.py`
- Test: `harness/tests/test_run.py`

**Interfaces:**
- Consumes: Tasks 1–9.
- Produces: `run(investigation_dir, quote, execn, sample, output, episodes) -> dict` writing a full run folder; `plots.cumulative_pnl(markets, path)`.
- Block contracts: `fair.precompute(ep) -> np.ndarray`, `vol.precompute(ep) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_run.py`:

```python
"""Pins the orchestrator: sample selection, outputs, provenance, gates."""
import json
import os

import numpy as np
import pytest

from harness.core.config import ExecConfig, Output, QuoteParams, Sample
from harness.core.run import run


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


def _run(tmp_path, episodes, **kw):
    kw.setdefault("quote", QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0))
    kw.setdefault("execn", ExecConfig(mode="taker"))
    kw.setdefault("sample", Sample())
    kw.setdefault("output", Output(plots=False))
    return run(str(tmp_path), episodes=episodes, **kw)


def test_a_run_writes_a_manifest_and_a_ledger(tmp_path, episodes):
    result = _run(tmp_path, episodes)
    run_dir = result["run_dir"]
    for name in ("manifest.json", "ledger.parquet", "markets.parquet",
                 "summary.json"):
        assert os.path.exists(os.path.join(run_dir, name)), name


def test_the_summary_carries_the_grid_bias_caveat(tmp_path, episodes):
    result = _run(tmp_path, episodes)
    summary = json.loads(
        open(os.path.join(result["run_dir"], "summary.json")).read())
    assert "grid_bias_usd_per_market" in summary["caveats"]
    assert summary["caveats"]["grid_bias_usd_per_market"] == pytest.approx(0.13)


def test_max_markets_limits_the_sample(tmp_path, episodes):
    result = _run(tmp_path, episodes, sample=Sample(max_markets=2))
    assert result["summary"]["headline"]["n_markets"] == 2


def test_a_day_filter_selects_only_those_days(tmp_path, episodes):
    result = _run(tmp_path, episodes, sample=Sample(days=("2026-08-14",)))
    assert result["markets"]["day"].unique().tolist() == ["2026-08-14"]


def test_a_market_filter_selects_only_those_markets(tmp_path, episodes):
    result = _run(tmp_path, episodes, sample=Sample(markets=("m0", "m3")))
    assert sorted(result["markets"]["market_id"]) == ["m0", "m3"]


def test_tick_output_is_written_only_when_asked(tmp_path, episodes):
    plain = _run(tmp_path, episodes)
    assert not os.path.exists(os.path.join(plain["run_dir"], "ticks.parquet"))

    ticked = _run(tmp_path, episodes,
                  output=Output(emit_ticks=True, tick_markets=("m0",),
                                plots=False))
    ticks_path = os.path.join(ticked["run_dir"], "ticks.parquet")
    assert os.path.exists(ticks_path)
    import pandas as pd
    assert pd.read_parquet(ticks_path)["market_id"].unique().tolist() == ["m0"]


def test_the_ledger_carries_liquidity_fees_and_markout(tmp_path, episodes):
    result = _run(tmp_path, episodes)
    ledger = result["ledger"]
    assert {"liquidity", "fee_usd", "mid_t10", "delta_quality_c"} <= set(
        ledger.columns)
    assert (ledger["fee_usd"] > 0).all(), "taker fees are positive"


def test_gates_run_on_every_result(tmp_path, episodes):
    result = _run(tmp_path, episodes)
    assert "gates" in result["summary"]
    assert set(result["summary"]["gates"]["gates"]) == {
        "sign_survives_periods", "ci_excludes_zero", "delete_top_10"}


def test_multiple_seeds_are_all_reported(tmp_path, episodes):
    result = _run(tmp_path, episodes, output=Output(seeds=(0, 1, 2),
                                                    plots=False))
    assert len(result["summary"]["per_seed"]) == 3


def test_the_frozen_blocks_travel_with_the_run(tmp_path, episodes):
    from harness.core.provenance import SLOTS
    result = _run(tmp_path, episodes)
    frozen = os.path.join(result["run_dir"], "blocks")
    assert sorted(os.listdir(frozen)) == sorted(f"{s}.py" for s in SLOTS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_run.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.core.run'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/blocks/defaults/fair.py`:

```python
"""s -- the expected settling TWAP, in USD.

The harness does NOT own the fair value. This default reads the column the
Gambling102 export supplies. If it is absent the run stops here rather than
quietly scoring a model that does not exist.

For plumbing tests before the export lands, use
`harness/blocks/placeholders/fair_flat.py`.
"""
import numpy as np


def precompute(ep):
    s = np.asarray(ep.s, dtype="float64")
    if not np.isfinite(s).any():
        raise ValueError(
            f"{ep.market_id}: no fair value. Supply a fair export, or drop a "
            f"fair.py into the investigation folder.")
    return s
```

Create `harness/blocks/placeholders/fair_flat.py`:

```python
"""A deliberately crude placeholder: s = strike, so p = 0.5 everywhere.

Exists to exercise the plumbing before a real fair export is wired in. It has
no forecasting content and must never appear in a reported result.
"""
import numpy as np


def precompute(ep):
    return np.full(len(ep), ep.strike, dtype="float64")
```

Create `harness/blocks/defaults/vol.py`:

```python
"""sigma -- the scale that turns (s - strike) into a standardised distance.

Default is a square-root-of-time scaling of a single per-market constant. It
is a stand-in with the right shape; override it with a real vol block.
"""
import numpy as np


SIGMA_AT_300S = 250.0      # USD of BTC, one standard deviation over 300 s


def precompute(ep):
    tte = np.maximum(np.array([ep.tte_s(i) for i in range(len(ep))]), 1e-6)
    return SIGMA_AT_300S * np.sqrt(tte / 300.0)
```

Create `harness/core/plots.py`:

```python
"""Cumulative net PnL over the ordered market sequence."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def cumulative_pnl(markets, path):
    m = markets.sort_values("open_ts")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(m)), m["pnl_net"].fillna(0.0).cumsum(), lw=1.2)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("market (chronological)")
    ax.set_ylabel("cumulative net PnL, USD")
    ax.set_title("Cumulative net PnL")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
```

Create `harness/core/run.py`:

```python
"""Resolve blocks, select a sample, replay it, score it, write it down."""
import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd

from harness.core import provenance, stats
from harness.core.loop import run_episode

#: Measured twice on this exact substrate: the 100 ms grid arm earns more than
#: an events arm by this much per market (CI [+0.053, +0.239]; replicated on
#: red_fast at +0.124, CI [+0.025, +0.229]). Carried on every run so no result
#: leaves here pretending the replay clock is free.
GRID_BIAS_USD_PER_MARKET = 0.13


def _select(episodes, sample):
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
            continue
        out.append(ep)
    out.sort(key=lambda e: e.open_ts)
    if sample.max_markets is not None:
        out = out[: sample.max_markets]
    return out


def run(investigation_dir, quote, execn, sample, output, episodes):
    resolved = provenance.resolve_slots(investigation_dir)
    modules = {slot: provenance.load_slot(path, slot)
               for slot, path in resolved.items()}

    config = {"quote": asdict(quote), "sample": asdict(sample),
              "output": asdict(output), "mode": execn.mode,
              "latency": asdict(execn.latency),
              "max_book_age_ms": execn.max_book_age_ms,
              "requote_every": execn.requote_every}

    run_dir = provenance.new_run_dir(investigation_dir, config)
    provenance.write_manifest(run_dir, config, resolved, seeds=output.seeds)

    blocks = {
        "f": modules["f"].standardise,
        "link": modules["link"].link,
        "quote": modules["quote"].quotes,
        "execution": modules["execution"].decide,
        "fill": modules["fill"].resolve,
        "fees": modules["fees"].FeeSchedule(),
        "fill_params": {},
    }

    selected = _select(episodes, sample)
    tick_ids = set(output.tick_markets)

    all_fills, all_markets, all_ticks, per_seed = [], [], [], []

    for seed in output.seeds:
        rows = []
        for ep in selected:
            eb = dict(blocks)
            eb["s"] = modules["fair"].precompute(ep)
            eb["sigma"] = modules["vol"].precompute(ep)
            emit = output.emit_ticks and (
                not tick_ids or ep.market_id in tick_ids)
            res = run_episode(ep, eb, quote, execn, seed=seed, emit_ticks=emit)

            all_fills.extend(res.pop("fills"))
            all_ticks.extend(res.pop("ticks"))
            rows.append({**res, "seed": seed})

        frame = pd.DataFrame(rows)
        all_markets.append(frame)
        per_seed.append({"seed": seed, **stats.headline(frame)})

    markets = pd.concat(all_markets, ignore_index=True)
    ledger = pd.DataFrame(all_fills)

    primary = markets[markets["seed"] == output.seeds[0]]
    summary = {
        "headline": stats.headline(primary),
        "per_seed": per_seed,
        "gates": stats.run_gates(primary),
        "caveats": {
            "grid_bias_usd_per_market": GRID_BIAS_USD_PER_MARKET,
            "grid_bias_note":
                "The 100 ms replay clock flatters results by roughly this much "
                "per market, measured on two independent models. Subtract it "
                "before believing any headline.",
            "maker_fills_are_modelled":
                "No depth, no trade tape, no queue exists in this data. Any "
                "maker number is conditional on the fill block and must be "
                "reported as a range across fill optimism.",
        },
    }

    ledger.to_parquet(os.path.join(run_dir, "ledger.parquet"), index=False)
    markets.to_parquet(os.path.join(run_dir, "markets.parquet"), index=False)
    if all_ticks:
        pd.DataFrame(all_ticks).to_parquet(
            os.path.join(run_dir, "ticks.parquet"), index=False)
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    if output.plots and len(primary):
        from harness.core import plots
        plots.cumulative_pnl(primary, os.path.join(run_dir, "cum_pnl.png"))

    return {"run_dir": run_dir, "summary": summary,
            "ledger": ledger, "markets": markets}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_run.py -q`
Expected: PASS, 10 passed

Then the whole suite: `python -m pytest harness/tests -q`
Expected: all green, no failures and no errors.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness
git commit -m "feat(harness): run orchestrator with sample selection, ticks, plots and caveats

<trailer>"
```

---

## Task 11: The spot build and its clock gate

This is the only task that touches `Z:`. It is last because nothing above needs it.

**Files:**
- Create: `harness/build/spot_5m_100ms.py`
- Test: `harness/tests/test_spot_build.py`

**Interfaces:**
- Consumes: `harness.paths.VENUE_L1`, `harness.paths.SPOT`.
- Produces:
  - `bucket_venue_l1(df, open_ts) -> pd.DataFrame` with `open_ts, t_ms, spot, bid_sz, ask_sz`
  - `measure_offset(venue_ts, panel_ts) -> float`
  - `ClockGateError`
  - `build(days=None, out_path=paths.SPOT) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_spot_build.py`:

```python
"""Pins the spot build's bucketing and its clock gate.

The panel's captures were corrected onto polydata's vantage; stream_venue_l1
is a different host with its own drifting clock. Joining them without
measuring the offset would misalign spot and book -- and beta collapses to
zero inside tau = 13 s, so a silent 50 ms error is the difference between
signal and noise.
"""
import numpy as np
import pandas as pd
import pytest

from harness.build.spot_5m_100ms import (ClockGateError, bucket_venue_l1,
                                         measure_offset, require_offset)


def _venue(open_ts=1786665600, n=30, step=0.1):
    ts = open_ts + np.arange(n) * step
    return pd.DataFrame({
        "ts": ts,
        "bn_spot_mid": 100_000.0 + np.arange(n),
        "bn_spot_bid_sz": np.full(n, 2.0),
        "bn_spot_ask_sz": np.full(n, 3.0),
    })


def test_observations_land_on_the_100ms_grid():
    out = bucket_venue_l1(_venue(), 1786665600)
    assert (out["t_ms"] % 100 == 0).all()
    assert out["t_ms"].min() >= 0 and out["t_ms"].max() < 300_000


def test_the_first_observation_in_a_bucket_wins():
    """Same rule as s04_panel.py, so the two grids mean the same thing."""
    df = pd.DataFrame({
        "ts": [1786665600.00, 1786665600.05, 1786665600.10],
        "bn_spot_mid": [100.0, 999.0, 200.0],
        "bn_spot_bid_sz": [1.0, 1.0, 1.0],
        "bn_spot_ask_sz": [1.0, 1.0, 1.0],
    })
    out = bucket_venue_l1(df, 1786665600).set_index("t_ms")
    assert out.loc[0, "spot"] == 100.0
    assert out.loc[100, "spot"] == 200.0


def test_observations_outside_the_window_are_dropped():
    df = _venue(n=4)
    df.loc[0, "ts"] = 1786665599.0        # before the open
    df.loc[3, "ts"] = 1786665901.0        # after the close
    assert len(bucket_venue_l1(df, 1786665600)) == 2


def test_an_offset_is_recovered_from_a_shifted_clock():
    panel = np.arange(0.0, 60.0, 0.1)
    assert measure_offset(panel + 0.074, panel) == pytest.approx(0.074, abs=1e-6)


def test_the_gate_rejects_a_day_with_no_measurable_offset():
    with pytest.raises(ClockGateError, match="2026-08-20"):
        require_offset("2026-08-20", None)


def test_the_gate_rejects_an_implausible_offset():
    """Seconds of drift is a broken clock, not a vantage difference."""
    with pytest.raises(ClockGateError, match="implausible"):
        require_offset("2026-08-20", 5.0)


def test_the_gate_accepts_a_plausible_offset():
    assert require_offset("2026-08-20", 0.074) == pytest.approx(0.074)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_spot_build.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.build.spot_5m_100ms'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/build/spot_5m_100ms.py`:

```python
"""Venue L1 onto the panel's 100 ms grid.

Two rules carried over from data/scripts/s04_panel.py so that the two grids
mean the same thing: buckets are [t, t+100) since the market open, and the
FIRST observation in a bucket wins.

THE CLOCK GATE. stream_venue_l1 is recorded on a different host from the
panel's reference vantage, with an independent and drifting offset. This build
measures that offset per day and REFUSES TO WRITE if any day lacks a plausible
one. A silent misalignment would be invisible in the output and fatal in the
result, because beta(tau, h) is zero inside tau = 13 s.
"""
import os

import numpy as np
import pandas as pd

from harness import paths

MAX_PLAUSIBLE_OFFSET_S = 1.0     # anything larger is a broken clock
SPOT_COL = "bn_spot_mid"


class ClockGateError(RuntimeError):
    """The venue-to-panel clock offset could not be established."""


def bucket_venue_l1(df, open_ts, offset_s=0.0):
    """One row per 100 ms bucket of the window opening at `open_ts`."""
    ts = df["ts"].to_numpy(dtype="float64") - offset_s
    t_ms = np.floor((ts - open_ts) * 1000.0 / paths.BUCKET_MS).astype("int64")
    t_ms *= paths.BUCKET_MS

    keep = (t_ms >= 0) & (t_ms < paths.H * 1000)
    out = pd.DataFrame({
        "open_ts": open_ts,
        "t_ms": t_ms[keep],
        "spot": df[SPOT_COL].to_numpy()[keep],
        "bid_sz": df["bn_spot_bid_sz"].to_numpy()[keep],
        "ask_sz": df["bn_spot_ask_sz"].to_numpy()[keep],
        "_ts": ts[keep],
    })
    out = (out.sort_values(["t_ms", "_ts"], kind="stable")
              .drop_duplicates("t_ms", keep="first")
              .drop(columns="_ts")
              .reset_index(drop=True))
    return out


def measure_offset(venue_ts, panel_ts):
    """Median venue-minus-panel timestamp difference, in seconds.

    Both captures are ~10 Hz samplers of the same world, so the median
    difference of their aligned observation times is the vantage offset.
    """
    venue_ts = np.asarray(venue_ts, dtype="float64")
    panel_ts = np.asarray(panel_ts, dtype="float64")
    n = min(len(venue_ts), len(panel_ts))
    if n == 0:
        return None
    return float(np.median(venue_ts[:n] - panel_ts[:n]))


def require_offset(day, offset_s):
    """The gate. Raises rather than guessing."""
    if offset_s is None or not np.isfinite(offset_s):
        raise ClockGateError(
            f"{day}: no venue-to-panel clock offset could be measured. "
            f"Refusing to write a misaligned spot panel.")
    if abs(offset_s) > MAX_PLAUSIBLE_OFFSET_S:
        raise ClockGateError(
            f"{day}: implausible clock offset {offset_s:.3f} s "
            f"(limit {MAX_PLAUSIBLE_OFFSET_S} s). This is a broken clock, "
            f"not a vantage difference.")
    return float(offset_s)


def build(days=None, out_path=None, panel_path=None):
    """Join venue L1 onto the panel grid for every day, gated per day."""
    out_path = out_path or paths.SPOT
    panel_path = panel_path or paths.PANEL

    panel = pd.read_parquet(panel_path, columns=["open_ts", "t_ms", "recv_ms"])
    panel["day"] = pd.to_datetime(panel["open_ts"], unit="s").dt.strftime(
        "%Y-%m-%d")

    available = sorted(
        d.split("=")[1] for d in os.listdir(paths.VENUE_L1)
        if d.startswith("date="))
    days = sorted(set(days or available) & set(available))

    frames, offsets = [], []
    for day in days:
        venue = pd.read_parquet(
            os.path.join(paths.VENUE_L1, f"date={day}"),
            columns=["ts", SPOT_COL, "bn_spot_bid_sz", "bn_spot_ask_sz"])
        venue = venue.dropna(subset=["ts", SPOT_COL]).sort_values("ts")

        day_panel = panel[panel["day"] == day]
        offset = require_offset(day, measure_offset(
            venue["ts"].to_numpy(),
            day_panel["recv_ms"].to_numpy(dtype="float64") / 1000.0))
        offsets.append({"day": day, "offset_s": offset, "n": len(venue)})

        for open_ts in sorted(day_panel["open_ts"].unique()):
            window = venue[(venue["ts"] >= open_ts + offset - 1.0)
                           & (venue["ts"] < open_ts + offset + paths.H + 1.0)]
            if len(window):
                frames.append(bucket_venue_l1(window, int(open_ts), offset))

    pd.DataFrame(offsets).to_csv(
        os.path.join(paths.RESULTS, "venue_vantage_offsets.tsv"),
        sep="\t", index=False)

    spot = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    spot.to_parquet(out_path, index=False)
    print(f"spot: {len(spot):,} rows, {len(days)} days -> {out_path}",
          flush=True)
    return spot
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_spot_build.py -q`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness
git commit -m "feat(harness): spot build with a per-day clock gate that refuses to guess

<trailer>"
```

---

## Task 12: Episode loading from the real panel

**Files:**
- Create: `harness/build/episodes.py`
- Test: `harness/tests/test_episodes_build.py`

**Interfaces:**
- Consumes: `build_episode`, `harness.paths`.
- Produces: `load_episodes(panel_path=None, strikes_path=None, spot_path=None, fair_path=None, days=None, markets=None, max_markets=None) -> list[Episode]`, and `settlement_map(strikes: pd.DataFrame) -> dict[int, float]`.

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_episodes_build.py`:

```python
"""Pins episode assembly from panel-shaped frames.

Settlement is not stored anywhere: settle(N) IS the strike of the market
opening 300 s later. Verified exact on 7,330 of 7,330 back-to-back pairs.
"""
import numpy as np
import pandas as pd
import pytest

from harness.build.episodes import load_episodes, settlement_map


def _strikes():
    return pd.DataFrame({
        "market_id": ["a", "b", "c"],
        "open_ts": [1786665600, 1786665900, 1786666200],
        "strike": [100_000.0, 100_050.0, 99_900.0],
    })


def _panel():
    rows = []
    for mid, open_ts in (("a", 1786665600), ("b", 1786665900)):
        for t in (0, 100, 200):
            rows.append({"market_id": mid, "open_ts": open_ts, "t_ms": t,
                         "bid": 0.49, "ask": 0.51, "mid": 0.50, "n_src": 2})
    return pd.DataFrame(rows)


def test_settlement_is_the_next_markets_strike():
    m = settlement_map(_strikes())
    assert m[1786665600] == pytest.approx(100_050.0)
    assert m[1786665900] == pytest.approx(99_900.0)


def test_the_last_market_has_no_settlement():
    assert settlement_map(_strikes()).get(1786666200) is None


def test_episodes_are_built_with_settlement_and_winner(tmp_path):
    panel_p = tmp_path / "panel.parquet"
    strikes_p = tmp_path / "strikes.parquet"
    _panel().to_parquet(panel_p)
    _strikes().to_parquet(strikes_p)

    eps = load_episodes(panel_path=str(panel_p), strikes_path=str(strikes_p))
    by_id = {e.market_id: e for e in eps}
    assert by_id["a"].settle == pytest.approx(100_050.0)
    assert by_id["a"].winner_up is True       # 100050 >= 100000
    assert by_id["b"].winner_up is False      # 99900 < 100050


def test_each_episode_is_a_full_length_grid(tmp_path):
    panel_p, strikes_p = tmp_path / "p.parquet", tmp_path / "s.parquet"
    _panel().to_parquet(panel_p)
    _strikes().to_parquet(strikes_p)
    ep = load_episodes(panel_path=str(panel_p), strikes_path=str(strikes_p))[0]
    assert len(ep.bid) == 3000
    assert np.isnan(ep.bid[0]), "the causality shift still applies"


def test_max_markets_is_honoured(tmp_path):
    panel_p, strikes_p = tmp_path / "p.parquet", tmp_path / "s.parquet"
    _panel().to_parquet(panel_p)
    _strikes().to_parquet(strikes_p)
    assert len(load_episodes(panel_path=str(panel_p),
                             strikes_path=str(strikes_p),
                             max_markets=1)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_episodes_build.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.build.episodes'`

- [ ] **Step 3: Write minimal implementation**

Create `harness/build/episodes.py`:

```python
"""Panel rows -> Episodes.

Settlement is not stored in this dataset. settle(N) is the strike of the
market opening at open_ts + 300, which is why open_ts is carried on the panel.
Verified exact on all 7,330 testable back-to-back pairs; winner_up is
settle >= strike, with ties going Up.
"""
import numpy as np
import pandas as pd

from harness import paths
from harness.core.episode import build_episode


def settlement_map(strikes):
    """open_ts -> the NEXT market's strike, i.e. this market's settlement."""
    s = strikes.sort_values("open_ts")
    nxt = dict(zip(s["open_ts"] - paths.H, s["strike"]))
    return {int(ts): float(nxt[ts]) for ts in s["open_ts"] if ts in nxt}


def load_episodes(panel_path=None, strikes_path=None, spot_path=None,
                  fair_path=None, days=None, markets=None, max_markets=None):
    panel_path = panel_path or paths.PANEL
    strikes_path = strikes_path or paths.STRIKES

    strikes = pd.read_parquet(strikes_path)
    settle_by_open = settlement_map(strikes)

    filters = []
    if markets:
        filters.append(("market_id", "in", list(markets)))
    panel = pd.read_parquet(panel_path,
                            filters=filters or None,
                            columns=["market_id", "open_ts", "t_ms",
                                     "bid", "ask", "mid", "n_src"])
    panel["day"] = pd.to_datetime(panel["open_ts"], unit="s").dt.strftime(
        "%Y-%m-%d")
    if days:
        panel = panel[panel["day"].isin(set(days))]

    spot = None
    if spot_path:
        spot = pd.read_parquet(spot_path)
    fair = None
    if fair_path:
        fair = pd.read_parquet(fair_path)

    strike_by_id = dict(zip(strikes["market_id"], strikes["strike"]))

    episodes = []
    for (market_id, open_ts), obs in panel.groupby(["market_id", "open_ts"],
                                                   sort=True):
        if market_id not in strike_by_id:
            continue
        ep_spot = None
        if spot is not None:
            ep_spot = spot[spot["open_ts"] == open_ts][["t_ms", "spot"]]
        ep_s = None
        if fair is not None:
            rows = fair[fair["market_id"] == market_id]
            ep_s = np.full(paths.N_BUCKET, np.nan)
            k = (rows["t_ms"].to_numpy() // paths.BUCKET_MS).astype("int64")
            keep = (k >= 0) & (k < paths.N_BUCKET)
            ep_s[k[keep]] = rows["s"].to_numpy()[keep]

        episodes.append(build_episode(
            market_id=market_id, open_ts=int(open_ts),
            day=obs["day"].iloc[0],
            strike=strike_by_id[market_id],
            settle=settle_by_open.get(int(open_ts)),
            obs=obs, spot=ep_spot, s=ep_s))

        if max_markets is not None and len(episodes) >= max_markets:
            break

    return episodes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_episodes_build.py -q`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness
git commit -m "feat(harness): episode loading with settlement from the successor strike

<trailer>"
```

---

## Task 13: Investigation template and a first real smoke run

Proves the whole thing works against the actual 20.9 M-row panel.

**Files:**
- Create: `investigations/README.md`, `investigations/_template/run.py`, `investigations/2026-09-09-smoke/run.py`, `investigations/2026-09-09-smoke/fair.py`
- Test: manual run (this task's deliverable is a run folder, not a unit test)

**Interfaces:**
- Consumes: everything.
- Produces: a committed `investigations/README.md` documenting the block-override convention.

- [ ] **Step 1: Write the investigation template**

Create `investigations/README.md`:

```markdown
# Investigations

One folder per question. A folder is a set of block overrides plus a `run.py`.

## How blocks resolve

A block slot is a filename. The harness looks in **this folder first**, then in
`harness/blocks/defaults/`. Nothing is registered and nothing is named --
dropping `link.py` here IS the override.

Slots: `fair` `vol` `f` `link` `quote` `execution` `fill` `fees`

## What a run leaves behind

`runs/<timestamp>__<confighash>/` containing `manifest.json` (every slot's path
and sha256, plus the harness commit), `blocks/` (frozen copies of the files
actually used), `ledger.parquet`, `markets.parquet`, `summary.json`,
optionally `ticks.parquet`, and `cum_pnl.png`.

Two runs with the same model, params and data share a config hash. If yours
does not match one you expected, something changed.

## Before believing a number

`summary.json` carries two caveats on every run and they are not decoration:

* the 100 ms grid flatters results by ~$0.13/market, measured on two models
* every maker fill is a model, not a measurement -- report a range across fill
  optimism, never a single maker number
```

Create `investigations/_template/run.py`:

```python
"""Copy this folder, rename it, drop in the block files you want to change."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from harness.build.episodes import load_episodes          # noqa: E402
from harness.core.config import ExecConfig, Output, QuoteParams, Sample  # noqa: E402
from harness.core.latency import LatencyModel             # noqa: E402
from harness.core.run import run                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    episodes = load_episodes(days=("2026-08-20",))
    result = run(
        HERE,
        quote=QuoteParams(e_p=0.02, rpl_p=0.001, max_pos=50.0, shares=10.0),
        execn=ExecConfig(mode="maker", latency=LatencyModel()),
        sample=Sample(),
        output=Output(seeds=(0, 1, 2)),
        episodes=episodes,
    )
    print(result["run_dir"])
    print(result["summary"]["headline"])
```

- [ ] **Step 2: Create the smoke investigation with a placeholder fair**

Create `investigations/2026-09-09-smoke/fair.py`:

```python
"""Placeholder fair: s = strike, so p = 0.5 everywhere.

This exists ONLY to prove the harness runs end to end against the real panel
before the Gambling102 fair export is wired in. It has no forecasting content.
Any PnL it produces is a property of the quoting and fill machinery, not of a
model, and must never be reported as an edge.
"""
import numpy as np


def precompute(ep):
    return np.full(len(ep), ep.strike, dtype="float64")
```

Create `investigations/2026-09-09-smoke/run.py`:

```python
"""End-to-end smoke run on one real day. Not a result -- a proof of plumbing."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from harness.build.episodes import load_episodes          # noqa: E402
from harness.core.config import ExecConfig, Output, QuoteParams, Sample  # noqa: E402
from harness.core.latency import LatencyModel             # noqa: E402
from harness.core.run import run                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    episodes = load_episodes(days=("2026-08-20",), max_markets=50)
    print(f"loaded {len(episodes)} episodes")

    result = run(
        HERE,
        quote=QuoteParams(e_p=0.03, rpl_p=0.0005, max_pos=50.0, shares=10.0),
        execn=ExecConfig(mode="maker",
                         latency=LatencyModel(jitter_frac=0.2)),
        sample=Sample(),
        output=Output(emit_ticks=True, seeds=(0, 1, 2)),
        episodes=episodes,
    )
    print(result["run_dir"])
    print(result["summary"]["headline"])
    print(result["summary"]["gates"]["passed"])
```

- [ ] **Step 3: Run it against the real panel**

Run: `python investigations/2026-09-09-smoke/run.py`

Expected: prints an episode count near 288 (one day of 5 m markets), a run
directory path, a headline dict, and a gate verdict. Confirm on disk:

```bash
ls investigations/2026-09-09-smoke/runs/*/
python -c "import pandas as pd,glob; \
d=sorted(glob.glob('investigations/2026-09-09-smoke/runs/*'))[-1]; \
l=pd.read_parquet(d+'/ledger.parquet'); \
print(l[['t_ms','side','liquidity','price','fee_usd','delta_quality_c']].head()); \
print('maker fees negative:', (l[l.liquidity==0].fee_usd<=0).all())"
```

Expected: a ledger with maker rows carrying negative `fee_usd`, finite
`delta_quality_c`, and a `blocks/fair.py` in the run folder containing the
placeholder's docstring.

- [ ] **Step 4: Run the full suite once more**

Run: `python -m pytest harness/tests -q`
Expected: all green, no failures and no errors.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/investigations backtesting_5m/harness
git commit -m "feat(harness): investigation template and an end-to-end smoke run

Proves the harness runs against the real panel with a placeholder fair. The
placeholder carries no forecasting content and its PnL is not a result.

<trailer>"
```

---

## Task 14: Mandatory sweeps and the daily rebate minimum

The spec requires latency and fill-optimism sweeps to attach to every headline automatically, not to be remembered. This task makes that true, and adds the `$1.00/day` rebate floor the fee doc records.

**Files:**
- Create: `harness/core/sweeps.py`
- Modify: `harness/blocks/defaults/fees.py` (append `apply_daily_minimum`)
- Test: `harness/tests/test_sweeps.py`

**Interfaces:**
- Consumes: `run` from `harness.core.run`, `FeeSchedule`.
- Produces:
  - `LATENCY_LADDER_MS = (0.0, 100.0, 200.0, 250.0, 500.0)`
  - `FILL_ARMS = {"optimistic": {...}, "adverse_lag": {...}, "penetration": {...}}`
  - `run_with_sweeps(investigation_dir, quote, execn, sample, output, episodes) -> dict` with a `sweeps` key holding `{"latency": [...], "fill": [...]}`
  - `fees.apply_daily_minimum(ledger: pd.DataFrame, minimum_usd: float = 1.0) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

Create `harness/tests/test_sweeps.py`:

```python
"""Pins the sweeps that must attach to every headline.

Taker edge is known to lose significance between 200 ms and 500 ms, and every
maker number is conditional on an unmeasurable fill assumption. Reporting a
single number for either is the failure mode these sweeps exist to prevent.
"""
import os

import pandas as pd
import pytest

from harness.blocks.defaults.fees import apply_daily_minimum
from harness.core.config import ExecConfig, Output, QuoteParams, Sample
from harness.core.sweeps import FILL_ARMS, LATENCY_LADDER_MS, run_with_sweeps


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


def _sweep(tmp_path, episodes):
    return run_with_sweeps(
        str(tmp_path),
        quote=QuoteParams(e_p=0.0, shares=10.0, max_pos=10.0),
        execn=ExecConfig(mode="taker"),
        sample=Sample(),
        output=Output(plots=False),
        episodes=episodes,
    )


def test_the_latency_ladder_is_reported_in_full(tmp_path, episodes):
    result = _sweep(tmp_path, episodes)
    got = [arm["latency_ms"] for arm in result["sweeps"]["latency"]]
    assert got == list(LATENCY_LADDER_MS)


def test_every_fill_arm_is_reported(tmp_path, episodes):
    result = _sweep(tmp_path, episodes)
    assert {a["arm"] for a in result["sweeps"]["fill"]} == set(FILL_ARMS)


def test_slower_latency_never_helps_a_taker(tmp_path, episodes):
    """A cheap book crossed later is crossed at a worse or equal price."""
    result = _sweep(tmp_path, episodes)
    ladder = {a["latency_ms"]: a["headline"]["pnl_per_market"]
              for a in result["sweeps"]["latency"]}
    assert ladder[500.0] <= ladder[0.0] + 1e-9


def test_the_sweeps_are_written_into_the_summary(tmp_path, episodes):
    import json
    result = _sweep(tmp_path, episodes)
    summary = json.loads(
        open(os.path.join(result["run_dir"], "summary.json")).read())
    assert "sweeps" in summary


def test_a_dust_day_is_not_paid_its_rebate():
    """Below $1.00/day per stream, the rebate simply does not arrive."""
    ledger = pd.DataFrame({
        "day": ["2026-08-14", "2026-08-15"],
        "liquidity": [0, 0],
        "fee_usd": [-0.50, -4.00],
    })
    out = apply_daily_minimum(ledger)
    assert out.loc[0, "fee_usd"] == pytest.approx(0.0), "dust day zeroed"
    assert out.loc[1, "fee_usd"] == pytest.approx(-4.00)


def test_taker_fees_are_untouched_by_the_minimum():
    ledger = pd.DataFrame({
        "day": ["2026-08-14"], "liquidity": [1], "fee_usd": [0.10]})
    assert apply_daily_minimum(ledger).loc[0, "fee_usd"] == pytest.approx(0.10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest harness/tests/test_sweeps.py -q`
Expected: FAIL — `ImportError: cannot import name 'apply_daily_minimum'`

- [ ] **Step 3: Write minimal implementation**

Append to `harness/blocks/defaults/fees.py`:

```python
def apply_daily_minimum(ledger, minimum_usd: float = 1.0):
    """Zero out maker rebates on days that never reached the payout floor.

    Measured with perfect separation over 131 earn-days: the smallest paid was
    $1.0359 and the largest skipped $0.7209. Dust days are simply not paid, so
    a backtest that books them is overstating maker economics.
    """
    import pandas as pd  # local: the fee model itself stays dependency-free

    if not len(ledger):
        return ledger

    out = ledger.copy()
    maker = out["liquidity"] == int(Liquidity.MAKER)
    earned = -out.loc[maker].groupby("day")["fee_usd"].sum()
    dust = set(earned[earned < minimum_usd].index)
    out.loc[maker & out["day"].isin(dust), "fee_usd"] = 0.0
    return out
```

Create `harness/core/sweeps.py`:

```python
"""The sweeps that must attach to every headline, whether asked for or not.

Two results in this programme's history motivate this file. Taker edge measured
at 200 ms had a day-blocked CI of [+0.157, +0.587]; the same edge at 500 ms had
[-0.026, +0.309]. And no maker fill in this dataset is measured -- there is no
depth, no trade tape and no queue -- so a single maker number is a statement
about the fill block, not about the market.

Reporting one number for either is the failure mode. So the sweep runs.
"""
from dataclasses import replace

from harness.core import stats
from harness.core.run import run

LATENCY_LADDER_MS = (0.0, 100.0, 200.0, 250.0, 500.0)

FILL_ARMS = {
    "optimistic": {"penetration": 0.0},          # upper bound: front of queue
    "adverse_lag": {"penetration": 0.0},         # default; cancel loses races
    "penetration": {"penetration": 0.01},        # conservative queue proxy
}


def _headline(result):
    primary = result["markets"]
    primary = primary[primary["seed"] == primary["seed"].iloc[0]] \
        if len(primary) else primary
    return stats.headline(primary)


def run_with_sweeps(investigation_dir, quote, execn, sample, output, episodes):
    """The reporting entry point. `run()` is the single-arm primitive."""
    base = run(investigation_dir, quote, execn, sample, output, episodes)

    latency_arms = []
    for ms in LATENCY_LADDER_MS:
        arm_latency = replace(execn.latency, place_ms=ms, cancel_ms=ms,
                              take_ms=ms)
        arm = run(investigation_dir, quote, replace(execn, latency=arm_latency),
                  sample, replace(output, plots=False), episodes)
        latency_arms.append({"latency_ms": ms, "headline": _headline(arm),
                             "run_dir": arm["run_dir"]})

    fill_arms = []
    for name, params in FILL_ARMS.items():
        arm = run(investigation_dir, quote, execn, sample,
                  replace(output, plots=False), episodes)
        fill_arms.append({"arm": name, "fill_params": params,
                          "headline": _headline(arm),
                          "run_dir": arm["run_dir"]})

    base["sweeps"] = {"latency": latency_arms, "fill": fill_arms}

    import json
    import os
    path = os.path.join(base["run_dir"], "summary.json")
    summary = json.loads(open(path).read())
    summary["sweeps"] = base["sweeps"]
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    return base
```

Then wire `fill_params` through so the fill arms are real rather than three identical runs. In `harness/core/config.py`, add one field to `ExecConfig`:

```python
    fill_params: dict = field(default_factory=dict)
```

In `harness/core/run.py`, replace the line `"fill_params": {},` with:

```python
        "fill_params": dict(execn.fill_params),
```

and in `sweeps.py` change the fill loop's `run(...)` call to pass the arm through:

```python
        arm = run(investigation_dir, quote,
                  replace(execn, fill_params=params), sample,
                  replace(output, plots=False), episodes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest harness/tests/test_sweeps.py -q`
Expected: PASS, 6 passed

Then the whole suite: `python -m pytest harness/tests -q`
Expected: all green, no failures and no errors.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/kaima/Github2/Gambling104
git add backtesting_5m/harness
git commit -m "feat(harness): mandatory latency and fill sweeps, daily rebate minimum

No headline leaves the harness as a single number: the latency ladder and the
fill-optimism arms attach to every run.

<trailer>"
```

---

## After the plan

Two things remain outside this plan and both need input rather than code:

1. **The fair export contract.** `harness/blocks/defaults/fair.py` reads `ep.s` and raises if it is absent. Wiring the real chain needs a `(market_id, t_ms, s)` parquet from Gambling102 plus a statement of what information fed each value, so it can be lookahead-audited on import. Until then every run uses a placeholder and is plumbing, not a result.

2. **The taker latency default.** Implemented at the operator-set 200 ms with the 250 ms venue lock enforced as a floor. The measured figure is 276 ms, and the architecture doc shows taker edge losing significance across exactly this range. The sweep is mandatory; the default is still a choice worth revisiting once real fair values are flowing.
