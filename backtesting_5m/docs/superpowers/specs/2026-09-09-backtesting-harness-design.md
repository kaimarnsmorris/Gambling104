---
title: A block-composed backtesting harness for the BTC up/down 5 m book
date: 2026-09-09
status: implemented; reconciled against the code 2026-09-09
superseded_in_part: 2026-09-09-stream-registry-and-module-design.md
related:
  - ../../../data/README.md
  - Gambling102 docs/strategy/2026-09-05-final-model-architecture.md
  - Gambling102 docs/polymarket/fees-and-taker.md
---

# Backtesting harness — design

> **Superseded in part by `2026-09-09-stream-registry-and-module-design.md`.**
> That spec replaces four things described below: tick data is referenced
> through the **stream registry** and a `streams=` run parameter rather than by
> hard-coded paths and hand-added `Episode` fields; an investigation names a
> shared **model directory** instead of copying block files into itself (run
> folders still freeze their own copies, so provenance is unchanged); research
> lives at the repository root under `../investigations` and `../models`, not
> inside `backtesting_5m/`; and **plotting has left the harness** for the
> separate, optional `harness.report` module, which reads run folders. The rest
> of this document still describes the code.

## The question this answers

**What quoting policy, priced off an externally supplied fair value, survives
this venue's fees, latency and fill mechanics on the 5 m BTC up/down book —
and how much of the answer is real?**

The harness owns execution, fills, fees, inventory, measurement and provenance.
It does **not** own the fair value: `s` arrives as data.

## Why rebuild rather than extend the 2026-08-23 lab

The prior 5 m lab ran on a book panel whose tradable coverage was ~4 days
(1,299 markets, and 25 markets across 08-14 to 08-16). This repo's build
produces a reconciled, multi-capture, gate-verified 100 ms panel: **7,111
markets over 2026-08-14 to 09-08**, of which ~6,200 fall inside the venue-L1
spot window. That is roughly 5x the tradable sample, with cross-capture
corroboration on 13.5 M of 20.9 M buckets. The prior lab is worth raiding for
measured latency constants; its structure predates the block algebra and its
fill model is the one Gambling102's own roadmap lists as *unbuilt*.

`docs/strategy/2026-09-05-final-model-architecture.md` names the maker fill
model as P1 and the binding uncertainty in the whole programme. This harness is
that item.

## Scope

**In:** data layer and spot join, block resolution, quote construction,
execution policy, fill models, fee accounting, inventory, ledger, per-tick
diagnostics, measurement gates, run provenance.

**Out:** the fair-value chain (`s` is imported), live trading, order routing,
any change to `data/scripts/s00..s04`.

**Non-goals:** an experiment framework, a result cache, a job scheduler. Sweeps
are loops in `investigations/`; they become library code only when a specific
investigation proves they need to be.

---

## 1. Data layer

### 1.1 Sources

| source | path | role |
|---|---|---|
| book panel | `data/book_5m_100ms.parquet` | UP token L1, 100 ms grid |
| strikes | `data/strikes_5m.parquet` | `market_id, open_ts, strike` |
| venue L1 | `Z:\parquet\stream_venue_l1` | BTC spot/perp L1 **with sizes**, 5 venues |
| fair export | `data/fair/<model_id>.parquet` | `s` per `(market_id, t_ms)`, from Gambling102 |

Venue L1 carries `bn_spot_{bid,ask,mid,bid_sz,ask_sz,stale}`, the same for
`bn_perp`, plus `cb`, `okx`, `bybit`, and `usdt_basis` / `perp_basis`. Sizes are
retained deliberately: `I = ln(bid_sz/ask_sz)` is priced at 0.93 c/share at
tau = 280 s in `l1_queue_decay/`, and cannot be computed without them.

### 1.2 The spot build — `harness/build/spot_5m_100ms.py`

Resamples venue L1 onto the panel's `(open_ts, t_ms)` grid, same 100 ms
buckets, same first-observation-in-bucket rule as `s04_panel.py`.

**The clock offset is a declared assumption, not a measurement.** The panel's
captures were corrected onto polydata's vantage; `stream_venue_l1` is a
different host with an independent, drifting clock. That offset **cannot be
measured from these two streams**: the panel records receipt of Polymarket book
updates and the venue feed records receipt of Binance/Coinbase/OKX/Bybit
updates on a different host, so they share no event to align on, and
cross-correlating them would confound the offset with the spot-to-book response
lag this harness exists to study. The function that tried to measure it is
gone.

So `build(..., offset_s=)` takes a **configured** offset, defaulting to 0.0 s.
`require_offset` gates it per day — it raises on an absent or non-finite
offset, and on one above `MAX_PLAUSIBLE_OFFSET_S = 1.0` s, which is a broken
clock rather than a vantage difference — and the offset actually used is
written per day to `results/venue_vantage_offsets<suffix>.tsv`. This repo's own
same-book inter-host drift measurements bound the plausible range at
**0-74 ms**, and every run stamps the admission into `summary["caveats"]`:
see `CLOCK_OFFSET_NOTE` and `_clock_offset_caveat` in `harness/core/run.py`,
which reads that TSV and says outright when no offset is recorded for the days
the run covered. Rationale for gating it at all: beta(tau,h) collapses to zero
inside tau = 13 s, so a silent 50 ms misalignment is not cosmetic — it is the
difference between signal and noise at exactly the horizons under test. Every
spot-derived number is conditional on the assumption and must be swept over
that band the way latency and fill optimism are.

**Coverage is asymmetric and must be reported.** Venue L1 begins 2026-08-17;
the book panel begins 2026-08-14. Markets before the spot window are retained
in the panel but carry `has_spot = False` and are excluded from any run whose
blocks read spot. The summary reports both counts so no equity curve silently
starts flat.

### 1.3 The Episode

One immutable object per market, arrays of length `N = 3000`:

| field | meaning |
|---|---|
| `t_ms` | 0 … 299900 |
| `has_book`, `book_age_ms` | observation present; ms since the last one |
| `bid, ask, mid` | UP token L1, probability |
| `n_src` | capture corroboration, from the panel |
| `spot`, `spot_age_ms`, `has_spot` | venue L1 on the grid, **USD** (the USDT basis already subtracted) |
| `spot_usdt`, `chainlink`, `chainlink_age_ms` | the raw USDT mid and the settlement oracle, same grid |
| `s`, `sigma` | fair value (USD) and vol, filled by the `fair` / `vol` blocks |
| `strike`, `settle`, `winner_up` | settlement truth |
| `day` | for blocked splits and CIs |
| `streams` | registered streams, `{name: {col, age_ms, has}}` — see the registry spec |

There is also an optional **pre-open warm-up** region (`warmup_*`,
`has_warmup`) on the same grid and under the same causality shift, for
`precompute` only: `len(ep)` still reports 3,000 so a loop written against
`range(len(ep))` cannot reach into it.

**Nothing is forward-filled.** A bucket no capture observed has no book; the age
counters grow across it. The engine refuses to fill against a book older than
`max_book_age_ms`. This makes the panel README's rule — *a stale book is worse
than a missing one* — an enforced property rather than a convention.

### 1.4 The causality shift — the load-bearing decision

Episode arrays are indexed so that **index `i` carries exactly the information
available for a decision at `t_ms = i*100`**. Per the panel README, a row
labelled `t_ms` holds an observation drawn from `[t_ms, t_ms+100)` and was not
knowable at `t_ms`; the loader therefore places it at index `i+1`.

This is done once, in one function, with one test. Afterwards **no block,
policy or strategy can reach a future tick, because no index contains one.**
Lookahead stops being a discipline and becomes unrepresentable. This is the
single reason the design pays for an event loop instead of vectorising
end-to-end.

---

## 2. Blocks are files

### 2.1 Slots and resolution

A block *slot* is a filename. Resolution for each slot, in order
(`provenance.resolve_slots`):

1. the investigation folder
2. the **model directory**, when the run named one (`backtest(model=...)`)
3. `harness/blocks/defaults/`

Nothing is registered and nothing is named. Writing `link.py` in your
investigation folder *is* the override, and it shadows the model directory the
same way.

| slot | file | contract |
|---|---|---|
| fair | `fair.py` | `precompute(ep) -> s[N]` (USD) |
| vol | `vol.py` | `precompute(ep) -> sigma[N]` |
| f | `f.py` | `standardise(level, strike, sigma) -> z`, called per tick with the episode's `strike` and `sigma[i]` |
| link | `link.py` | `link(z) -> p` in (0,1) |
| quote | `quote.py` | `quotes(s_i, q, strike, sigma_i, params, standardise, link) -> (eff_bid, eff_ask)`, theoretical and *not* snapped to the cent grid — see §3.1 |
| execution | `execution.py` | `decide(i, eff_bid, eff_ask, q, ep, live_orders, execn, params) -> (to_place, to_cancel)` — post / cross / cancel / gate |
| fill | `fill.py` | `resolve(orders, ep, i, params) -> [Fill]` — whether a live order trades, and at what price |
| fees | `fees.py` | a `FeeSchedule` whose `charge(liquidity, shares, p) -> usd` is negative for rebates |

`fair` and `vol` are **SignalBlocks**: vectorised, stateless, precomputed per
episode. `quote`, `execution`, `fill` are per-tick and inventory-dependent.
That split keeps array speed on the expensive part and causal exactness on the
path-dependent part.

`execution` and `fill` are separate slots even though execution changes rarely.
The reason is P1: the maker fill assumption is unmeasurable from this data, so
every maker result must be reported as a *range* over fill optimism with the
policy held fixed. Two slots make that a one-line swap.

### 2.2 Quote construction

```
z_bid   = f(s[i] - e_s - q*rpl_s)
z_ask   = f(s[i] + e_s - q*rpl_s)
eff_bid = link(z_bid - e_z - q*rpl_z) - e_p - q*rpl_p
eff_ask = link(z_ask + e_z - q*rpl_z) + e_p - q*rpl_p
```

`e_*` are **symmetric half-spread edges** — they widen the pair.
`rpl_*` are **retreat per lot** — they push both quotes the same direction as
inventory grows, they do not widen. `q` is clamped to `max_pos`.

| symbol | units | layer |
|---|---|---|
| `s`, `e_s`, `rpl_s` | USD of BTC | settlement level |
| `z`, `e_z`, `rpl_z` | sigma (moneyness) | latent |
| `p`, `e_p`, `rpl_p` | probability | price |

Three layers rather than one because the pass-through is not constant: 0.79 c/$
at the money at tau = 300 s, rising to 14.5 c/$ at tau = 30 s
(`book_passthrough/`). A single-layer edge parameter is mis-specified across the
window by construction.

Mapping to the shipped chain `E[A] -> z -> p = NIG_sf(z; zeta)`: `s` is `E[A]`,
`f` is the moneyness standardisation, `link` is the NIG survival function.

---

## 3. Engine and execution

### 3.1 Loop

```python
for i in range(len(ep)):                      # harness/core/loop.py
    live = [o for o in live if not o.is_dead(i)]
    eff_bid, eff_ask = quotes(s[i], q, ep.strike, sigma[i], params,
                              standardise, link)
    for fl in resolve(live, ep, i, fill_params):        # orders already live
        q, cash = apply(fl, fee_schedule, ledger)
    to_place, to_cancel = decide(i, eff_bid, eff_ask, q, ep, live,
                                 execn, params)
    # the engine stamps timing: live_from / cancel_at / expires_at
    live = stamp(live, to_place, to_cancel, latency, rng)
settle(q, ep.winner_up)
```

Illustrative, but every signature above is the real one. Two things the shape
carries: fills against already-live orders are resolved **before** policy runs
at the same index, and **latency lives in the loop, not in the policy block** —
`decide` returns intentions with no timing, and the engine turns them into
`live_from` / `cancel_at` / `expires_at` indices. That is why fill optimism and
latency can be swept with policy held fixed. Cancels are also issued when the
model is *not* quoting (`s` or `sigma` NaN), and only ever to orders a cancel
can still reach (`types.cancellable_ids`).

`execution` owns policy: order lifetime, requote cadence, cancel rules,
`max_book_age_ms` gating, position caps (**enforced on the taker path too**
— the 2026-05-20 taker-cap-bypass lesson, and enforced per side on in-flight
size, not on signed net inventory). `fill` owns whether a live order trades and
at what price.

**Making and taking are not a mode.** There is one policy: it rests on both
sides *and* crosses the book in the same pass, and there is no `mode` knob to
pick between them. Fees enter the thresholds themselves, not the PnL
afterwards — the schedule is dollars per share and the price is a probability,
so they are directly comparable:

```
resting_bid = eff_bid - maker_fee(resting_bid)     # maker_fee < 0: posts HIGHER
resting_ask = eff_ask + maker_fee(resting_ask)     # posts LOWER

buy  (taker) when  book_ask <= eff_bid - taker_fee(book_ask)
sell (taker) when  book_bid >= eff_ask + taker_fee(book_bid)
```

The maker rules are *fixed points* — the fee depends on the price being solved
for — and are solved by two passes of the obvious iteration, which contracts by
a factor 0.014 per pass and so lands four orders of magnitude inside the 0.01
tick. The taker rules are not: the price paid is the price on the screen, so
the fee is evaluated at the book directly. That asymmetry is deliberate.

The rebate justifies posting *inside* fair by up to 0.35 c/share (the all-in
cost of a maker buy at `P` is `P + fee(P)`, and it is that which must clear
`eff_bid`).

**Snapping to the 1 c grid happens exactly once, in `execution.py`, and only on
prices actually sent to the venue.** `quote.py` returns the *theoretical*
`eff_bid`/`eff_ask` off the grid, clipped to `[0, 1]` and nothing more. Order
placement adjusts for the maker fee first and snaps afterwards — conservatively
as before, bids down and asks up. Snapping in both places was a silent no-op by
construction: the whole rebate is sub-tick, so adding it to an already-on-grid
number and re-rounding in the same direction can never move the price. With one
snap it does — a full tick whenever the theoretical quote sits within the rebate
of the next tick, which is `rebate(p)/tick` of the time: **35 % at the money,
33 % over 0.30–0.70, 24 % across the whole 0.01–0.99 range**. On a 1 c venue
that is the difference between resting at the touch and resting behind it.

The conservative guarantee is unchanged and is what the tests now pin: the
all-in cost of the placed price never sits *through* fair. The taker thresholds
are deliberately left unsnapped on both sides — `book_bid`/`book_ask` are
already on the grid because the venue put them there, and `eff_bid`/`eff_ask`
enter as valuations rather than prices; only the protective limit sent with a
cross is snapped. The taker threshold is also where fee-awareness bites hardest:
crossing on a 1 c edge against a ~1.6 c fee loses by construction.

Because `eff_bid`/`eff_ask` are now theory rather than order prices, that is
what `ledger.parquet` and `ticks.parquet` record under those names; the order
price is the fill `price` on a maker fill, and `price - eff_bid` is exactly the
rebate the single snap let through.

### 3.2 Latency

Parametric and seeded, one `LatencyModel` dataclass. Defaults:

| field | default | source |
|---|---|---|
| `place_ms` / `cancel_ms` (maker) | **100 ms** | operator-set |
| `take_ms` | **200 ms** | operator-set |
| `taker_lock_ms` | **250 ms** | venue mechanic — see (a) |
| `move_cancel_ms` | `None` (falls back to `cancel_ms`) | see below |
| `jitter_frac` | `0.0` — off unless set, then a lognormal multiplier | |

Two caveats carried in code, not prose:

**(a) The 250 ms taker lock is a venue mechanic, not an assumption.** The venue
holds marketable crypto up/down orders ~250 ms, non-cancellable.
`LatencyModel.draw` enforces `taker_lock_ms` as a *floor* on the take path,
independent of the latency parameter, and `Order.is_cancellable` refuses to
retract a cross inside that window. It also
reconciles the measured 276 ms taker fill as 26 ms POST flight + 250 ms lock.

**(b) Taker edge is not robust across this range.**
`final-model-architecture.md` §2.1 shows it losing significance between 200 ms
and 500 ms (CI [+0.157, +0.587] falling to [-0.026, +0.309]). The 200 ms default
is therefore reported *always* alongside its sweep, never alone.

The measured constants are configured on that same dataclass rather than being a
separate model: POST flight 26 ms floor / 97 ms p5; cancel round trip 28 ms
quiet, **73-165 ms in a move** — the latter is `move_cancel_ms`, drawn whenever
the book updated at the decision index. The move-conditional cancel is the
adverse-selection mechanism itself (§3.3) and belongs in any serious maker run.

Seeding is **per-episode**: `hash(market_id, run_seed)`. Results are
byte-identical regardless of worker count or completion order, so reproducibility
and parallelism do not trade off.

### 3.3 Fill models

Default `fill.py` is **touch-fill**: a resting order fills **at its own price**
the moment the book's L1 reaches it. A limit order trades at its limit.

**Adverse selection is not modelled by degrading the fill price. It is modelled
by the cancel losing the race.** `cancel_at` is an index: policy decides to pull
a quote at index `i`, the cancel lands at `i + delay_idx(cancel_latency)`, and
any cross before then fills us anyway (`Order.is_live`). Because cancel latency
rises inside a move — 28 ms quiet against **73-165 ms in a move**, carried as
`LatencyModel.move_cancel_ms` and drawn with `in_move=True` when the book
updated at this index — we are filled hardest exactly when we are most wrong.
That mechanism *is* the adverse selection, which is why the move-conditional
cancel belongs in any serious maker run.

`penetration` (an entry in `execn.fill_params`, default 0.0) is the
conservative arm: require the book to trade *through* our price by that margin,
a cheap proxy for the queue position this data cannot observe.

Marketable orders take a separate branch: they pay the book — `ask` on a buy,
`bid` on a sell — but never through their own protective limit, and they live
for exactly the tick they arrive on (`expires_at = live_from + 1`; the venue
held the order through its 250 ms lock, so a cross that is no longer marketable
on arrival is over, not resting).

The three shipped arms (`harness/core/sweeps.py`, `FILL_ARMS`) do **not** differ
only in `penetration`:

| arm | `penetration` | cancel latency |
|---|---|---|
| `optimistic` | 0.0 | **zeroed** — `cancel_ms=0`, `move_cancel_ms=None` |
| `adverse_lag` | 0.0 | exactly as the caller configured it |
| `penetration` | 0.01 | exactly as the caller configured it |

An optimistic arm that keeps a realistic cancel is not an upper bound: with the
fill price fixed at the limit, it would be byte-identical to the default arm and
the sweep would silently run the same configuration twice. Every maker headline
reports the range across all three.

None of this is measurable from the panel — no depth, no trade tape, no queue.
Report a range, never one number.

---

## 4. Fees — `fees.py`

Verified on-chain, 9,073 fills, 0.0 % median error, constant since 2026-05-20 or
earlier.

```python
BASE_FEE_RATE      = 0.07
MAKER_REBATE_PHI   = 0.20     # measured 0.199-0.209, both wallets, all periods
TAKER_REBATE_RHO   = 0.0833   # Silver tier

@dataclass(frozen=True)
class FeeSchedule:                     # fields default to the three constants
    def charge(self, liquidity, shares, p) -> float:
        """USD owed on a fill. Negative means we are paid."""
        base = self.base_fee_rate * p * (1.0 - p) * shares
        if liquidity == Liquidity.TAKER:
            return  base * (1.0 - self.taker_rebate_rho)
        return -base * self.maker_rebate_phi
```

The `fees` slot supplies the **class**; the schedule a run actually prices and
charges against is one instance, resolved once by `run()` — `ExecConfig.fees`
wins if set, otherwise the resolved block's `FeeSchedule()` — and stamped back
onto the config so policy and ledger cannot use different schedules.

Taker fee peaks at **1.75 c/share at p = 0.50** and vanishes at the tails —
which is *why* taking is a tail-only exception: near mid it needs more than
1.75 c of edge before adverse selection. Maker is a **negative fee** averaging
0.234 c/share.

The 1.263 c/share figure in the strategy docs is the realised average over the
fill distribution, **not** a constant. Do not hard-code it.

**Off by default: the $1.00/day per-stream minimum rebate payout** — dust days
are simply not paid, so a backtest that books them overstates maker economics.
`ExecConfig.apply_daily_minimum` (default `False`) turns it on; `run()` then
passes the fill ledger through the resolved fees block's
`apply_daily_minimum(ledger, minimum_usd=1.0)`, which zeroes `fee_usd` on the
maker rows of any `day` whose total maker rebate fell short of the floor. The
choice is stamped either way into
`summary["caveats"]["daily_rebate_minimum_applied"]`. Separation was perfect
over 131 earn-days: smallest paid $1.0359, largest skipped $0.7209.

**It rewrites `ledger.parquet` only.** `markets.parquet`, the headline and the
gates are computed per episode *before* the adjustment, so with the flag on the
withheld dust shows up in the ledger and not in the PnL rollup. Read the two
together, or re-aggregate from the ledger.

---

## 5. Outputs

### 5.1 Fill ledger — `ledger.parquet`, one row per fill

`market_id, open_ts, day, t_ms, side, liquidity, shares, price, fee_usd,
s, sigma, z, fair_p, eff_bid, eff_ask, book_bid, book_ask, mid_at_fill,
q_before, q_after, latency_ms, order_age_ms, mid_t10, delta_quality_c,
markout_settled, seed, order_id, reason`

(the order `Ledger.record_fill` writes them in)

- `liquidity` — maker or taker, on every row.
- `eff_bid` / `eff_ask` — the **theoretical** quote, off the cent grid. The
  order price is `price` on a maker fill (a limit order trades at its limit).
- `fee_usd` — **negative for maker** (rebate). Gross and net are always both
  recoverable; a harness that reports only one will eventually report the wrong
  one, and the taker case (gross +1.085 against the fee) is exactly that trap.
- `fair_p` — `link(z)` at the fill, recorded rather than re-derived. It is
  what lets the report layer compute calibration under **whatever `link` block
  the run actually used**; re-deriving `p` from `z` with a default logistic
  would be silently wrong for any custom link. The `calibration` figure in
  `harness.report.figures` reads it straight off the row.
- `day` — the calendar day, carried so day-blocked statistics and the daily
  rebate minimum can group without a re-join.
- `mid_t10` — the book mid 10 s after the fill, or the settlement outcome
  (0 or 1) where that lands past the end of the window.
- `delta_quality_c` — signed markout in c/share: `100 * (mid_t10 - price)` for a
  buy, `100 * (price - mid_t10)` for a sell (prices are probabilities, so the
  factor of 100 puts it in cents). Positive is good. Where `t_ms + 10 s >= 300 s`
  the reference is the settlement outcome (0 or 1) rather than a mid, and the
  row is flagged `markout_settled = True`.

### 5.2 Per-market rollup — `markets.parquet`

One row per (market, seed): `market_id, open_ts, day, pnl_gross, pnl_net,
fees, shares, n_fills, max_abs_q, settled, has_spot, winner_up, seed`.

`winner_up` is carried here so **calibration can use every fill**. It used to be
reconstructed from the ledger's settlement markout, which only covers fills in
roughly the last 10 s of a 300 s market — an unrepresentative slice. Joining the
ledger to this frame on `(run, market_id)` gives every fill the market's actual
outcome.

Per-liquidity fill counts and the mean `delta_quality_c` are **not** in this
frame; both are one `groupby` off `ledger.parquet`, which is where the
per-liquidity detail lives.

### 5.3 Per-tick diagnostics — `ticks.parquet`, opt-in

Enabled by `emit_ticks=True`, optionally restricted to a market subset (it is
~3,000 rows per market). Carries `market_id, t_ms, s, sigma, z, fair_p,
eff_bid, eff_ask, book_bid, book_ask, mid, book_age_ms, spot, q, cash,
cum_pnl, orders_live`, so
a single market can be watched play out tick by tick and the retreated quotes
inspected directly against the book, and (via `spot`) against the venue BTC
price itself.

### 5.4 Plots — not here

`run()` writes data and stops. It produces no PNGs, and `Output` has no `plots`
flag. Plotting is the separate, optional `harness.report` module, which reads
run folders (`load_runs({label: dir, ...})`) and is never called by the engine —
a run sees exactly one configuration and so cannot draw the sweep comparison
anyone actually wants. See §4a of the stream-registry spec.

---

## 6. Run parameters

One public entry point, `harness.core.api.backtest` (re-exported as
`harness.backtest`), returning a `BacktestResult(run_dir, summary, ledger,
markets, ticks)`:

```python
from harness import backtest, QuoteParams, ExecConfig, Sample, Output
from harness.core.latency import LatencyModel

result = backtest(
  model   = "../models/normal_qq",   # shared block set; None -> defaults only
  streams = ("book", "strikes", "spot"),      # recorded into the manifest
  quote   = QuoteParams(e_s=, e_z=, e_p=, rpl_s=, rpl_z=, rpl_p=,
                        max_pos=, tick=0.01, shares=1.0),
  execn   = ExecConfig(latency=LatencyModel(place_ms=100, cancel_ms=100,
                                            take_ms=200, taker_lock_ms=250,
                                            jitter_frac=0.0,
                                            move_cancel_ms=None),
                       fees=None,             # a POLICY input: see 3.1
                       max_book_age_ms=1000.0, requote_every=10,
                       min_tte_s=0.0, max_tte_s=300.0,
                       fill_params={},        # e.g. {"penetration": 0.01}
                       apply_daily_minimum=False),
  sample  = Sample(t0=, t1=, days=(), markets=(), max_markets=None,
                   require_spot=False, require=("spot",)),
  output  = Output(emit_ticks=False, tick_markets=(), seeds=(0,)),
  episodes = episodes,               # the Episode list to replay
  investigation_dir = HERE,          # blocks resolve here; runs are written here
)
```

There is **no `blocks=` argument and no `BlockSet`**: slots are files, resolved
investigation-first, then `model`, then `harness/blocks/defaults/` (§2.1).
There is no `mode` either: making and taking are one policy (§3.1). `fees` is a
run parameter that the *policy* reads, because the post and cross thresholds are
fee-adjusted; leaving it `None` falls back to the resolved `fees` block, and
`run` stamps whichever schedule it resolved back onto the config so the policy
prices against exactly the schedule the ledger charges.

`Sample` selects **markets**: a date range (`t0`/`t1` on `open_ts`), an explicit
day list, an explicit market list, a cap, and `require=(...)`, which excludes any
market where a named stream has no usable observation — excluded meaning absent,
never scored as zero, with the per-stream drop counts reported in
`summary["sample"]["dropped"]`. It has no `split` and no `tte_range`: the
time-to-expiry window is a *policy gate*, `ExecConfig.min_tte_s` /
`max_tte_s`, applied inside the episode rather than by dropping markets.

Below the API sit `harness.core.run.run(investigation_dir, quote, execn, sample,
output, episodes, inputs=(), model_dir=None, streams=())` — the single-arm
primitive, which also fingerprints `inputs` into the manifest — and
`harness.core.sweeps.run_with_sweeps(...)`, which adds the arms of §8.

---

## 7. Provenance

```
../investigations/2026-09-09-t-distribution-link/
  link.py            # the override, shadowing the model directory
  run.py
  runs/2026-09-09T14-22-05__a3f9c1/
    manifest.json    # every slot: resolved path + sha256; plus the config,
                     #   the fingerprinted inputs, the seeds, the git commit
    blocks/          # frozen copy of every block file used
    ledger.parquet   markets.parquet
    ticks.parquet    # if emit_ticks
    summary.json     # headline, per_seed, gates, sample drops, caveats
```

`manifest.json` records the sha256 of every block file actually used, the
harness git commit, the fingerprinted input files (whatever was passed as
`inputs`), the resolved config and the seeds. The block files themselves are
copied in, so a run folder still answers *what was this model?* after the
investigation folder has moved on — which is why referencing a shared model
directory instead of copying blocks costs nothing in provenance. The run id is a
timestamp plus a short hash of the **config**, making identical configurations
visibly identical; that hash covers `fill_params`, the tte window and the fee
schedule, because two sweep arms differing only in `fill_params` once shipped
run folders with the same suffix.

There is no `params.json`, no `inputs.json`, no `report.md` and no PNG: the
config and the inputs live inside `manifest.json`, and drawing is
`harness.report`'s job (§5.4).

---

## 8. Measurement and gates

Metrics in **c/share** and **$/market** — the units the existing studies use, so
results are comparable across repos.

Four discipline gates. `final-model-architecture.md` §4 states these exist and
are *"applied inconsistently"*; making them automatic is the cheapest available
fix. **Three of the four are automatic** — `stats.run_gates`, which every run
calls, writing each gate's verdict and its numbers into `summary["gates"]`:

1. `sign_survives_periods` — the sign survives every calendar period
2. `ci_excludes_zero` — the **day**-blocked bootstrap CI excludes zero (days,
   not markets: markets inside a day share a regime, a book and a competitor
   set)
3. `delete_top_10` — deleting the ten best markets does not flip the sign

The fourth — **period-blocked refits, never a single global fit** — is a
property of how `s` was fitted, and this harness imports `s` rather than fitting
it (§Scope). It cannot be enforced here; it belongs to whatever produced the
fair export, and a run cannot certify it.

Three sweeps attach to a headline. The first two are
`harness.core.sweeps.run_with_sweeps`, which replays the base configuration
once per arm and writes the arms into `summary["sweeps"]`; the third is just
`Output.seeds`, which every `run()` already loops over into
`summary["per_seed"]`:

- **latency** — 0 / 100 / 200 / 250 / 500 ms, each applied to `place`, `cancel`
  and `take` together (the 250 ms venue lock still floors the take path, so the
  lower rungs move the maker paths only)
- **fill optimism** — `optimistic` / `adverse_lag` / `penetration`, which differ
  in `penetration` *and* in cancel latency (§3.3)
- **seeds** — multi-seed, since jitter makes runs stochastic

`run()` itself is the single-arm primitive and attaches none of them: a run that
reports one maker number without the fill sweep is a run that has not been
asked the question.

### The standing caveat

`summary.json` carries, on every run, the measured bias of the substrate itself:
the 100 ms grid arm earns **+$0.142/market more** than an events arm
(CI [+0.053, +0.239]), replicated independently on `red_fast` at **+$0.124**
(CI [+0.025, +0.229]). Same size, same sign, two models — a property of the
replay clock, not of a chain. No result leaves this harness pretending the
100 ms grid is free.

---

## 9. Repository layout

```
Gambling104/                     # git root (initialised 2026-09-09)
  backtesting_5m/
    pyproject.toml               # installs the `harness` package (pip -e)
    data/                        # existing build, untouched
    docs/superpowers/specs/
    harness/
      paths.py  io.py
      core/       api.py config.py episode.py latency.py ledger.py loop.py
                  provenance.py run.py stats.py sweeps.py types.py
      streams/    spec.py registry.py reader.py writer.py catalog.py
                  validate.py
      report/     load.py figures.py    # optional; never called by run()
      build/      spot_5m_100ms.py spot_5m_100ms_london.py episodes.py
      blocks/defaults/      fair.py vol.py f.py link.py quote.py
                            execution.py fill.py fees.py
      blocks/placeholders/  fair_flat.py
      tests/
  models/                        # shared block sets, referenced by name
  investigations/                # all research lives here, not in the module
```

`harness` is an installed package, so a runner anywhere — including
`../investigations/<x>/run.py` — does `import harness` with no `sys.path`
surgery. See §4b of the stream-registry spec for why that matters.

---

## 10. Declared assumptions and known limits

1. **No depth, no trade tape, no queue.** Every maker fill is a model, not a
   measurement. Maker results are ranges over fill optimism; a single maker
   number is not a valid output of this harness.
2. **The 100 ms grid flatters results** by about +$0.13/market, measured twice.
   Carried on every run.
3. **Spot coverage starts 2026-08-17**, three days after the book panel.
4. **`s` is imported, never fitted here.** `RUN_MARKETS` §9.5 bars book
   information from `fair`; this harness never feeds the book back into the
   chain. Execution policy may read the book — the chain may not.
5. **One instrument, one regime, 25 days.** The blocked gates exist because two
   backtest splits already disagreed 2x on where `min_edge` optimises.
6. **Size is assumed nil-impact.** Lot sizes are small by default and impact is
   not modelled; size is a declared assumption, never a fitted parameter.
7. **Endgame is treated as settled, not as opportunity.** beta = 0 inside
   tau = 13 s; two independent studies agree the endgame is not collectable. The
   harness can measure it, but a positive endgame result should be read as a bug
   first.
8. **The venue-to-panel clock offset is assumed, not measured** (§1.2). Default
   0.0 s, plausible band 0-74 ms, admitted in `summary["caveats"]` on every run.
   Every spot-derived number is conditional on it and should be swept over that
   band the way latency and fill optimism are.

## Open items

- Whether the taker latency default should be 200 ms (operator-set) or 276 ms
  (measured). Implemented as 200 ms, with the 250 ms venue lock enforced as a
  floor in `LatencyModel.draw` and the latency sweep expected on every headline.
- The fair export format from Gambling102 — needs a per-`(market_id, t_ms)`
  contract and a statement of what information went into each value, so it can
  be lookahead-audited on import.
