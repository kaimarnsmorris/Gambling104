---
title: A block-composed backtesting harness for the BTC up/down 5 m book
date: 2026-09-09
status: design — approved in brainstorming, not yet implemented
related:
  - ../../../data/README.md
  - Gambling102 docs/strategy/2026-09-05-final-model-architecture.md
  - Gambling102 docs/polymarket/fees-and-taker.md
---

# Backtesting harness — design

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

**Clock gate — the build refuses to write without it.** The panel's captures
were corrected onto polydata's vantage; `stream_venue_l1` is a different host
with an independent, drifting clock. A per-day venue-to-polydata offset is
measured the way `s03b_vantage.py` measures capture offsets, written to
`results/venue_vantage_offsets.tsv`, and the build **fails loudly** if any day
lacks one. Rationale: beta(tau,h) collapses to zero inside tau = 13 s, so a
silent 50 ms misalignment is not cosmetic — it is the difference between signal
and noise at exactly the horizons under test.

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
| `n_src, src_spread_c` | corroboration, from the panel |
| `spot_*`, `spot_age_ms`, `has_spot` | venue L1 fields on the grid |
| `s` | fair value, USD, from the export (NaN where absent) |
| `strike`, `settle`, `winner_up` | settlement truth |
| `day`, `period` | for blocked splits and CIs |

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

A block *slot* is a filename. Resolution for each slot, in order:

1. the investigation folder
2. `harness/blocks/defaults/`

Nothing is registered and nothing is named. Writing `link.py` in your
investigation folder *is* the override.

| slot | file | contract |
|---|---|---|
| fair | `fair.py` | `precompute(ep) -> s[N]` (USD) |
| vol | `vol.py` | `precompute(ep) -> sigma[N]` |
| f | `f.py` | `standardise(s, strike, sigma) -> z`; bound to `strike` and `sigma[i]` per tick, so the quote algebra calls it as `f(.)` on the level alone |
| link | `link.py` | `link(z) -> p` in (0,1) |
| quote | `quote.py` | `quotes(s_i, q, params) -> (eff_bid, eff_ask)` |
| execution | `execution.py` | order policy — post / cross / cancel / gate |
| fill | `fill.py` | whether a live order fills, and at what price |
| fees | `fees.py` | `charge(...) -> usd`, negative for rebates |

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
for i in range(N):
    eff_bid, eff_ask = quote(s[i], q, params)
    actions = execution.decide(i, eff_bid, eff_ask, q, book[i], state)
    fills   = fill.resolve(actions, ep, i, latency.draw(rng), state)
    q, cash = apply(fills, fees, ledger)
settle(q, ep.winner_up)
```

`execution` owns policy: post vs cross, order lifetime, requote cadence, cancel
rules, `max_book_age_ms` gating, position caps (**enforced on the taker path
too** — the 2026-05-20 taker-cap-bypass lesson). `fill` owns whether a live
order trades and at what price.

### 3.2 Latency

Parametric, seeded, with jitter. Defaults:

| path | default | source |
|---|---|---|
| place / cancel (maker) | **100 ms** | operator-set |
| take | **200 ms** | operator-set |

Two caveats carried in code, not prose:

**(a) The 250 ms taker lock is a venue mechanic, not an assumption.** The venue
holds marketable crypto up/down orders ~250 ms, non-cancellable. The taker fill
model enforces that as a *floor* independent of the latency parameter. It also
reconciles the measured 276 ms taker fill as 26 ms POST flight + 250 ms lock.

**(b) Taker edge is not robust across this range.**
`final-model-architecture.md` §2.1 shows it losing significance between 200 ms
and 500 ms (CI [+0.157, +0.587] falling to [-0.026, +0.309]). The 200 ms default
is therefore reported *always* alongside its sweep, never alone.

Measured constants available as an alternative `LatencyModel`: POST flight 26 ms
floor / 97 ms p5; cancel round trip 28 ms quiet, **73-165 ms in a move**. The
move-conditional cancel is the adverse-selection mechanism itself and belongs in
any serious maker run.

Seeding is **per-episode**: `hash(market_id, run_seed)`. Results are
byte-identical regardless of worker count or completion order, so reproducibility
and parallelism do not trade off.

### 3.3 Fill models

Default `fill.py` is **touch-fill with adverse-selection lag**: a resting order
fills when the market's L1 crosses it, booked at the price after the reaction
delay — so you are filled hardest when the book is moving against you. Shipped
alternatives for bounding: optimistic touch-fill (upper bound) and
penetration-required (conservative). Every maker headline reports the range.

---

## 4. Fees — `fees.py`

Verified on-chain, 9,073 fills, 0.0 % median error, constant since 2026-05-20 or
earlier.

```python
BASE_FEE_RATE      = 0.07
MAKER_REBATE_PHI   = 0.20     # measured 0.199-0.209, both wallets, all periods
TAKER_REBATE_RHO   = 0.0833   # Silver tier

def charge(side, liquidity, shares, p) -> float:
    """USD. Negative means we are paid."""
    base = BASE_FEE_RATE * p * (1.0 - p) * shares
    if liquidity == TAKER:
        return  base * (1.0 - TAKER_REBATE_RHO)
    return -base * MAKER_REBATE_PHI
```

Taker fee peaks at **1.75 c/share at p = 0.50** and vanishes at the tails —
which is *why* taking is a tail-only exception: near mid it needs more than
1.75 c of edge before adverse selection. Maker is a **negative fee** averaging
0.234 c/share.

The 1.263 c/share figure in the strategy docs is the realised average over the
fill distribution, **not** a constant. Do not hard-code it.

**Not modelled by default:** the $1.00/day per-stream minimum rebate payout
(dust days are not paid). Available as `fees.apply_daily_minimum()` for runs
whose daily maker rebate is small enough to matter.

---

## 5. Outputs

### 5.1 Fill ledger — `ledger.parquet`, one row per fill

`market_id, open_ts, t_ms, side, liquidity, shares, price, fee_usd,
s, sigma, z, eff_bid, eff_ask, book_bid, book_ask, mid_at_fill,
q_before, q_after, latency_ms, order_age_ms, mid_t10, delta_quality_c,
markout_settled, seed, order_id, reason`

- `liquidity` — maker or taker, on every row.
- `fee_usd` — **negative for maker** (rebate). Gross and net are always both
  recoverable; a harness that reports only one will eventually report the wrong
  one, and the taker case (gross +1.085 against the fee) is exactly that trap.
- `mid_t10` — the book mid 10 s after the fill.
- `delta_quality_c` — signed markout in c/share: `100 * (mid_t10 - price)` for a
  buy, `100 * (price - mid_t10)` for a sell (prices are probabilities, so the
  factor of 100 puts it in cents). Positive is good. Where `t_ms + 10 s >= 300 s`
  the reference is the settlement outcome (0 or 1) rather than a mid, and the
  row is flagged `markout_settled = True`.

### 5.2 Per-market rollup — `markets.parquet`

PnL gross and net, fill counts by liquidity, max `|q|`, mean
`delta_quality_c`, settlement, coverage and `has_spot`.

### 5.3 Per-tick diagnostics — `ticks.parquet`, opt-in

Enabled by `emit_ticks=True`, optionally restricted to a market subset (it is
~3,000 rows per market). Carries `t_ms, s, sigma, z, fair_p, eff_bid, eff_ask,
book_bid, book_ask, mid, q, cash, cum_pnl, orders_live`, so a single market can
be watched play out tick by tick and the retreated quotes inspected directly
against the book.

### 5.4 Plots

`cum_pnl.png` (cumulative net PnL over the ordered market sequence) always;
per-market quote-versus-book panels when tick output is on.

---

## 6. Run parameters

```python
run(
  blocks = BlockSet(),                      # resolved from files; all optional
  quote  = QuoteParams(e_s=, e_z=, e_p=, rpl_s=, rpl_z=, rpl_p=, max_pos=),
  execn  = ExecConfig(mode="maker"|"taker"|"both",
                      latency=LatencyModel(place_ms=100, cancel_ms=100,
                                           take_ms=200, jitter=...),
                      max_book_age_ms=, fees=FeeSchedule()),
  sample = Sample(t0=, t1=, days=[], markets=[], tte_range=(),
                  split="train"|"test"|"all", max_markets=),
  output = Output(emit_ticks=False, tick_markets=[], seeds=[0,1,2], plots=True),
)
```

`Sample` supports an arbitrary time subset — date range, explicit day list,
explicit market list, or a tau window — so an investigation can target the
endgame, one day, or one market without touching the engine.

---

## 7. Provenance

```
investigations/2026-09-09-t-distribution-link/
  link.py            # the override
  run.py
  runs/2026-09-09T14-22-05__a3f9c1/
    manifest.json    # slot -> resolved path + sha256, for EVERY slot
    blocks/          # frozen copy of every block file used
    params.json  inputs.json  ledger.parquet  markets.parquet
    ticks.parquet    # if enabled
    summary.json  report.md  cum_pnl.png
```

`manifest.json` records the sha256 of every block file actually used, the
harness git commit, the panel and spot build identities, and the seeds. The
block files themselves are copied in, so a run folder still answers *what was
this model?* after the investigation folder has moved on. The run id is a
timestamp plus a short hash of the manifest, making identical configurations
visibly identical.

---

## 8. Measurement and gates

Metrics in **c/share** and **$/market** — the units the existing studies use, so
results are comparable across repos.

Four discipline gates, promoted from habit to a function every run calls.
`final-model-architecture.md` §4 states these exist and are *"applied
inconsistently"*; making them automatic is the cheapest available fix.

1. sign survives every calendar period
2. day-blocked bootstrap CI excludes zero
3. delete-the-ten-best does not flip the sign
4. period-blocked refits, never a single global fit

Three sweeps attach to every headline automatically:

- **latency** — 0 / 100 / 200 / 250 / 500 ms plus the measured distribution
- **fill optimism** — optimistic / adverse-lag / penetration-required
- **seeds** — multi-seed, since jitter makes runs stochastic

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
    data/                        # existing build, untouched
    harness/
      core/       episode.py  loop.py  ledger.py  stats.py  provenance.py
      build/      spot_5m_100ms.py  episodes.py
      blocks/defaults/  fair.py vol.py f.py link.py quote.py
                        execution.py fill.py fees.py
      tests/
    investigations/
    docs/superpowers/specs/
```

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

## Open items

- Whether the taker latency default should be 200 ms (operator-set) or 276 ms
  (measured). Implemented as 200 ms with the 250 ms venue lock enforced in the
  fill model and the sweep mandatory.
- The fair export format from Gambling102 — needs a per-`(market_id, t_ms)`
  contract and a statement of what information went into each value, so it can
  be lookahead-audited on import.
