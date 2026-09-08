# BTC up/down 5 m book — reconciled 100 ms panel

Two files, covering every BTC up/down 5 m market from the moment the book moved
to the **60 s** Chainlink TWAP.

| file | rows | what it is |
|---|---|---|
| `book_5m_100ms.parquet` | 20,938,352 | the UP token's L1, one row per (market, 100 ms bucket) actually observed |
| `strikes_5m.parquet` | 7,340 | `market_id`, `open_ts`, `strike` |

**7,111 markets**, 2026-08-14 00:05 → 2026-09-08 13:45 UTC.

⚠ **There is no book size in this dataset, because none exists anywhere.** The
archive carries no depth or quantity column for the Polymarket CLOB — only L1
price. So fill probability and queue position cannot be measured from this, and
a maker backtest built on it is resting on an assumed fill model. It is sound
for price-path, calibration and settlement work; it is not a fill-level
market-making substrate. That would need a new depth + trade-tape capture,
which nothing can backfill.

Rebuild with `scripts/s00..s04` in order; `s01` needs `Z:` mounted, `s02` needs
SSH to polydata. Numbers here are reproduced into `results/` by the build —
read them from there in code, not from this prose.

## The era boundary

The 5 m book settled on a **30 s** TWAP from the 2026-08-07 cutover and moved to
**60 s** at the market opening **2026-08-14 00:00:00 UTC** (epoch 1786665600).
Established by binary-searching gamma's `cryptoMarketConfig.twapLookbackSeconds`
across the cutover: the market opening 23:55:00 reads 30, the next reads 60.
All 7,365 markets enumerated at or after that boundary report 60, and the build
refuses to run if any reports otherwise — the filter is a per-market property,
not a date range.

⚠ The repo disagreed with itself about this. `2026-09-05-final-model-architecture.md`
says "around 2026-08-16"; that is wrong. The `datastreams_memo` figure of
2026-08-14 is right.

## Schema — `book_5m_100ms.parquet`

| column | type | meaning |
|---|---|---|
| `market_id` | string | CLOB token id of the **UP** token (`clobTokenIds[0]`) |
| `open_ts` | int64 | the market's window open, epoch seconds, on a 300 s boundary |
| `t_ms` | int32 | milliseconds since open, 0…299900, always a multiple of 100 |
| `recv_ms` | int64 | **when we received this book**, epoch ms, on polydata's vantage |
| `bid`, `ask`, `mid` | float32 | the UP token's L1, as probability |
| `src` | string | which capture supplied it: `archive` 72.5 %, `tdb_old` 24.9 %, `tdb_live` 2.6 % |
| `n_src` | int32 | independent captures that saw this bucket **and agreed to the cent** |
| `src_spread_c` | float32 | max − min across captures, in cents; null where only one saw it |

`tte = 300 − t_ms/1000`, so it is not materialised.

## Two traps in the time columns

**`recv_ms` is a receive time on one vantage, but it is not event-resolved.**
Neither capture records book *events*. Both are 10 Hz state samplers — the
archive's `ts` is the quoter's republish cadence, dt exactly 0.100 s. So
`recv_ms` means "the first 10 Hz sample that showed this book", and the tick
actually arrived somewhere in the ~100 ms before it. Nothing in this repo can
do better; there is no event-level CLOB capture anywhere in it.

The captures sit at different distances from the exchange, so their raw
timestamps are corrected onto polydata's vantage before bucketing. The offset is
measured per source per day and **drifts**: the laptop runs ~0 ms behind
polydata on 08-14, out to 74 ms on 08-28, back to ~15 ms in September
(`results/vantage_offsets.tsv`). A constant would have been wrong.

**`t_ms` is a lookahead trap.** A row labelled `t_ms` holds an observation drawn
from `[t_ms, t_ms+100)`, so it was *not* knowable at `t_ms` — the earliest a
strategy could act on it is `t_ms+100`. The first observation in the bucket is
the one kept, so the row is as close to its own label as the capture allows, but
shift by one bucket before treating a row as a decision input.

## What is absent, and what is dropped

Nothing is carried forward. A bucket no capture observed has **no row** — a
market's rows are not a fixed 3,000 (median 2,940, i.e. 98 % coverage). A stale
book is worse than a missing one, because at read time it is indistinguishable
from a real quote.

**253 of 7,365 markets are excluded**, each with its reason in
`results/dropped_markets.tsv`:

| reason | markets |
|---|---|
| `no_open_10s` — capture missed the first 10 s | 115 |
| `no_close_10s` — capture missed the last 10 s | 83 |
| `coverage_below_90pct` — a hole in the middle | 30 |
| `no_strike` — no `priceToBeat` and no usable predecessor | 25 |

A further **328 single-bucket spikes** (0.0015 %) are removed: a book both
neighbours contradict by more than 20 c, which is the previous market's resolved
book surviving the roll. Being *extreme* is not the test — 94 % of rows in the
last 10 s legitimately sit at a bound, because that is what a binary does near
expiry.

## Where the numbers come from

| capture | span | role |
|---|---|---|
| polydata archive (`stream_poly_crypto_updown`) | 08-14 → live | carries `market_id`; the reference vantage |
| archived v1 `trading.db` (polydata `legacy_v1`) | 08-14 → 09-05 | a second host on the same CLOB |
| live v1 `trading.db` (this laptop) | 09-05 → live | same, current |
| gamma | all | which markets exist, `clobTokenIds`, `priceToBeat`, twap config |
| on-chain `ReportVerified` | 08-14 → 08-21 | Chainlink's **signed** 60 s TWAP — the strike gate |

⚠ **`trading.db` carries no market id.** Its quotes are attributed to the 300 s
window containing them — and the v1 quoter keeps publishing the *previous*
market's book for a few seconds after a roll, because it emits no signal until
the new strike resolves. Measured against the archive, disagreement over 5 c runs
100 % in the first second and decays to baseline by 6 s, so **trading.db quotes
before t_ms 6000 are dropped**. The archive is unaffected; it carries the id.

⚠ **Do not classify the timeframe by span.** `tf` and `kind` do not exist before
2026-08-17, and recovering the book from each market's observed span — what the
older `export_books_polydata.py` does — agrees with `tf` on only **66.6 %** of
the markets that carry one, because a partially captured 15 m market has a
5 m-sized span. Coverage cannot also be the classifier. Markets come from gamma.

## Gates — the build writes nothing if any fails

| gate | result |
|---|---|
| token orientation | direct agreement 0.895 / 0.908 vs **inverted 0.019 / 0.021** — both captures are the UP token |
| cross-capture, large gaps | disagreement over 5 c on **0.86 % / 0.51 %** of shared buckets |
| strike vs the signed on-chain 60 s report | 1,992 markets, median **and max** difference **$0.00** |
| `strike(N) == settle(N−1)` | **100 % exact** on all 7,330 testable back-to-back pairs |
| grid integrity | 0 duplicate (market, bucket); every `t_ms` on the 100 ms grid and in range |
| era | 0 rows before the cutover; 0 rows without a strike |

**On the ~10 % of shared buckets where the two captures differ by a cent:** that
is not error, it is physics. Both are 10 Hz samplers, so while the book moves
they catch it at different instants — the median difference when they disagree is
exactly **1.0 c**, one tick, and the rate scales with movement (5.6 % when the
book is static, 46.5 % when it moved 3+ ticks). `src_spread_c` exposes it per
row. The gate is therefore set on gaps over 5 c, which sampling cannot explain.

## Known limits

- **13.5 M of 20.9 M buckets have two or more captures; 7.5 M have one.** Where
  `n_src` is 1 nothing corroborates the quote.
- **`strikes_5m.parquet` has 7,340 markets, the panel 7,111.** The strike table
  is every market with a known strike; the panel is those that also survived the
  completeness filter. Every panel market has a strike; not every strike has a
  panel.
- **Settlement is not stored — it is the next market's strike.** `settle(N)` is
  the strike of the market opening at `open_ts + 300`, which is why `open_ts`
  is carried here. Verified exact on 7,330 of 7,330 pairs. It resolves for
  7,108 of the 7,111 panel markets; the 3 that fail are at the very end of the
  sample, where no successor has settled yet. `winner_up = settle >= strike`,
  ties Up.
- **The on-chain strike gate only covers 08-14 → 08-21**, the span of the
  `chain_reports` capture. After that, the strike rests on gamma plus the chain
  property, both of which agreed perfectly where they could be checked.
