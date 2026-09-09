# How much venue price movement does the 100 ms grid hide?

`measure.py` in this directory. Deterministic, no randomness. Run with:

```
python investigations/2026-09-09-grid-excursion/measure.py
```

## The question

`harness/build/spot_5m_100ms.py` samples the FIRST observation in every
100 ms bucket. Any excursion that forms and reverts inside a bucket is
invisible to the backtest. The owner's worry is directional: a venue
overreacting for tens of milliseconds could pick off a real quoter, and a
backtest that never samples the spike never books that loss. The grid is
biased toward calm, and calm is optimistic. This measures the size of that
blind spot using `l1_mid_10ms.parquet`, a 10 ms-cadence L1 mid series across
five venues (`bybit`, `cb`, `okx`, `perp`, `spot`).

## Coverage -- read this before the numbers

`l1_mid_10ms.parquet` covers 2026-08-19 02:00:00.01 UTC through
2026-08-21 01:59:59.99 UTC -- almost exactly 2 days. The harness's 6-day
evaluation sample runs 2026-08-19 -> 2026-08-24 (1,554 5-minute markets
total). Of those, only 569 markets (36.6%) have their full 300-bucket
window inside the data's time range -- everything from the second half of
08-19, and all of 08-20 through 08-24, is outside this file's coverage.

This is a partial-overlap measurement of roughly the first third of the
harness's sample, not a statement about the full 6 days. The numbers below
describe that 569-market, ~2-day slice. If venue volatility regimes differ
across the other 4 days (e.g. a genuinely calmer or wilder stretch), this
report does not know it and does not claim otherwise.

A second, unavoidable caveat: `l1_mid_10ms` is itself event-driven, not a
fixed 10 ms clock -- median inter-tick gap is 10 ms for `bybit`/`perp`/`spot`,
20 ms for `okx`, and 60 ms for `cb` (p99 gaps run 180-610 ms). A spike that
both forms and fully reverts between two consecutive ticks of this file is
invisible here too. That means every number below is a lower bound on
the true hidden excursion -- the real tail is at least this large, probably
larger, especially for `cb`.

## Method

Per-bucket excursion is computed exactly as max(|mid_t - mid_first|),
which for any set containing mid_first reduces to
max(bucket_max - mid_first, mid_first - bucket_min, 0) -- obtainable from a
single groupby(['venue','bucket'])['mid'].agg(['first','last','max','min','count']),
no per-row transform needed. Bucket boundaries are the harness's own: since
every market open (open_ts) is an integer multiple of 300 s, flooring
recv_ns to the global 100 ms epoch grid reproduces
bucket_venue_l1's per-market bucketing exactly (proved algebraically and
checked at runtime by `_prove_bucket_equivalence()` in `measure.py`).

## 1. Excursion distribution

All buckets, all 5 venues, restricted to the 569 covered markets
(6,710,393 bucket-venue observations):

| | median | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| USD | 0.00 | 0.005 | 8.20 | 19.60 | 540.69 |
| bps | 0.00 | 0.0007 | 1.16 | 2.77 | 78.97 |

The median is exactly zero -- not a rounding artifact. 27.0% of buckets
have only a single tick (no measurable intra-bucket movement is possible
by construction), and among the remaining multi-tick buckets, 86% still
show zero net mid change -- the venue simply didn't reprice between L1
updates in that window. Only about 1 in 7 buckets contains any price
movement at all. The blind spot the grid hides is a thin but real tail, not
a pervasive effect on every bucket. That tail is what stats 2-4 quantify.

Per venue (same restriction):

| venue | median $ | p90 $ | p99 $ | p99.9 $ | max $ | p99.9 bps | frac single-tick |
|---|---|---|---|---|---|---|---|
| bybit | 0.00 | 0.50 | 9.00 | 20.70 | 126.20 | 2.92 | 24.4% |
| cb | 0.00 | 0.48 | 6.07 | 15.26 | 135.34 | 2.15 | 60.9% |
| okx | 0.00 | 0.00 | 9.00 | 20.70 | 162.70 | 2.93 | 19.6% |
| perp | 0.00 | 0.00 | 8.40 | 19.68 | 444.60 | 2.76 | 14.7% |
| spot | 0.00 | 0.00 | 7.34 | 19.90 | 540.69 | 2.82 | 24.7% |

`cb` (Coinbase) has by far the most single-tick buckets (61%) because it
ticks roughly 3-6x less often than the other four venues -- its numbers are
the least reliable of the five and understate its true excursion tail the
most.

See `excursion_distribution.png` (left panel): a log-log histogram of
positive excursions. The bulk sits around 0.3-3 bps; a genuine, roughly
power-law tail runs out past 20 bps, with a handful of buckets above 50 bps.

## 2. Reversion -- does the hidden spike come back?

Materiality baseline: the median excursion among buckets where the mid
actually moved (raw and multi-tick medians are both 0, so they're useless
as a threshold -- see stat 1). Pooled, that's $2.49. Of buckets exceeding
this baseline, the fraction that returned to within 1 bp of mid_first by
the bucket's end:

| venue | n material buckets | frac reverted |
|---|---|---|
| bybit | 89,569 | 76.2% |
| cb | 37,111 | 80.6% |
| okx | 78,644 | 69.9% |
| perp | 86,895 | 75.2% |
| spot | 46,422 | 69.8% |

70-81% of material intra-bucket moves reverted by the time the harness's
next sample would have been taken. These are precisely the moves the grid
is guaranteed to miss (the panel only ever sees mid_first, before and
after) -- and precisely the moves a real quoter would have been exposed to
for the tens of milliseconds they were live. This is the core of the
owner's concern: a majority of the tail is exactly the "spike then reverts"
shape that hurts a live quoter and is invisible to the backtest.

## 3. Frequency of material excursions per market

Per venue, using that venue's own median moving-bucket excursion as the 1x
baseline (thresholds differ by venue: `cb` moves less per tick, so its
baseline is smaller), events per 300-bucket (5-minute) market:

| venue | 1x baseline ($) | events/market >=1x | events/market >=2x | events/market >=5x |
|---|---|---|---|---|
| bybit | 3.20 | 127.6 | 48.3 | 4.9 |
| cb | 0.74 | 149.8 | 101.2 | 39.2 |
| okx | 3.80 | 102.2 | 37.4 | 3.3 |
| perp | 3.25 | 125.6 | 48.6 | 4.6 |
| spot | 2.37 | 86.7 | 46.0 | 8.4 |

Reading `bybit` as representative: in an average 5-minute market, roughly
5 buckets exceed 5x the typical intra-bucket move -- i.e. about once per
minute, a bucket on this venue contains a hidden excursion at least 5x the
size of the excursion the grid usually hides. That is not a once-in-the-sample
tail event; it's a recurring, per-market feature of every 5-minute window in
this slice.

## 4. Translating an excursion into probability (vol.py / f.py / link.py defaults)

Using the shipped defaults exactly (SIGMA_AT_300S = 250 USD, sigma(tau) =
250*sqrt(tau/300), z = (level-strike)/sigma, logistic link), assuming the
market sits at-the-money when the spike hits (z=0 baseline, p=50c --
the most probability-sensitive point on the curve):

| excursion | $ | tau=240s: dp (c) | tau=120s: dp (c) | tau=30s: dp (c) |
|---|---|---|---|---|
| pooled p99 | 8.20 | 0.92 | 1.30 | 2.59 |
| pooled p99.9 | 19.60 | 2.19 | 3.10 | 6.17 |
| pooled max (observed) | 540.69 | 41.82 | 46.83 | 49.89 |

A p99.9 excursion moves fair value by 2-6 cents of probability depending
on time-to-expiry, and the effect gets larger as expiry approaches (same
dollar move, smaller sigma). At 30 seconds to expiry, a p99.9-sized hidden
spike is a 6-cent probability swing -- large enough to flip which side of a
quote is profitable, and events at this scale recur multiple times per
market (see stat 3). The observed maximum excursion ($541, on `spot`) would
move fair value by 42-50 cents at any of these horizons -- essentially from
whatever it was to fully priced-in -- though a single $541 event in a
~2-day, 5-venue sample of 6.7M buckets is exactly that: one extreme
observation, not a typical one.

## 5. Per-venue breakdown -- is one venue responsible?

Threshold = pooled p99.9 ($19.60). Share of extreme buckets, and the fairer
per-venue rate (extremes divided by that venue's own bucket count):

| venue | n extreme buckets | share of extremes | rate per bucket |
|---|---|---|---|
| okx | 1,731 | 25.8% | 0.118% |
| bybit | 1,543 | 23.0% | 0.117% |
| perp | 1,540 | 23.0% | 0.100% |
| spot | 1,367 | 20.4% | 0.104% |
| cb | 520 | 7.8% | 0.049% |

No single venue dominates. `okx` has the highest rate but only
marginally above `bybit`; `perp` and `spot` sit close behind. `cb`'s rate is
roughly half the others' -- but `cb` also ticks 3-6x less often than the rest
(see coverage caveats), so at least part of its apparent calm is a sampling
artifact of the source data, not necessarily calmer venue behaviour. This
is not "one venue overreacts and the others are fine" -- it's a broadly
shared property across venues, with `okx`/`bybit` marginally the noisiest
and `cb` the most under-observed.

See `excursion_distribution.png` (right panel).

## Verdict: does this justify converting raw venue feeds from npz?

On the numbers in this partial (2-of-6-day) sample: no, not as the next
priority.

What the data shows:
- The grid hides real, recurring, quantifiable risk: ~1 in 7 buckets moves
  at all, but when one does, it's material more often than not, 70-81% of
  those moves fully reverse within the bucket, and a p99.9 spike is worth
  2-6 cents of fair-value probability near expiry.
- No venue is uniquely responsible -- this is a property of the 100 ms grid
  interacting with normal multi-venue tick noise, not one bad feed that
  conversion would specifically fix.
- The tail is real but it IS a tail: the median bucket has zero movement,
  and the largest observed single-bucket effect (a 42-50c probability swing)
  occurred once in 6.7M bucket-venue observations.

What would change this verdict:
- Confirming the pattern holds over the other 4/6 days of the harness
  sample (this report cannot speak to that -- no data).
- A finding that fills/PnL in the existing (100 ms) harness are unusually
  sensitive near expiry specifically because of missed spikes of this size
  -- that's a strategy-level question this measurement doesn't answer.

Given the owner has a competing, already-identified priority (the
miscalibrated vol model) and this measurement shows a real but bounded,
tail-only, not-venue-specific effect on roughly a third of the sample
period, the vol model is the better use of limited time right now. The
npz conversion is a reasonable thing to revisit if a future strategy proves
sensitive to expiry-adjacent probability swings of a few cents, or once the
full 6-day sample can be checked -- not before.

## Files

- `measure.py` -- the measurement.
- `excursion_distribution.png` -- log-scale tail histogram (left) and
  per-venue typical-move-vs-tail comparison (right).
- `coverage.csv`, `excursion_distribution.csv`, `reversion_stats.csv`,
  `frequency_per_market.csv`, `probability_translation.csv`,
  `venue_extreme_share.csv`, `venue_extreme_counts.csv` -- the raw numbers
  behind every table above.
