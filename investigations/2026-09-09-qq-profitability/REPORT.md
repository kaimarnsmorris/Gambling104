# Making the QQ model profitable — first pass

**Sample:** 1,102 markets, 2026-08-17..21, the three-way overlap where the
book panel, the venue spot build and the Chainlink capture all exist.
**Baseline arm:** `e_p=0.02, rpl_p=0.002, e_z=0.3`, one seed.

**One arm came out positive.** It fails two of the three discipline gates, so
it is a lead, not a result. What follows is how it was found, because the
route matters more than the number.

---

## 1. Fees stopped being the problem

| | c/share |
|---|---|
| gross markout over all fills | **−2.285** |
| fee | 0.138 |

Fee drag is 6 % of the loss. Every earlier conclusion about fees being ~60 %
of it was from a configuration that crossed the book far more often. The loss
is now in the fills themselves.

## 2. Both sides lose, and the maker side loses 3.4× more

| liquidity | fills | markout c | fee c | net c |
|---|---|---|---|---|
| maker | 10,579 | −2.868 | −0.220 | −2.648 |
| taker | 4,297 | −0.850 | +1.020 | −1.870 |

Maker much worse than taker is the adverse-selection signature — we are filled
when the book is about to move through us. But taker markout is **also**
negative, and a taker fill is one we *chose*: crossing on our own signal and
still losing 0.85 c means the fair is wrong as well. Two problems, not one.

## 3. The model is badly overconfident

Reliability over 3.24 M quoting ticks:

| predicted | realised | book | model error | book error |
|---|---|---|---|---|
| 0.024 | 0.138 | 0.153 | **−0.114** | +0.014 |
| 0.150 | 0.244 | 0.267 | −0.094 | +0.023 |
| 0.750 | 0.668 | 0.675 | **+0.082** | +0.006 |
| 0.850 | 0.741 | 0.737 | **+0.109** | −0.004 |
| 0.989 | 0.937 | 0.917 | +0.052 | −0.021 |

The book is inside ±0.02 throughout. The model is out by up to 11 points, and
43 % of all ticks sit in the two extreme buckets where it is worst. This is
what "the gap is resolution, not calibration" looked like all along.

## 4. Sigma is 2.2–2.5× too small — measured, not inferred

Probability space cannot separate "wrong location" from "understated
variance", because `link(z)` folds both into one number. BTC space can, using
`settle(N) = strike(N+1)`:

| tte (s) | bias of s | rmse(s) | sigma claimed | **realised / claimed** |
|---|---|---|---|---|
| [0,15) | −0.06 | 2.70 | 0.54 | 4.97 |
| [15,30) | +0.01 | 5.02 | 2.54 | 1.98 |
| [30,60) | +0.68 | 16.32 | 7.31 | **2.23** |
| [60,120) | +1.38 | 38.49 | 17.43 | **2.21** |
| [120,240) | +6.06 | 69.50 | 28.60 | **2.43** |
| [240,300) | +10.33 | 91.67 | 37.10 | **2.47** |

Sweeping `link(z / k)` offline over the same ticks, Brier bottoms at
**k = 1.6**, not 2.2–2.5. That gap is the fat tails: RMSE is inflated by the
tails, so matching it over-widens the body. **A normal link cannot be right at
both ends** — k = 2.5 zeroes the tail error and then under-confidences the
middle.

The `bias of s` column growing to +$10 is almost certainly this window's
upward drift, not a model property. It should not be fitted.

## 5. The fair value has no location edge at all

| forecast of the settle | rmse, USD |
|---|---|
| `s` (the model, basis-learned) | **62.77** |
| the raw venue mid (BTC/USDT, uncorrected) | **62.68** |
| carrying the last Chainlink print | 63.40 |

The basis-learning fair block is **no better than the uncorrected venue mid**
at forecasting where the oracle settles. All of that machinery buys $0.6 out
of $63, and the raw mid actually edges it.

This is the finding that reframes everything else.

## 6. So fixing the calibration does not pay — confirmed

Sigma scale against half-spread, `e_z ≤ 0.02` so no arm can win by declining
to trade:

| scale | pnl/market | c/share | fills |
|---|---|---|---|
| 1.0 | **−5.94** | −1.459 | 44,836 |
| 1.3 | −6.35 | −1.475 | 47,435 |
| 1.6 | −6.77 | −1.561 | 47,775 |
| 1.9 | −6.94 | −1.593 | 47,973 |
| 2.5 | −7.03 | −1.653 | 46,846 |

Monotonically worse, and fills *rise*. Better forecasts, worse PnL. With no
location edge, recalibrating the width does not create one — it only changes
which coin flips get taken, and takes more of them.

## 7. The book's tail bias is largely a tick-count illusion

The reliability table suggested a classic favourite-longshot bias. Re-measured
with the **market** as the unit and a day-blocked bootstrap, the sign in
(0, 0.1] **reverses**. Three bands have intervals excluding zero:

| band | markets | book | realised | edge to SELL, c | CI |
|---|---|---|---|---|---|
| [0.00,0.05) | 540 | 0.012 | 0.033 | −2.14 | [−3.06, −0.69] |
| [0.80,0.90) | 508 | 0.851 | 0.870 | −1.90 | [−3.88, −0.06] |
| [0.95,1.00) | 539 | 0.989 | 0.983 | +0.55 | [+0.07, +1.40] |

Neither survives the gates. Buying sub-5c longshots is +2.14 c/share and
**+0.34 after deleting the best 10 markets** — 18 winners in 540, so the edge
*is* about ten bets. The 0.80–0.90 band survives concentration (+1.90 →
+1.58) but flips sign on 08-19.

Three thousand ticks inside one market are one bet observed 3,000 times. The
tick-level table's apparent precision was fiction, and it pointed the wrong
way.

## 8. The lead: the model beats the book only in the last minute

Brier by time to expiry, model against book:

| tte (s) | model | book | gap |
|---|---|---|---|
| [0,30) | 0.0159 | 0.0145 | +0.0014 |
| **[30,60)** | **0.0329** | **0.0354** | **−0.0025** |
| [60,120) | 0.0880 | 0.0851 | +0.0028 |
| [120,240) | 0.1678 | 0.1575 | +0.0103 |
| [240,300) | 0.2260 | 0.2167 | +0.0093 |

One window where the model forecasts better than the market — and it is where
the fair block starts blending realised Chainlink prints, i.e. where it knows
something the book prices more slowly. The strategy was diluting that across
240 s of trading at a disadvantage.

Gating the quote on time to expiry:

| window | pnl/market | c/share | fills |
|---|---|---|---|
| **[0,60)** | **+0.309** | **+0.749** | 4,538 |
| [30,60) | +0.269 | +0.802 | 3,695 |
| [25,65) | +0.223 | +0.511 | 4,806 |
| [20,75) | −0.015 | −0.024 | 6,741 |
| [15,90) | −0.358 | −0.421 | 9,386 |
| [0,120) | −1.305 | −0.964 | 14,920 |
| [0,300) | −5.935 | −1.459 | 44,836 |

Monotone in window width across eight windows, and the ordering was
**predicted by the Brier table before the test was run**. Widening sigma still
hurts here too (scale 1.0 > 1.6 > 2.2 at every window).

### The gates say no

```
sign_survives_periods  FAIL   per-period [-0.433, +0.940, +1.180]
ci_excludes_zero       FAIL   CI [-0.658, +1.934]
delete_top_10          pass   +0.309 -> +0.045   (85 % of it is 10 markets)
```

Positive, structurally coherent, and statistically indistinguishable from
zero. On 1,102 markets over 3.9 priceable days it could not be otherwise.

---

## What to do next, in order

1. **Get more oracle days.** Every gate here fails for want of sample: the
   Chainlink capture stops 2026-08-21 01:59 while the book panel runs to
   09-08. The NDJSON recorder archive reaches 08-26 and is not yet converted.
   This is the binding constraint on *every* question below, and it is
   engineering rather than research.
2. **Replace the normal link with a fat-tailed one.** §4 shows no single
   scale can fix both ends, which is a shape statement, not a tuning one. The
   sibling session's t-distribution work is the natural candidate.
3. **Understand the last minute before trusting it.** §8's edge may be the
   blend genuinely leading the book, or it may be the 60 s TWAP becoming
   mechanically predictable as it fills with realised prints. Those have very
   different futures. Compare `s` against the book conditioned on how much of
   the settlement window has already elapsed.
4. **Attack the maker markout separately.** −2.87 c/share is an execution
   problem, not a forecast one, and it is the largest single number in this
   report. The cancel is losing the race; the latency model is where that
   lives.
5. **Do not** fit the +$10 long-tte bias in §4, and do not tune `e_z` upward:
   wide arms improve PnL only by trading less, and their c/share gets worse.

---

## 9. A second quote rule, and what the plot shows that the table did not

`e_p=0.04, rpl_p=0.0035, max_pos=100` (**B**) against `0.02, 0.002, 50`
(**A**). Note these differ in kind near the cap, not only in degree: `q` in
the quote algebra is raw shares, so the lean at full inventory is
`max_pos * rpl_p` -- 10 c for A, **35 c for B**, past the point where one
side clips to the [0,1] bound and simply stops quoting.

| arm | fills | pnl/market | c/share |
|---|---|---|---|
| **B, last 60 s** | 4,086 | **+1.005** | **+2.709** |
| B, 30-60 s | 3,266 | +0.972 | **+3.278** |
| B, 20-75 s | 5,776 | +0.643 | +1.227 |
| A, last 60 s | 4,538 | +0.309 | +0.749 |
| A, 30-60 s | 3,695 | +0.269 | +0.802 |
| B, all 300 s | 35,796 | −4.696 | −1.446 |
| A, all 300 s | 44,836 | −5.935 | −1.459 |

**B beats A at every one of the twelve window/e_z combinations tested**, by
roughly 3x on the last-minute arms and by $1.2/market even on the full
window. That comparison is the robust part of this report: it is a consistent
ordering across twelve paired arms, not a single maximum. B also loses less
to concentration -- delete-top-10 takes A from +0.309 to +0.045 (85 % gone)
but B only from +1.005 to +0.501.

### And then the cumulative plot

`cum_pnl.png` shows the whole profit arriving as one step around market 560,
then flat for the remaining 450. Broken out by day:

| day | markets | PnL |
|---|---|---|
| 2026-08-17 | 236 | **−177** |
| 2026-08-18 | 275 | **−93** |
| **2026-08-19** | 283 | **+1,595** |
| 2026-08-20 | 284 | **−244** |
| 2026-08-21 | 24 | +27 |

**Four of the five days lose.** The 120 markets of 2026-08-19 02:45-12:45
made +1,779 while the other 982 markets lost −672. The headline +1.005 per
market is one morning.

This is not what delete-top-10 measures -- it is not ten lucky markets, it is
one regime lasting hours, which is why that gate passed while
`sign_survives_periods` failed. The cumulative plot showed it immediately and
no summary statistic in this report did. That is the argument for drawing the
curve before believing the mean.

**Standing conclusion, unchanged:** B is a genuinely better quote rule and the
last-minute window is a genuinely better place to trade, both robustly. The
strategy is still not profitable -- it has one good day in five and no
evidence of an edge that persists.
