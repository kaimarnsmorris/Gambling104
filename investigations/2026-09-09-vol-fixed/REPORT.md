# vol-fixed — is sigma why the model is badly calibrated?

> **CURRENT ANSWER IS IN PART III — read that first.** With the warm-up on, the
> basis halflife at 180 s and sigma refitted on the oracle-overlap panel, the
> model has essentially **caught the book on calibration** (test Brier 0.1359
> against the book's 0.1353, a gap of 0.0006 where the previous pass measured
> 0.0239) — and it **still loses money in every one of the six arm/half cells,
> every gate passing, every CI below zero.** Two results reverse Part II: the
> CALIBRATED arm is now *worse* than the baseline out of sample, because an
> unconditional sigma table cannot cross the 2.28x volatility regime change
> inside this five-day window; and the "better calibration, bigger loss"
> pattern is confirmed in BOTH directions on the same run — the test half's
> smaller loss is the model trading 13 % less because it got more overconfident,
> not the model getting better.

> **PART II — superseded but not withdrawn.** On the corrected BTC/USD panel:
> **fixing sigma DOES fix the calibration** (test Brier 0.1528 -> 0.1398 against
> the book's 0.1289; saturation 27 % -> 11 % and now settling 0.995 correctly;
> the repaired curves sit on the diagonal), and it **does NOT change the sign of
> the PnL** — the loss grows from -$5.85 to -$10.37 per market, because a
> correctly wide sigma quotes nearer 0.5 and fills 39 % more at an edge that
> never covered fees. Part I below is the earlier pass on the UNCORRECTED panel
> and is kept verbatim for the before/after; its deferral is resolved in II.3
> and two of its conclusions are withdrawn in II.2 and II.4.

# PART I — the uncorrected panel (superseded, kept for comparison)

**Sigma really is far too small — that part of the diagnosis is confirmed by
measurement, not assumed. Fixing it removes the saturation (31 % of test-day
observations pinned at p >= 0.999 -> 4 %) and improves Brier from 0.1651 to
0.1550, but leaves the model nowhere near the book's 0.1289 and does NOT change
the sign of the PnL: the loss roughly doubles, from -$4.16 to -$8.18 per market.
The residual is not a scale error at all. It is a persistent LEVEL bias, and
that bias is a defect in the spot panel — built from Binance BTC/USDT while the
market settles on Chainlink BTC/USD. Every calibration and PnL number here is
measured on that uncorrected panel, so it cannot separate "sigma too small" from
"level biased", and the verdict on whether fixing sigma fixes calibration is
DEFERRED until the corrected panel exists.**

---

## 0. The data caveat that governs everything below

`data/spot_5m_100ms.parquet` was built from `bn_spot_mid` (BTC/**USDT**). The
market settles on Chainlink's BTC/**USD** 60 s TWAP. The venue capture carries a
`usdt_basis` column that the build did not apply. The harness owner measured the
gap at **+43.70 USD, sd 16.63** over 81,278 overlapping seconds on 2026-08-20,
falling to +5.67 (sd 8.02) once `usdt_basis` is subtracted.

**This investigation measured the same thing independently, from the other
end**, and agrees to within a dollar. Taking the model's own forecast error
`r = ln(A_T / s_t)` in the last 5 seconds of each market — where `fair.py`'s
`E[A]` is almost entirely the realised venue TWAP, so `r` is essentially the
venue-to-Chainlink basis:

| day | mean r (bp) | sd (bp) | fraction negative | mean, USD |
|---|---|---|---|---|
| 2026-08-19 | -6.76 | 2.46 | **1.000** | -44.70 |
| 2026-08-20 | -6.14 | 2.19 | **1.000** | **-43.82** |
| 2026-08-21 | -3.89 | 1.25 | **1.000** | -29.80 |
| 2026-08-22 | -1.65 | 0.70 | 0.996 | -12.74 |
| 2026-08-23 | -1.72 | 0.58 | 0.994 | -13.24 |
| 2026-08-24 | -1.80 | 1.11 | 0.945 | -14.06 |

08-20 reads -43.82 USD against the owner's +43.70 (opposite sign convention,
same quantity) — a $0.12 agreement between two independent measurements on
different data paths. The settling value is BELOW the model's forecast on
**827 of 827** fit-day markets. That is not vol. No vol block can fix it.

It is also **strongly non-stationary**: the basis is 3.6x larger on the fit half
than on the test half. This is a first-order confound for exactly the fit/test
split this investigation was asked to run, and it is why the fitted sigma does
not transfer (section 3).

## 1. The diagnostic: model sigma against realised movement

`fit_sigma.py`. For 64 time-to-expiry buckets, compare `vol_baseline.py`'s
sigma_T against the realised scale of `r = ln(A_T / s_t)`, where `A_T` is the
settling value and `s_t` is `fair.py`'s `E[A_T | F_t]` — the same level
`f.standardise` divides by. 99,347 observations over 1,553 markets. See
`sigma_vs_realised.png`.

**The model's sigma is below the realised scale at every single tau.** Ratio
`model / realised`, fit days:

| tau (s) | 0.5 | 5 | 30 | 60 | 90 | 150 | 210 | 270 | 295 |
|---|---|---|---|---|---|---|---|---|---|
| ratio | 0.0003 | 0.009 | 0.130 | 0.328 | 0.435 | 0.476 | 0.430 | 0.285 | 0.121 |

It never exceeds **0.48**, and it collapses toward zero inside the last minute —
exactly where `tau_eff` is most aggressive. That is the saturation mechanism the
diagnosis proposed, and it is real.

Three further findings from the same fit:

* **`tau_eff`'s functional form is wrong, not just its window.** Refitting the
  baseline's own `k * sqrt(tau_eff(tau; w))` with `w` free drives `w` to the
  bottom of the search range (0.5 s against the asserted 60 s). The realised
  term structure has a **floor**, not a decay: the data cannot be fitted by that
  family at any window.
* **The floor is the basis, not vol.** Decomposing the realised scale into
  |bias| and dispersion, at tau = 30 s the fit-day error is `|bias| = 5.67e-4`
  against `sd = 2.71e-4` — near expiry the error is *mostly bias*. Only past
  ~90 s does dispersion dominate.
* **The baseline's EWMA carries no conditional information.**
  `corr(log relative sigma, log relative |r|)` = **0.0043** over 52,883 fit
  rows. Whatever the EWMA is measuring, it is not next-market forecast error.
  Replacing it with an unconditional table therefore costs nothing measurable.

## 2. The three arms

Identical `fair.py`, `f.py`, `link.py` (copied unchanged from
`2026-09-09-normal-qq-eval`). Only `vol.py` differs. Quote parameters are the
unoptimised `e_p=0.01, rpl_p=0.0005, max_pos=50, shares=10` and were **not**
tuned. Fit on 2026-08-19/20/21, test on 2026-08-22/23/24; the sigma table is
fitted on the fit days only.

* **baseline** — `vol_baseline.py`, the EWMA + `tau_eff` block, unchanged.
* **calibrated** — `vol.py`, log-log interpolation over the empirical
  `sigma(tau)` table in `sigma_fit.json`, held flat below the shortest bucket.
* **flat** — `vol_flat.py`, `sigma = 1.203e-4 * sqrt(tau)`: the same weighted
  fit with the exponent pinned at 0.5. The level-only control.

### TEST DAYS (2026-08-22..24) — 3,630 observations, 726 markets. Lead with these.

| arm | Brier | book | p>=0.999 | its realised | worst decile gap | mean p | mean y | c/share | $/market | day-blocked CI | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.1651 | 0.1289 | **0.310** | 0.896 | -0.284 | 0.597 | 0.499 | -1.463 | -4.156 | [-4.375, -3.915] | all pass |
| calibrated | **0.1550** | 0.1289 | **0.040** | 0.993 | -0.239 | 0.546 | 0.499 | -2.174 | -8.175 | [-10.411, -6.230] | all pass |
| flat | 0.1581 | 0.1289 | 0.035 | 0.992 | -0.254 | 0.541 | 0.499 | -2.098 | -7.764 | [-10.020, -6.282] | all pass |

### FIT DAYS (2026-08-19..21) — 4,131 observations, 827 markets. In sample.

| arm | Brier | book | p>=0.999 | its realised | worst decile gap | mean p | mean y | c/share | $/market | day-blocked CI |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.2621 | 0.1367 | **0.482** | 0.704 | -0.382 | 0.733 | 0.512 | -1.366 | -2.677 | [-3.936, -1.493] |
| calibrated | 0.1871 | 0.1367 | 0.106 | 1.000 | -0.273 | 0.653 | 0.512 | -1.119 | -3.003 | [-4.327, -1.549] |
| flat | 0.1840 | 0.1367 | 0.095 | 1.000 | -0.236 | 0.639 | 0.512 | -1.250 | -4.029 | [-5.262, -2.717] |

All three gates (`sign_survives_periods`, `ci_excludes_zero`, `delete_top_10`)
pass on all six runs. They are passing on a **negative** mean: every arm loses
money robustly, not by accident of a few markets.

**The fit and test halves diverge sharply, and that IS the finding.** The
baseline's Brier is 0.2621 on the fit half against 0.1651 on the test half, and
its saturation 48 % against 31 % — while the book's Brier barely moves (0.1367
vs 0.1289). The model's miscalibration tracks the *panel bias magnitude* across
the two halves (-6.5 bp -> -1.7 bp), not anything about the model. Read the two
halves as two different data regimes, not as in-sample versus out-of-sample
skill.

## 3. What the arms actually say

**Sigma was the saturation mechanism.** Pinning falls 31 % -> 4 % on the test
half and 48 % -> 11 % on the fit half, and what saturation remains is now nearly
right: realised 0.993 among p >= 0.999, against 0.896 for the baseline. The
diagnosis was correct about the mechanism.

**It was not the calibration failure.** Brier improves by 0.010 on the test half
and stops 0.026 short of the book. The worst decile gap barely moves,
-0.284 -> -0.239. And `mean p - mean y` — the model's directional bias — stays
positive at **+4.7 pp** on the test half and **+14.1 pp** on the fit half, down
from +9.8 and +22.1 but nowhere near zero. A vol block can only dilute a level
bias toward 0.5; it cannot move it. `calibration_curves.png` shows the shape
directly: the baseline is too shallow (overconfident), the two repaired arms are
too **steep** (now under-confident), both sit right of the diagonal by the bias,
and the book sits on it.

**The problem is the LEVEL, not the term structure.** `calibrated` and `flat`
are within 0.003 Brier and 0.4 $/market of each other on both halves — flat is
even slightly *better* in sample. The measured term structure buys essentially
nothing over `k * sqrt(tau)`. So `tau_eff` is not the culprit its docstring made
it look like; a sigma that was simply too small everywhere is.

**The fitted sigma does not transfer.** `model / realised` on the test days runs
1.47 at tau = 270 s rising to **3.15** at tau = 30 s: the calibrated arm is
1.5-3x too WIDE out of sample. Two causes, both visible in section 0 — the
genuine dispersion fell (sd at tau = 270 s, 1.87e-3 -> 1.19e-3) and the basis
that inflated the fitted floor fell 3.6x. The second cause is an artefact and
will change when the panel is rebuilt.

**PnL gets worse, not better.** Both repaired arms roughly double the loss
(-$4.16 -> -$8.18 per market on test). A wider sigma quotes closer to 0.5, which
fills more (20,622 -> 27,299 fills) and therefore loses more per market against a
directional bias that is still there. Better calibration on an uncorrected level
is a more efficient way to lose.

## 4. Verdict — deferred, deliberately

Whether fixing sigma fixes calibration is **not answerable on this panel**. The
+$43 USDT-basis bias produces exactly the "too many confident UP calls" pattern
the sigma diagnosis was built to explain, so the two cannot be separated here.
What can be said now:

1. Sigma **is** materially too small (0.03 %-48 % of realised). Confirmed by
   direct measurement, and this measurement survives the panel defect: a level
   offset barely moves a scale estimate at the long-tau end.
2. Fixing it **does** remove the saturation at p = 1.000. Confirmed.
3. It does **not** close the gap to the book, and does **not** change the sign
   of the PnL. Confirmed on both halves.
4. The residual is a level bias whose size matches a known panel defect to
   within a dollar on the one day both were measured independently.

The next run should be: rebuild the panel with `usdt_basis` applied, re-run
`fit_sigma.py` — the fitted table is panel-conditional and **must** be refitted,
its short-tau floor being mostly basis, and should collapse from ~7.1 bp to
roughly 1 bp — then re-run these three arms. Only then does the headline
question have an answer. A further step, out of scope here because only `vol.py`
was allowed to change: any residual bias after the panel fix belongs in
`fair.py` as a drift term, not in `vol.py` as extra width.

## 5. Standing caveats

* The 100 ms replay grid flatters every result by roughly **+$0.13 per market**
  (measured twice on this substrate). Subtract it before believing any headline.
* **Maker fills are modelled, not measured.** No depth, no trade tape and no
  queue exist in this data; every maker number is conditional on the fill block.
* The **venue-to-panel clock offset is a configured 0.0 s**, an assumption and
  not a measurement. This repo bounds the plausible range at 0-74 ms.
* The sample is **6 days of one instrument** (2026-08-19..24; 2026-08-18 is
  absent to a source-archive schema defect). Three days per half is a thin basis
  for a day-blocked CI.
* `sigma_fit.json` is **specific to the uncorrected panel** and is invalid
  against the rebuilt one.

## 6. Reproducing

```
python fit_sigma.py          # measure, fit on 08-19..21, write sigma_fit.json + plot
python run.py                # three arms x two halves, results.json + calibration plot
```

`fit_sigma.py --rebuild` discards the cached observation panel. Run artefacts
live under `runs/` (gitignored); each arm's `manifest.json` pins the sha256 of
the four block files that priced it. `python -m pytest harness/tests -q` reports
168 passed, unchanged — no harness code was touched.

---

# PART II — the corrected BTC/USD panel (appended, nothing above overwritten)

Everything above was measured on `data/spot_5m_100ms.parquet`, which carried a
~+43 USD BTC/USDT-vs-BTC/USD level bias. That panel has been rebuilt as
`harness.paths.SPOT_USD` -> `data/spot_5m_100ms_usd.parquet` (`bn_spot_mid`
less the capture's `usdt_basis`; the oracle gap falls from +43.17, sd 16.36, to
+4.50, sd 7.44). Sigma was **refitted from scratch** on it — the old
`sigma_fit.json` is preserved as `sigma_fit_uncorrected.json` and was not
reused — and all three arms re-run. Same calendar split, same unoptimised
`QuoteParams(e_p=0.01, rpl_p=0.0005, max_pos=50, shares=10)`.

## II.0 The split is now clean

Short-tau (tte <= 5 s) forecast bias, the quantity that was a pure basis
artefact:

| | fit half | test half | fit / test |
|---|---|---|---|
| uncorrected panel | -6.51 bp | -1.79 bp | **3.64** |
| corrected panel | **-0.705 bp** | **-0.459 bp** | **1.535** |

The bias falls by roughly 9x in level and the fit/test asymmetry from 3.6x to
1.5x. It is not exactly 1, but both halves are now **under 1 bp** — about
$5-8 at these prices, i.e. inside the sd 7.44 residual the panel rebuild itself
reports. The remaining 1.5x is a real, small difference between the two halves,
not a data defect masquerading as one. **The fit/test comparison is finally
measuring the model.** The fraction of markets with a negative forecast error
falls from 1.000 (827 of 827) to 0.479-0.889 depending on tau, and the mean
bias changes sign across the window (-0.73 bp at tau = 30 s, +1.62 bp at 295 s)
— it now looks like noise, not an offset.

## II.1 Sigma against realised, corrected — the shortfall is real but smaller

`model / realised`, fit days, before and after (`sigma_vs_realised.png` is the
corrected version; `sigma_vs_realised_uncorrected.png` the old one):

| tau (s) | 0.5 | 5 | 30 | 60 | 90 | 150 | 210 | 270 | 295 |
|---|---|---|---|---|---|---|---|---|---|
| uncorrected | 0.000 | 0.009 | 0.130 | 0.328 | 0.435 | 0.476 | 0.430 | 0.285 | 0.121 |
| **corrected** | 0.002 | 0.065 | **0.603** | 0.665 | **0.678** | 0.612 | 0.497 | 0.317 | 0.133 |

**Yes, the bias was inflating realised moves — but sigma is still too small
everywhere.** Mid-window the shortfall roughly halves (at tau = 30 s, 0.13 ->
0.60; at 90 s, 0.44 -> 0.68), so about half the apparent shortfall was the
basis. The ratio still never reaches 1: it peaks at **0.68** and collapses to
0.065 by tau = 5 s. The fit-day and test-day ratio curves now lie on top of each
other, where before they were a factor of two apart.

Two shape findings survive and one reverses:

* **`tau_eff`'s form is still wrong.** Refitting `k * sqrt(tau_eff(tau; w))`
  with `w` free still drives `w` to the search floor (0.5 s vs the asserted
  60 s). The free power-law exponent moves from 0.174 to **0.602** — close to
  plain diffusion — so the corrected term structure is roughly `sqrt(tau)`,
  which is exactly what `tau_eff`'s aggressive late shrink is not.
* **The residual floor is the residual basis.** Realised scale still flattens
  near expiry at ~0.9 bp (~$7), matching the rebuild's own sd-7.44 residual
  rather than any volatility.
* **REVERSED — the EWMA does carry conditional information after all.**
  `corr(log relative sigma, log relative |r|)` goes from **0.0043 to 0.3805**
  over the same 52,883 fit rows. On the uncorrected panel a large constant bias
  swamped the per-market variation and made the EWMA look worthless. It is not.
  See II.4.

## II.2 The three arms, corrected panel — TEST days (3,630 obs, 726 markets)

Book Brier on identical rows: **0.1289**. Old (uncorrected) figures in
parentheses.

| arm | Brier | p>=0.999 | its realised | worst decile gap | mean p - mean y | c/share | $/market | day-blocked CI | gates |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.1528 (0.1651) | 0.266 (0.310) | 0.924 (0.896) | -0.205 (-0.284) | +0.040 (+0.098) | -1.668 (-1.463) | **-5.853** (-4.156) | [-7.107, -3.364] | all pass |
| calibrated | **0.1398** (0.1550) | 0.109 (0.040) | 0.995 (0.993) | -0.156 (-0.239) | **+0.020** (+0.047) | -2.128 (-2.174) | **-10.368** (-8.175) | [-11.795, -9.689] | all pass |
| flat | 0.1414 (0.1581) | 0.083 (0.035) | 0.987 (0.992) | **-0.140** (-0.254) | +0.021 (+0.042) | -2.242 (-2.098) | -11.466 (-7.764) | [-13.647, -9.773] | all pass |

FIT days (4,131 obs, 827 markets; book 0.1367): baseline 0.1573 / sat 0.274 /
-4.672 $/mkt; calibrated 0.1419 / 0.149 / -7.460; flat 0.1445 / 0.131 / -7.290.

**The two halves now agree.** Baseline Brier is 0.1573 (fit) against 0.1528
(test), where on the uncorrected panel it was 0.2621 against 0.1651. The
calibrated arm is 0.1419 in sample against 0.1398 out of sample — it
generalises, and the earlier "does not transfer, 1.5-3.1x too wide" finding was
an artefact of the basis being 3.6x larger on the fit half. Withdraw it.

## II.3 Verdict — the question I deferred

**Fixing sigma does fix the calibration. It does not change the sign of the
PnL; it makes the loss larger.**

*Calibration, fixed.* Test-day Brier 0.1528 -> **0.1398** against the book's
0.1289: the excess over the book falls from 0.024 to **0.011**. Saturation at
p >= 0.999 falls from 27 % to 11 % and what remains now settles at **0.995** in
the money — the pinned bucket is finally telling the truth. The worst decile gap
falls from -0.205 to -0.156 (flat: -0.140), and the directional bias
`mean p - mean y` halves, +0.040 -> +0.020. On `calibration_curves.png` the two
repaired arms sit **on the diagonal**, at least as close to it as the book,
while the baseline is still visibly shallow. The residual 0.011 Brier gap to the
book is now **resolution, not calibration**: the book is sharper — better
informed, not better calibrated.

*PnL, unchanged in sign and worse in size.* -$5.85 -> **-$10.37** per market
(flat: -$11.47), c/share -1.67 -> -2.13. All three gates pass on all six runs,
on a negative mean: these are robust losses, not noise. The mechanism is
mechanical — a correctly wide sigma quotes nearer 0.5, so fills rise 25,471 ->
35,370 (+39 %), and every extra fill is taken at an edge that does not cover
fees and adverse selection. **Better calibration is a more efficient way to
lose.** Calibration was never the binding constraint on profitability under this
quote policy; the model can be almost as well calibrated as the book and still
be a losing counterparty to it, because it is less *sharp* and pays the spread
and the fee for the privilege.

## II.4 Does the EWMA conditioning earn its place?

**Yes in principle, no as `vol_baseline.py` currently spends it — and neither of
the two repaired arms actually cashes it in.**

* The EWMA carries **real** conditional information: `corr = 0.3805` on the
  corrected panel, against 0.0043 on the biased one. My earlier "the EWMA
  measures nothing" conclusion is **withdrawn** — that was the artefact.
* But `vol_baseline.py` converts that information into a *worse* probability
  than a constant does: Brier 0.1528 against `flat`'s 0.1414, because its level
  is wrong by ~1.5x mid-window and by 15x near expiry. Good conditioning at the
  wrong level loses to no conditioning at the right one.
* `calibrated` now beats `flat` on both halves — 0.1398 vs 0.1414 on test,
  0.1419 vs 0.1445 on fit. Small but consistent, where on the uncorrected panel
  flat was marginally *better*. The measured term structure earns its keep once
  the basis is out; it did not before.
* **Honest recommendation:** neither shipped arm is the right block. Both are
  unconditional, and a corr of 0.38 is too much information to leave on the
  floor. Build next the EWMA's *shape* rescaled to the empirical *level* — keep
  the per-market conditioning, discard `tau_eff` and the annualise/de-annualise
  scaling, pin the level to `sigma_fit.json`'s term structure. Until that is
  measured, a constant sigma at the right level is the honest baseline, and it
  already beats what ships today.

## II.5 What did NOT change

Every standing caveat in section 5 holds unaltered: the 100 ms grid flatters by
~$0.13/market, maker fills are modelled and not measured, the venue-to-panel
clock offset is a configured 0.0 s (plausible band 0-74 ms), and this is six
days of one instrument with three days per half. The corrected panel removes a
level bias; it does not make any of those go away, and the PnL conclusion in
particular still rests on a modelled fill.

---

# PART III — warm-up on, sigma refitted, oracle-overlap panel (2026-09-09)

Four upstream defects were fixed before this pass, and every number in Parts I
and II predates at least one of them:

1. the spot panel was BTC/USDT, not BTC/USD (fixed before Part II);
2. `fair.py` now LEARNS the basis against the Chainlink oracle, gridded by
   receipt (`px_first_recv_ns`) rather than by the oracle's own stamp;
3. `s` is shifted onto the decision grid like every other Episode array;
4. Episodes carry a pre-open warm-up region, so stateful estimators burn in
   instead of restarting cold every 300 s.

This pass turns (4) on, raises the basis halflife to match it, refits sigma
from scratch, and re-runs the three arms. **Nothing here is tuned.** The quote
parameters are the same unoptimised `QuoteParams(e_p=0.01, rpl_p=0.0005,
max_pos=50, shares=10)` as every previous pass.

## III.0 What changed in the blocks

| | before | after |
|---|---|---|
| panel | `SPOT_USD`, 08-19..24 | `SPOT_ORACLE_WINDOW`, 08-17..21 |
| calendar split | fit 08-19..21 / test 08-22..24 | fit **08-17..18** / test **08-19..21** |
| warm-up | none | **900 s**, both fit and run |
| `fair.BASIS_HALFLIFE_S` | 60 s | **180 s** |
| `fair` gap sampling | fresh oracle, ANY mid | fresh oracle **and** mid ≤ 1 s old |
| `vol_baseline` RV EWMA | seeded at 0 every open | burnt in across the warm-up |
| `vol_baseline` MIN_UPDATES gate | in-window | unchanged, in-window |

The split is chosen so BOTH halves have oracle coverage, which the old one did
not: the RTDS capture stops at 2026-08-21 01:59 UTC, so on the 08-19..24 panel
the test half had no Chainlink line at all and `fair.py` — which returns NaN
without one — could not price a single test market. Scorable markets (spot,
oracle and a settlement) run 236 / 275 / 283 / 284 / 24 across 08-17..21, so
this cut is 511 fit against 591 test.

## III.1 The warm-up is used, not merely enabled — measured

Turning the flag on proves nothing by itself; a block that ignores the extra
arrays produces byte-identical output. `warmup_check.py` runs the same blocks
over the same 568 markets (2026-08-19..20) twice, warm and cold:

| | cold | warm |
|---|---|---|
| B_t at the open, identical to cold | — | **0.00 %** of 567 markets |
| \|shift in B at the open\| | — | mean **$5.32**, median $2.66, p90 $11.86 |
| cross-market sd of B at the open | $17.72 | **$14.89** |
| **\|step in B across a 300 s boundary\|** | mean **$5.64**, median $2.94 | mean **$0.18**, median **$0.09** |
| that step, reduction | — | **32.1x** |
| `vol_baseline` sigma, warm / cold | — | median **1.139** (p10 1.046, p90 1.379) |

The boundary step is the decisive one. Markets are back to back on a 300 s
grid, so B at the last bucket of market *k* and B at the first bucket of market
*k+1* are two estimates of the same quantity 100 ms apart. Cold, the second
throws away everything the first learned and the series sawtooths by $2.94 at
the median; warm, it crosses the seam and the step collapses to the size of one
EWM update. `warmup_basis.png` draws eight consecutive markets: the warm series
is continuous across every dotted boundary, the cold one jumps at each.

The vol row is the second half of the same story. `vol_baseline.py`'s
realised-variance EWMA has a 100 s halflife against a 300 s window, so seeding
it at zero every open reported a variance biased low for most of the episode it
was scored on: **the cold EWMA was 14 % too small at the median and 38 % too
small at the 90th percentile.**

### A defect the warm-up exposed, and the fix

Burning the basis in across the region initially made it *worse*, not better:
B arrived at some opens $25 off and decayed back over the following 15 minutes.
The cause is not the warm-up. **50 of the 1,389 market slots on
`SPOT_ORACLE_WINDOW` carry no venue spot at all**, and each one blanks the last
300 s of the warm-up region of the three markets that follow it. There the
panel carries the last mid forward for up to 295 s while the oracle keeps
printing at 1 Hz — so every `M - C` sampled in that stretch is *minus the BTC
move since the venue feed died*, not a basis. Measured on 2026-08-19 that
dragged B to +136 where the true basis was +44.

The block was guarding one side of the difference and not the other: it
required a fresh ORACLE print but accepted whatever mid the panel happened to
be carrying. `MAX_MID_AGE_MS = 1000` closes it. The spot age distribution is
bimodal — the fraction over 200 ms (0.25 % in-window, 1.17 % in warm-up) and
the fraction over 5 s agree to three decimals — so the threshold selects
outages and nothing else, and **on 60 cold in-window markets the guard changes
the output by exactly 0.0**: it is a warm-up-region fix that leaves every
previously measured in-window number untouched.

## III.2 Sigma refitted — the forecast error collapsed

`sigma_fit.json`, refitted from scratch on the new panel, split and warm-up.
The estimator is unchanged: `sqrt(pi/2) * mean|ln(A_T / s_t)|` per tte bucket,
fit days only.

| | previous (Part II) | **this pass** |
|---|---|---|
| fit rows / markets | 52,883 / 827 | 33,165 / **511** |
| near-expiry floor, sigma at tau = 0.5 s | 0.97 bp | **0.26 bp** |
| sigma at tau = 300 s | 15.8 bp | **5.96 bp** |
| short-tau bias, fit half | −0.705 bp | **−0.012 bp** (se 0.005) |
| short-tau bias, test half | −0.459 bp | **−0.020 bp** (se 0.008) |
| **fit / test bias ratio** | **1.535** | **0.600** |
| forecast bias across tau | −0.73 … +1.67 bp | **−0.04 … +0.57 bp** |
| free power-law exponent | 0.602 | **0.682** |
| free `tau_eff` window | 0.5 s (floor) | **0.5 s (floor)** |
| EWMA conditional corr | 0.3805 | 0.2186 |

**Read the ratio with the levels beside it.** 0.600 is not obviously "closer to
1" than 1.535 — on a log scale it is very slightly further. But both halves'
biases are now **35 to 60 times smaller in absolute terms**: −0.012 bp and
−0.020 bp are about **$0.08 and $0.13** on a $65,000 index. They remain
distinguishable from zero (t = −2.43 and −2.62 on 3,066 and 3,564 rows) and are
economically nil. The ratio has stopped being a diagnostic because it is now a
ratio of two numbers indistinguishable from zero; the honest statement is that
**the short-tau level bias the earlier passes were chasing is gone**, not that
the ratio improved.

The near-expiry floor is the same result seen from the other end. Part II
attributed a ~0.9 bp (~$7) floor to "the residual basis between the venue and
the oracle, which does not shrink with elapsed time". Learning that basis over
a 180 s halflife with 900 s of burn-in cuts the floor to **0.26 bp (~$1.7)**.
Most of what Part II called irreducible was the cold-start seeding error.

### The one thing this split cannot avoid

Realised movement rises monotonically across these five days:

| day | 08-17 | 08-18 | 08-19 | 08-20 | 08-21 |
|---|---|---|---|---|---|
| markets | 236 | 275 | 283 | 284 | 27 |
| realised scale at 300 s (bp) | 6.47 | 5.53 | 11.19 | 14.60 | 27.69 |

The fit half averages **5.96 bp** and the test half **13.57 bp** — the test
half is **2.28x more volatile**. Any calendar cut on this window puts the quiet
days in the fit half, and an UNCONDITIONAL sigma table carries none of that
across, so the calibrated arm enters the test half with a sigma roughly half
the size those days warrant. This is a property of a five-day window with a
trend in it, not of the fit, and it is the largest caveat on every test-half
number below.

## III.3 The three arms

Sample: markets carrying spot, a settlement oracle and a settlement.
`QuoteParams(e_p=0.01, rpl_p=0.0005, max_pos=50, shares=10)`, seeds (0, 1, 2),
warm-up 900 s. Calibration is five decision indices per market (tte 270, 210,
150, 90, 30 s), the book scored on identical rows.

### TEST days, 2026-08-19..21 — 591 markets, 2,951 observations. Lead with these.

Book Brier on identical rows: **0.1353**.

| arm | Brier | p≥0.999 | its realised | worst decile gap | fills | c/share | $/market | day-blocked CI | gates |
|---|---|---|---|---|---|---|---|---|---|
| baseline | **0.1359** | 0.205 | 0.974 | +0.129 | 23,079 | −1.148 | **−4.482** | [−7.487, −2.013] | all pass |
| calibrated | 0.1407 | 0.232 | 0.955 | −0.154 | 19,994 | −0.951 | **−3.216** | [−6.817, −1.017] | all pass |
| flat | 0.1445 | 0.240 | 0.951 | −0.158 | 19,274 | −0.999 | −3.257 | [−5.152, −1.217] | all pass |

### FIT days, 2026-08-17..18 — 511 markets, 2,551 observations. In sample for the calibrated arm.

Book Brier on identical rows: **0.1204**.

| arm | Brier | p≥0.999 | its realised | worst decile gap | fills | c/share | $/market | day-blocked CI | gates |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.1328 | 0.203 | 0.971 | −0.140 | 15,394 | −1.525 | **−4.595** | [−6.630, −2.848] | all pass |
| calibrated | **0.1315** | **0.123** | 0.994 | −0.138 | 20,777 | −2.106 | **−8.564** | [−10.041, −7.297] | all pass |
| flat | 0.1327 | 0.109 | 0.982 | +0.080 | 20,490 | −2.163 | −8.673 | [−9.315, −8.122] | all pass |

### The headline calibration result

**The model has closed on the book.** On the test half the baseline's Brier is
0.1359 against the book's 0.1353 — a gap of **0.0006**, where the previous pass
measured 0.1528 against 0.1289, a gap of **0.0239**. `calibration_curves.png`
shows all three model curves sitting essentially on top of the book's and close
to the diagonal, where the earlier passes had them bowed well below it.

It is not perfect and this report will not say it is. The residual S-shape is
still there and still in the overconfident direction: on the test half the
baseline predicts 0.957 in its second-highest decile and realises 0.841, and
**20.5 % of observations remain pinned at p ≥ 0.999 while settling in the money
97.4 % of the time.** That is a much smaller error than the 40 %-at-77.3 % of
the first evaluation, but it is the same sign, and a model that says "certain"
one time in five had better be right more than 97.4 % of the time.

### The calibrated arm LOSES to the baseline out of sample — and that is the vol regime

On the fit half the calibrated table does its job: saturation 0.203 → **0.123**,
Brier 0.1328 → **0.1315**, the best of the three. On the test half it goes the
other way: saturation 0.205 → **0.232**, Brier 0.1359 → **0.1407**, the worst
except for flat.

That reversal is the 2.28x volatility gap of III.2, not a defect in the fit. An
unconditional table fitted on the quiet half is simply too small on the loud
one, so on the test days the calibrated arm is MORE overconfident than the
baseline — whose EWMA at least conditions on each market's own realised
movement and therefore tracks the regime. **The lesson is not that the table is
wrong; it is that an unconditional sigma cannot survive a vol regime change,
and this five-day window contains one.** II.4's suggestion — keep the EWMA's
per-market conditioning and pin its LEVEL to the empirical table — is now
supported by a second, independent piece of evidence.

## III.4 The prediction: better calibration enlarges the loss. It HELD, in both directions.

The standing pattern is that every fix so far has improved calibration and made
the loss BIGGER, because a better-calibrated model quotes nearer 0.5, fills
more, and every extra fill is taken at an edge that never covered the fee. This
pass tests it twice, because the calibrated arm got better in sample and worse
out of sample:

| | saturation | Brier | fills | $/market |
|---|---|---|---|---|
| **FIT** baseline → calibrated | 0.203 → **0.123** (better) | 0.1328 → **0.1315** (better) | 15,394 → **20,777** (+35 %) | −4.595 → **−8.564** (loss +86 %) |
| **TEST** baseline → calibrated | 0.205 → **0.232** (worse) | 0.1359 → **0.1407** (worse) | 23,079 → **19,994** (−13 %) | −4.482 → **−3.216** (loss −28 %) |

Fills track calibration, and the loss tracks fills — in both directions, on the
same run, with the same blocks. Better calibration bought more fills and a
bigger loss; worse calibration bought fewer fills and a smaller one. **The
test-half PnL "improvement" is not the model getting better. It is the model
getting more overconfident and therefore trading less.** Anyone quoting
−$3.22 as progress has the causality backwards.

**Nothing here changes the sign.** All six arm/half cells lose money, all six
pass every gate (sign survives all calendar periods, day-blocked CI excludes
zero, deleting the ten best markets does not flip the sign), and all six CIs
lie entirely below zero. The best cell in the whole table is −$3.22/market
before the ~$0.13/market the 100 ms replay grid is known to flatter by.

## III.5 What did NOT change

Every standing caveat in sections 5 and II.5 holds. In addition, two new ones
belong to this pass:

* **The vol regime trend is the dominant confound on the test half.** Realised
  movement rises 2.28x from the fit half to the test half and no calendar
  split of these five days avoids it.
* **50 of 1,389 market slots have no venue spot at all**, and 237 of
  2026-08-21's 261 spot-covered markets have no oracle. The oracle-less ones
  are dropped rather than scored as $0.00 markets (see `run.py`); the
  spot-less ones blank part of three warm-up regions each, which is what
  `MAX_MID_AGE_MS` exists to survive.
