# vol-fixed — is sigma why the model is badly calibrated?

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
