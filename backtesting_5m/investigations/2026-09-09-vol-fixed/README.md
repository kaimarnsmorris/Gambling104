# vol-fixed

Does the fair-value model's `sigma` explain its calibration failure? One
measurement and three arms.

## Provenance

`fair.py`, `f.py` and `link.py` were copied **unmodified** on 2026-09-09 from
`investigations/2026-09-09-normal-qq-eval/`. `vol_baseline.py` is that folder's
`vol.py`, also unmodified, kept under a different name so it can be run as a
control beside its replacements. Only the vol slot varies across arms; every
run's `manifest.json` pins the sha256 of the four files that actually priced
it.

## Layout

| file | what it is |
|---|---|
| `fit_sigma.py` | measures `vol_baseline.py`'s sigma against realised `ln(A_T/s_t)` by tau bucket, fits `sigma(tau)` on the FIT days only, writes `sigma_fit.json`, draws `sigma_vs_realised.png` |
| `sigma_fit.json` | the fitted table and the two parametric summaries, from the CORRECTED panel. Committed, so the blocks are reproducible without re-running the fit |
| `*_uncorrected.*` | the same artefacts from the first pass on the original BTC/USDT panel, kept so REPORT.md's before/after reproduces. Do not fit or run against them |
| `vol.py` | **calibrated** arm: log-log interpolation over the fitted table, held flat below the shortest bucket |
| `vol_flat.py` | **flat** arm: `k * sqrt(tau)`, the level-only control |
| `vol_baseline.py` | **baseline** arm: the EWMA + `tau_eff` block as it stands |
| `calibration.py` | unconditional model-vs-book calibration, plus saturation, per day set |
| `run.py` | materialises the three arms, scores calibration and backtests each on fit and test days, writes `results.json` and `calibration_curves.png` |

## Panel

Everything current runs against `harness.paths.SPOT_USD`
(`data/spot_5m_100ms_usd.parquet`), the corrected BTC/**USD** build. The
original `paths.SPOT` is BTC/**USDT** and carries a ~+43 USD level bias against
the Chainlink BTC/USD settlement; `fit_sigma.py --panel=usdt` still runs against
it, but only to reproduce Part I of the report. `sigma_fit.json` is
panel-conditional and must be refitted whenever the panel changes.

## Split

Fit on 2026-08-19/20/21. Test on 2026-08-22/23/24. Nothing fitted on the test
days; every headline leads with the test half. On the corrected panel the
short-tau bias ratio between the halves is 1.53 (it was 3.64), so the split is
now measuring the model rather than two different data defects.

## Slots left at harness defaults

`quote`, `execution`, `fill`, `fees`. This folder replaces the forecasting
model only. Quote parameters are the unoptimised
`e_p=0.01, rpl_p=0.0005, max_pos=50, shares=10`.

## Read `REPORT.md` first

It opens with a data caveat that governs how far any of these numbers can be
taken.
