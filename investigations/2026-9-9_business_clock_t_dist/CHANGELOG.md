# Changelog

## fv-1.0.0 — 2026-09-09

First frozen version. The model is `btc_volatility_clock_chainlink` as of
2026-09-09, vendored unchanged. Nothing is refitted here.

### Base

| | |
|---|---|
| forecaster | `btc_volatility_clock` v2.1.0 (`tables/params_v2_1_0.json`) |
| fit protocol | fitted through 2026-07-19; holdout 2026-07-20 .. 2026-08-31 |

### Fitted tables

| table | fitted by | window | report | what it carries |
|---|---|---|---|---|
| `print_model.json` | `scripts/10_print_model.py` | 2026-04-13 .. 2026-07-19, 07-07 masked | §4 | filter (w=0.6, tau=0.8447, delta=0.7, p_late=0.02), basis half-life 60 s, eps model, reconstruction |
| `kernels.json` | `scripts/11_kernels.py` | as above | §6 | activity-conditioned return autocorrelation, activity tercile cuts, input variance ratio |
| `alpha.json` | `scripts/12_alpha.py` | five book captures to 2026-04-02 | §10 | beta(dT) in bp per unit imbalance, quote-age interaction |
| `tails.json` | `scripts/13_eval.py` | as print_model | §11 | settlement tail (nu, mu, sigma) by log business time, per kind |
| `xi_cap.json` | `scripts/22_fit_xi_cap.py` | as print_model | §11.2 | short-business-time forward-curve cap. **Off by default** |
| `filter_by_w.json` | `tools/extract_filter_by_w.py` | as print_model | spec R11 | best (tau, delta, p_late) per blend weight, **coarse grid** |

### Known limitation

The order-book captures do not overlap the Chainlink history — they end
2026-04-02, Chainlink starts 2026-04-13. The location term is measured on the
perp and applied to a Chainlink settlement on the strength of the two being the
same price to within a basis point.

## ov-1.0.0 — 2026-09-09

The overrides block. Every key defaults to its off value; with all defaults the
model is bit-identical to fv-1.0.0, which `tests/test_golden.py` enforces.
