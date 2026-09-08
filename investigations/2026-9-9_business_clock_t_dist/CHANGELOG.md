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

### The one deliberate behaviour change: `kappa_vol` moves to the forward curve

v2.1 applied `kappa_vol` by shifting `log v` on the **register bank**. This
version applies `2·kappa_vol` to `log ξ` on the **forward curve** (design spec
§4.2/§4.3). That is what makes the register bank variant-invariant, and so
cacheable once across every variant run instead of re-swept per variant.

The two are not the same map. The forward model's cumulative curve is
`exp(θ₀(z) + θ₁(z)·(log v − centre))`, so a shift in `log v` scales each second
by `exp(2·κ·θ₁(z))` with `θ₁` varying along the curve — not by the single factor
`exp(2·κ)`. At `kappa_vol = 0` the two are bit-identical, which is what lets the
golden test hold with `==`.

Measured at the unconditional register bank, `chainlink_twap60`, n = 300
(m_blk = 180, H = 120):

| `kappa_vol` | per-second forward variance | settlement return variance |
|---|---|---|
| 0 | bit-identical | bit-identical |
| +0.15 | median 1.66% apart, max 4.02% | **6.17%** apart (old 3.614e−07, new 3.404e−07) |
| −0.15 | — | **5.64%** apart |

Asserted, with these numbers, by
`tests/test_overrides_pertick.py::test_kappa_vol_on_the_forward_curve_differs_from_the_register_bank`
(design spec §10, test 2).

## Fix round 2 — 2026-09-09

The final whole-branch review. Model-affecting changes are the first two; none
touches the default path, and `tests/test_golden.py` still passes 15/15 with
`==`.

- **`curve.unconditional_xi` was misaligned by one element.** The shrink target's
  first increment was differenced against a prepended zero, so `xi_bar[0]` was
  the whole head integral (~74–161× the true increment at n = 300) instead of
  `g(H+1) − g(H)`. It bit only where `H > 0`, i.e. the first 121 of each market's
  301 rows. `unconditional_xi` now takes the business age of the second BEFORE
  the block; `forward_block`'s own block is the reference and equality with it is
  asserted bit-for-bit. Affects `shrink_05` only (the sole `shrink_w ≠ 0`
  variant), which has been rebuilt and re-scored.
- **`eps_scale = 0` no longer switches off `var_basis`.** The basis-drift variance
  is a different process from the fast ε residual and answers to `basis_tracker`;
  both pricing paths gated it on the ε sigma instead. Measured at n = 60,
  baseline `var_basis` = 4.011e−11 and `eps_x0`'s was 0.0 — about 13% of that
  variant's Var_Y reduction was the basis channel, not ε. `eps_x0` rebuilt and
  re-scored.
- The scorecard's trading proxy now prices off the export's `p_quoted` column
  rather than `p_model`, so `temperature` reaches a number. Every calibration
  metric still runs on `p_model`.
- `export/cache.py` narrows the register bank to float32 on the cold path as
  well as the warm one, so a rebuild reproduces the build it is rebuilding.
- `engine.evaluate` returns an empty `Cell` when its `keep` mask removes every
  expiry, instead of raising `IndexError`.
- `runs/holdout_log.tsv` is no longer gitignored (spec §8.1).
- Documentation: the top-of-book imbalance sign is `I = ln(ask_size / bid_size)`
  — the sign `scripts/12_alpha.py` fitted — corrected in `variants/README.md` and
  `report_variants.html`; `basis_alt`'s claimed effect on the carry and `var_eps`
  corrected (both are bit-for-bit identical to baseline).
- `backtesting_5m/docs/fair-export-handoff.md` records the export's causality
  contract in writing and the two open items with the harness team.
