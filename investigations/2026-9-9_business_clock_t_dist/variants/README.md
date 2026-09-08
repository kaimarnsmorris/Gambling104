# Variants

A variant is a file: `{name, base_params, overrides, notes}`. Reading one runs
nothing. `python run_variants.py --all` builds and scores every one of them.

Every override defaults to its off value, and with all defaults the model is
bit-identical to `fv-1.0.0` — `tests/test_golden.py` enforces that.

## The overrides

| key | sign convention | typical range | expected to move |
|---|---|---|---|
| `kappa_vol` | positive widens; vol × exp(κ) | −0.2 … +0.2 | QLIKE excess, `E[resid²/Var_Y]`, pooled log-loss |
| `kappa_vol_short` | as above, sub-minute only, weight `min(1, 60/D)` | −0.4 … +0.2 | the near-strike last-30 s cell, and little else |
| `shrink_w` | positive shrinks toward the unconditional level | 0 … 1 | the same cell as `kappa_vol_short`, by a different mechanism |
| `shrink_decay_s` | larger = the shrink reaches further out | 15 … 300 | how far up the horizon the shrink is felt |
| `rho_kernel` | `off` removes the variance inflation | — | `E[resid²/Var_Y]` upward when off |
| `eps_scale` | scales σ_ε only, so the ε **variance** scales as `eps_scale²`; the conditional mean is untouched (see the note below); at `eps_scale = 0` the ε term — mean and variance — is switched off by the sigma gate (R1), and **nothing else is**: `var_basis` answers to `basis_tracker` alone | 0 … 2 | Var_Y in the last ~10 s; the near-strike cell |
| `eps_condition` | false forgets the last observed residual | — | `eps_bar`, and log-loss near expiry |
| `basis_tracker` | `main` 60 s, `alt` 15 s, `off` no basis | — | the level terms, not the variance — but see the note below, it is not only the level |
| `information_set` | `prints_only` is the counterparty (R9) | — | everything; this is the lag edge |
| `reconstruct_in_transit` | false stops reconstructing stamped-unreceived prints | — | the carry, in the last few seconds |
| `kappa_tail` | **negative = fatter**; Var(z) held fixed | −1 … +1 | log-loss on confident quotes; not QLIKE — but see the note below, it is a no-op at the shortest horizons |
| `tail_scale` | multiplies μ and σ of the t together (R2) | 0.8 … 1.25 | the same as `kappa_vol`, at the other end |
| `tail_family` | `normal` drops the fitted t | — | near-expiry log-loss |
| `alpha_scale` | 0 = alpha off | 0 … 2 | `m_Y`; small effect on pooled log-loss |
| `alpha_scale_age` | scales the quote-age interaction only | 0 … 2 | `m_Y` on stale quotes |
| `alpha_cap_sd` | caps \|m_Y\| in settlement sd; null = uncapped | 0.5 … 3 | the tail of `m_Y`, not its centre |
| `w_spot` | blend weight; τ, δ re-read, never refitted | grid only | the whole print model — leave unset; see the trap below |
| `xi_cap_c` | the short-business-time cap; null = off | 1.5 … 4 | the forward curve during vol bursts |
| `temperature` | > 1 pulls toward a half; quoting layer only | 0.8 … 1.5 | `p_quoted` **only** — never `p_model` |

## Traps and gotchas

**`alpha_half`, `alpha_x2` and `no_alpha` are currently inert on the real-panel
export.** `export/build_export.py` calls `evaluate(..., book=None)`, so `m_Y` is
zero on every row of every variant's export, including baseline - there is no
order-book imbalance for `alpha_scale` to scale, so these three variants score
bit-for-bit identical to baseline no matter what `alpha_scale` is set to. The
precise reason: top-of-book imbalance is `I = ln(ask_size / bid_size)` — the
sign `scripts/12_alpha.py` fitted (`"imb": np.log(av[j] / bv[j])`, ask volume
over bid volume) and the sign `fvmodel/alpha.py` documents, which is why the
saturated `beta0` is **negative** (−0.19 bp per unit of I: heavier size resting
on the ask predicts a lower price). It needs SIZES, and the Polymarket 5 m book
panel this export reads
(`backtesting_5m/data/book_5m_100ms.parquet`) carries no size column at all -
`backtesting_5m/data/README.md` says so explicitly (L1 price only). This is not
a permanent limit: the venue L1 source the harness uses elsewhere
(`stream_venue_l1`) DOES carry sizes (`bn_spot_bid_sz`/`ask_sz` and friends), so
these three variants become meaningful the moment a book feed with sizes is
wired into `build_export`'s `evaluate()` call instead of `book=None`. Until
then, a null result from any of the three is silence about missing plumbing,
not evidence the fitted alpha term is worthless - do not delete these variants
on the strength of that null result.

**`eps_scale` moves the variance, not the mean.** `eps_bar = c * eps_last`, and
`c = w @ rho_at(ages)` depends only on the fitted autocorrelation - it never
sees sigma. `eps_conditional` is always called with `sigma=1.0`, and the actual
variance is `q_unit * sigma_at(v)**2` computed afterward, so `eps_scale` reaches
only that second factor: `eps_x2`'s variance is ×4, not ×2, and its mean
coefficient is bit-identical to baseline's. The only way `eps_scale` touches the
mean channel is at `eps_scale = 0`, where `model.eps.sigma_bp == 0.0` gates the
ε block off in `engine.py` - mean and variance both, because the term never
runs, not because the scale reached the mean. If you want to test the
conditional-mean correction itself, use `eps_condition: false` instead - that is
the knob for that question.

The ε gate stops at ε. `var_basis` — the variance of the slow basis's own drift
across the settlement window — is a different process with its own selector, and
it sits *outside* the sigma gate in both pricing paths. It used to sit inside
it, which made `eps_x0` silently switch off two channels while claiming one:
measured at n = 60, baseline `var_basis` = 4.011e−11 and `eps_x0`'s was exactly
0.0, about 13% of that variant's whole Var_Y reduction. `basis_tracker: off` is
the knob that removes the basis channel.

**`w_spot: 0.6` is not baseline.** The per-`w` table is the coarse grid, so it
gives τ = 0.7872 where the shipped fine-refined value is τ = 0.8447. Leave
`w_spot` unset to get the shipped filter. (Spec ruling R11.)

**`kappa_tail` is a no-op exactly where the tail matters most.** The floor is
`nu' = max(2 + (nu-2)*exp(kappa_tail), 2.5)`, and 3 of the 9 `chainlink_twap60`
z-bins are already pinned at that 2.5 floor (`nu = [2.60, 2.5, 2.5, 2.5, 2.61,
3.56, 4.64, 5.03, 5.64]`). `fat_tails` and `thin_tails` therefore cannot move the
shape at the shortest horizons — those bins are already at the floor before the
override is applied. Any effect you see there is `tail_scale` or `tail_family`
territory, not `kappa_tail`'s.

**`basis_tracker` moves the LEVEL channel only, and less than an earlier
draft of this file claimed.** Selecting `alt` necessarily selects
`eps_alt = d_lvl − b_alt` alongside `b_alt` — pairing one tracker's level with
the other's residual would double-count (spec ruling 13) — so the conditional
mean `eps_bar` moves with the tracked level. Nothing else does. Measured on the
shipped selection-window export, `basis_alt` against `baseline`:

| column | moves? |
|---|---|
| `eps_bar` | yes, max \|Δ\| 6.6e−5 |
| `p_ref`, `s`, `sigma` | yes — but only because `p_ref = exp(x_t + b_t)` carries `b_t`; `sigma = sqrt(Var_Y)·p_ref·ω` inherits it as a scale factor |
| `carry`, `var_eps`, `var_basis`, `m_Y` | **bit-for-bit identical** |
| `Var_Y` itself | identical to 1.1e−15 relative |

`var_eps = q_unit · sigma_at(v_loc)²` and `var_basis` both depend on the
component ages and the fitted curves, never on which tracker supplied the level,
so the variance channel cannot move; and the Chainlink carry is built from
`win.F` and `x_t`, which carry no basis at all.

**`temperature` never touches `p_model`.** It acts only in the quoting layer
(`link`) and is reported as `p_quoted` beside `p_model` precisely so it cannot
pollute a calibration statistic. If a temperature sweep ever shows identical
PnL at every temperature, that is the bug signature — the knob is not reaching
the quote. That was the case until the final fix round: `score/scorecard.py`
priced its trading proxy off `p_model` and nothing under `score/` read the
export's `p_quoted` column at all. The proxy now reads `p_quoted`; every
calibration metric (log-loss, Brier, reliability, QLIKE, the level ratios)
still runs on `p_model`, which is deliberate and is asserted by
`tests/test_scorecard.py::test_temperature_moves_the_pnl_and_leaves_calibration_bit_identical`.

**The near-expiry cell is thin.** Only about 6.2% of markets are still
undecided (0.02 < p < 0.98) in the final 30 s, and median settlement sd falls
from about $69 at open to about $1.40 near expiry. A variant that claims to
move "the near-strike last-30 s cell" is claiming to move a small, heavily
filtered sample, not a pooled statistic — read that cell's numbers with the
sample size in mind.

**The sweep name tag collides `kappa_vol` and `kappa_tail`.** `expand_sweep`
tags each combination with `k[:6]` per grid key, and `kappa_vol` and
`kappa_tail` both truncate to `kappa_`. Names stay unique only because the
values differ within the same tag; don't add a third `kappa_*` key to one grid
without checking `list({v["name"] for v in out})` by hand.

## Sweeps

`variants/sweeps/*.yaml` take `{name_prefix, max_variants, grid}` and expand to
the product. `name_prefix` is required and `max_variants` is checked **before**
expansion, because a four-key grid is easy to write and expensive to discover by
running it.

Sweep-generated variants are written to `variants/generated/` (git-ignored),
never to the top level of `variants/` — running a sweep must never change the
shipped variant set that `fvmodel.variants.list_variants()` reports.

## The runner and the holdout guard

    python run_variants.py --all
    python run_variants.py --variants baseline,rho_off
    python run_variants.py --sweep variants/sweeps/example.yaml

`--holdout` scores the reserved window (2026-08-26 .. 08-31) and refuses to run
without `--i-know` — it is for the one chosen variant, once, at the end, not for
iterating. A holdout run is appended to `runs/holdout_log.tsv` regardless of
`--no-score`, so every attempt (successful or not) leaves a trail. That file is
**committed** (spec §8.1) even though the rest of `runs/` is ignored; see this
folder's `.gitignore` for why the negation is shaped the way it is, and
`tests/test_variants.py::test_the_holdout_log_is_committable` for the guard.

## The harness handoff

Two items are open with the harness team and are written up in
`backtesting_5m/docs/fair-export-handoff.md`: the per-tick binding `link` needs
(the harness calls `link(z)` with no tick index, and this tail's parameters are
per-tick), and the export's timestamp contract, which `harness/core/episode.py`
requires in writing before `fair_is_causal=True` may be set. That document is
the written confirmation.
