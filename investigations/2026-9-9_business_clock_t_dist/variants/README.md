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
| `eps_scale` | scales the ε **process**, mean and sd (R1); 0 = off | 0 … 2 | Var_Y in the last ~10 s; the near-strike cell |
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

**`basis_tracker` moves more than the label suggests.** Selecting `alt`
necessarily selects `eps_alt = d_lvl - b_alt` alongside `b_alt`; pairing one
tracker's level with the other's residual would double-count. So `basis_alt`
moves `eps_bar` and `var_eps`, and the carry, together with the tracked level —
not the level in isolation.

**`temperature` never touches `p_model`.** It acts only in the quoting layer
(`link`) and is reported as `p_quoted` beside `p_model` precisely so it cannot
pollute a calibration statistic. If a temperature sweep ever shows identical
PnL at every temperature, that is the bug signature — the knob is not reaching
the quote.

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
`--no-score`, so every attempt (successful or not) leaves a trail.
