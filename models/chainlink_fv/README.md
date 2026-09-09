# chainlink_fv

The Chainlink fair-value model, as blocks. Three files, because the harness's own
defaults are already right for the rest.

| slot | here? | why |
|---|---|---|
| `fair` | yes | the fair settlement level, shifted by the tail's location |
| `vol` | yes | the settlement sd with the whole per-tick tail folded into it |
| `link` | yes | one fixed shape for the run: a Student-t, or a normal |
| `f` | **no** | the default is `(level − strike)/sigma`, which is exactly our convention |
| `quote`, `execution`, `fill`, `fees` | **no** | the venue, not the model |

The model itself — the v2.1 volatility clock, the Chainlink print model, the
settlement functional, the tails and the order-book term — is not here. It lives in
`investigations/2026-9-9_business_clock_t_dist/`, which writes a fair export per
variant. These blocks read that export and nothing else.

## The one modelling concession

The harness passes **two numbers per tick** into the quote algebra, `s[i]` and
`sigma[i]`, plus a `link(z)` that is a pure function with no tick index and no
episode. The model wants **three**: a fair level, a settlement standard deviation,
and a tail shape `(ν, μ, σ_t)` that varies with business time, ν running 2.5 to 5.6
across the nine fitted bins.

`tailfold.py` is where the third one goes. Writing `u = level − strike`, the model's
price curve is `F_t((u/σ + μ)/σ_t ; ν)` and the harness computes `F_t(u/S ; NU0)`.
Matching those at their quartiles gives

    s' = s + μ·σ                        the location, riding on the fair value
    S  = σ·σ_t·q₇₅(ν)/q₇₅(NU0)          the scale, carrying the shape mismatch

The location rides on `s` because `f` measures from the strike and cannot be told to
centre anywhere else.

**What it costs.** Exact wherever `ν = NU0`. Elsewhere, measured on 100k real export
rows, a median of **0.07 c** and a 95th percentile of **0.42 c** of probability in
the band where quoting happens, against a taker fee peaking at 1.75 c and a maker
rebate averaging 0.234 c. `tests/test_harness_model.py` holds that budget.

`NU0 = 3.483` is the minimax choice over the fitted bins. It is not the lowest
typical error available — ν₀ = 4.35 reaches 0.21 c at p95 — but it has the best
worst case, 0.82 c against 1.26 c, and on a venue where one bad fill costs more than
several good ones that is the trade worth making.

## Two things the fold cannot carry, and where they went instead

Both are per-**variant** constants rather than per-tick ones, so `link.py` reads them
once from the export's sidecar and applies them for the whole run.

- **`tail_family`.** `SettlementTail.prob_up` ignores `ν`, `μ` and `σ_t` entirely
  when the family is normal, while those columns stay in the export carrying their
  fitted Student-t values. Quartile-matching a normal onto the t link was tried and
  costs 1.64 c at p95 — four times the t fold — which would make the `normal_tail`
  variant measure the fold rather than the model. With a normal link the fold is the
  identity and the variant is exact.
- **`temperature`.** `p → sigmoid(logit(p)/T)` is neither a location nor a scale
  change, so no fold can express it.

## If the harness ever binds `link` per tick

Delete `tailfold.py`, hand the tail over whole, and the concession goes away. Nothing
else here changes.
