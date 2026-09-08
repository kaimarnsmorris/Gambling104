---
title: Consolidating the Chainlink fair-value model into harness blocks
date: 2026-09-09
status: design — approved in brainstorming, not yet implemented
source_brief: "Consolidation brief: override parameters and variant backtesting"
related:
  - ../../../../../backtesting_5m/docs/superpowers/specs/2026-09-09-backtesting-harness-design.md
  - "autoresearch/btc_volatility_clock_chainlink/README.md"
  - "autoresearch/btc_volatility_clock_chainlink/report/report_final.html"
---

# Fair-value consolidation — design

## The question this answers

**How does the Chainlink fair-value model become a set of files the 5 m
backtesting harness can run, such that a variant of the model is a data change
rather than a code change — and such that the default variant is provably the
model we have today?**

The model is not modified. Nothing here refits anything. Every fitted table
ships as-is; the work is packaging, one clearly-scoped relocation of
`kappa_vol`, the deletion of unreachable branches, and a set of knobs that all
default to no-ops.

## Why this is worth doing

The 5 m Polymarket up/down book settles on a **60 s Chainlink TWAP** from
2026-08-14 (`backtesting_5m/data/README.md`, "The era boundary"). That is
exactly `kind = "chainlink_twap60"`. The fair-value model and the venue are
pricing the same object, and the harness's own design already names the seam:
`s` arrives as data from `data/fair/<model_id>.parquet`, and blocks are files
resolved out of the investigation folder.

So the consolidation is not speculative infrastructure. It is the join that
makes the harness's `s` input exist at all.

## Scope

**In:** vendoring the model into this folder; one versioned config; the
`overrides` block and its application sites; moving `kappa_vol` to the forward
curve; deleting unreachable ablation branches; the export builder and its
caches; the four block files; the variant set and its sweep spec; a calibration
scorecard on the panel's real markets; the tests.

**Out:** any refit; any change to the vol model `btc_volatility_clock`; the
execution harness itself; live trading.

**Non-goals:** the brief's §3 `backtest(variant, quotes, outcomes)` on synthetic
Chainlink markets. Variant ranking is the execution harness's job. The
scorecard in §8 below is the interim substitute and is deliberately smaller.

---

## 1. Decisions taken in brainstorming

| decision | ruling |
|---|---|
| Which backtester | The 5 m execution harness. The brief's §3 synthetic-market backtester is dropped. |
| Where the code lives | Vendored into `investigations/2026-9-9_business_clock_t_dist/`. |
| Backtest window | 2026-08-14 → 2026-08-31 only. |
| Config | A new consolidated `model.json` here. `btc_volatility_clock/output/params.json` is copied in and never edited. |
| Acceptance | Model side, plus a calibration scorecard on real markets so variants can be ranked before the harness lands. |
| Vendor depth | Code and fitted tables committed; raw 1 s parquets referenced through one configurable path. |
| Holdout | Select on 08-14 → 08-25; reserve 08-26 → 08-31 behind `--holdout` + `--i-know`. |

### 1.1 The data gap, stated once

`chainlink_1s.parquet` runs to 2026-09-08 10:15 and covers the panel. The
model's *input* does not: `btcusdt_1s.parquet` ends 2026-08-31 23:59:59 and the
spot caches end with August. The model's `X` is a 60 % spot / 40 % perp blend,
so 2026-09-01 → 09-08 (~2,300 markets) is unpriceable. This is a data-build
task, not a modelling one, and is explicitly out of scope here. Every report
this design produces states the window it covers.

---

## 2. Layout

The harness resolves block slots **by filename at the investigation folder
root** (`backtesting_5m` spec §2.1: "Writing `link.py` in your investigation
folder *is* the override"). The four slot files therefore sit at the top level
and everything else lives in subpackages.

```
investigations/2026-9-9_business_clock_t_dist/
  fair.py  vol.py  f.py  link.py      the four block slots — thin readers
  model.json                          the consolidated, versioned config
  CHANGELOG.md                        every fitted table, and what produced it
  tables/                             committed, ~150 KB total
    params_v2_1_0.json                verbatim copy of the v2.1 params
    print_model.json  kernels.json  alpha.json  tails.json  xi_cap.json
    filter_by_w.json                  extracted per-w filter table (§5, w_spot)
  fvmodel/
    __init__.py  config.py  overrides.py
    base.py weights.py chainlink.py curve.py variance.py alpha.py tails.py
    fairvalue.py engine.py build.py
    _vendor/
      rvforecast/   config forward kernels regression seasonality streaming
                    distribution tzclock  (+ a trimmed __init__)
      clkit/        common.py state.py
  export/build_export.py
  score/scorecard.py  score/report_variants.py
  variants/  README.md  baseline.yaml … rho_off.yaml  sweeps/
  tests/
  cache/  runs/                       gitignored
```

> **Hazard.** The root directory is the harness's slot namespace. Do not create
> a root-level `quote.py`, `execution.py`, `fill.py` or `fees.py` here unless we
> intend to override those slots. A test asserts the root contains exactly the
> four intended slot filenames.

### 2.1 What gets vendored

`fv/` needs `clkit/{common,state}.py` and eight `rvforecast` modules:
`streaming` (which pulls `config`, `distribution`, `kernels`, `regression`,
`seasonality`), `forward`, and `seasonality` (which pulls `tzclock`). The
vendored `rvforecast/__init__.py` is trimmed to re-export only those, so
`pipeline`, `evalkit`, `data` and the rest are not dragged in.

Raw 1 s inputs are **not** copied. `model.json` carries a `data` block of paths
with the current OneDrive locations as defaults, overridable by environment
variable, and `config.py` fails loudly with the expected path when one is
missing.

---

## 3. `model.json`

One file, versioned, with its own provenance and the `overrides` block carrying
its own `version` and `source` as the brief requires.

```jsonc
{
  "version": "fv-1.0.0",
  "generated_at": "2026-09-09T…Z",
  "source": "btc_volatility_clock_chainlink @ <fingerprint>",
  "window": { "backtest_t0": 1786665600, "backtest_t1": 1788220799 },
  "base": {
    "params": "tables/params_v2_1_0.json",
    "params_version": "2.1.0",
    "params_sha256": "…"
  },
  "tables": {
    "print_model": { "file": "tables/print_model.json", "sha256": "…",
                     "fitted_by": "scripts/10_print_model.py",
                     "report": "report_final.html §4" },
    "kernels":     { … "scripts/11_kernels.py",  "§6" },
    "alpha":       { … "scripts/12_alpha.py",    "§10" },
    "tails":       { … "scripts/13_eval.py",     "§11" },
    "xi_cap":      { … "scripts/22_fit_xi_cap.py", "§11.2" },
    "filter_by_w": { … "extracted from cache/print_model.npz:surface_train" }
  },
  "defaults": { "eps_fit": "train", "input_mode": "blend" },
  "overrides": { "version": "ov-1.0.0", "source": "consolidation brief 2026-09-09",
                 "kappa_vol": 0.0, … }
}
```

`CHANGELOG.md` restates the `tables` block in prose: for each fitted artefact,
the script that produced it, the window it was fitted on, and the report section
that justifies it. That is brief §4.1.

Loading is one function, `fvmodel.config.load()`, which verifies every
`sha256`, resolves data paths, and returns `(FairValueModel, Overrides,
Provenance)`. A mismatched hash is a hard failure, not a warning.

---

## 4. Overrides

One frozen `Overrides` dataclass in `fvmodel/overrides.py`. Every value defaults
to its off value; any non-default is logged at startup and carried in the
provenance block of every export and every result row.

Application is in two layers, because six application points cannot honestly be
one call site:

**Build time — `apply_overrides(model, ov) -> FairValueModel`.** Everything
statically resolvable is folded into a new model object: `w_spot`, `xi_cap_c`,
`basis_tracker`, `alpha_scale`, `alpha_scale_age`, `eps_scale`, `rho_kernel`,
`kappa_tail`, `tail_scale`, `tail_family`. The tail transforms are elementwise
on the stored `(ν, μ, σ)` grids, so they bake here rather than per tick.

**Per tick — three pure transforms, in the same module, called by both the
scalar and the vectorised path.** This is what stops the two implementations
drifting:

- `xi_adjust(ov, xi, dT_k_business_s, xi_bar)` — `kappa_vol`,
  `kappa_vol_short`, `shrink_w`, `shrink_decay_s`
- `cap_m_Y(ov, m_Y, var_y)` — `alpha_cap_sd`
- `quoted_prob(ov, p)` — `temperature`

`eps_condition` and `reconstruct_in_transit` remain boolean fields read at their
existing branch points; `information_set` (see §4.3) likewise.

### 4.1 The table, with application points

| key | default | effect | applied |
|---|---|---|---|
| `kappa_vol` | 0.0 | `log ξ_k += 2·kappa_vol` | forward curve, after the v2.1 link, before the ρ kernel |
| `kappa_vol_short` | 0.0 | same, weighted `min(1, 60/D_k)` | same site |
| `shrink_w`, `shrink_decay_s` | 0.0, 60 | `log ξ'_k = (1−w_k) log ξ_k + w_k log ξ̄`, `w_k = shrink_w·exp(−D_k/decay)` | same site, after `kappa_vol` |
| `rho_kernel` | `conditional` | `conditional` / `unconditional` / `off` | Var_Y assembly |
| `eps_scale` | 1.0 | scales σ_ε only, so Var_Y's ε contribution scales as `eps_scale²`; ε̄'s coefficient `c` is untouched; `eps_scale = 0` gates the whole term off (R1, amended) | Var_Y assembly; `eps_scale = 0` also removes the ε̄ term via the gate |
| `eps_condition` | true | condition ε̄ on the last observed ε | ε term |
| `basis_tracker` | `main` | `main` (60 s) / `alt` (15 s) / `off` (R8) | print model |
| `information_set` | `full` | `full` / `prints_only` (R9) | print model, quote origin |
| `reconstruct_in_transit` | true | off ⇒ stamped-but-unreceived prints are not reconstructed | reconstruction |
| `kappa_tail` | 0.0 | `ν' = 2 + (ν−2)·exp(κ)`, σ rescaled to hold Var(z) | tail, after lookup |
| `tail_scale` | 1.0 | scales μ **and** σ of the t (R2) | tail |
| `tail_family` | `t` | `t` / `normal` | tail |
| `alpha_scale` | 1.0 | scales β(ΔT) and the age interaction together | location term |
| `alpha_scale_age` | 1.0 | scales the age interaction only | location term |
| `alpha_cap_sd` | null | caps `\|m_Y\|` at this many settlement sd | location term |
| `w_spot` | null | blend weight; (τ, δ, p_late) re-read from `filter_by_w` (R11) | input blend |
| `xi_cap_c` | null | the existing short-business-time cap | forward curve |
| `temperature` | 1.0 | `p ← σ(logit(p)/T)`, reported as `p_quoted` beside `p_model` | after P(up) |

### 4.2 Rulings where the brief is underdetermined

- **R1 (amended by Ruling 17) — `eps_scale` scales σ_ε only, not the carried
  `eps_last`.** This supersedes the original R1 text above, which called for
  `eps_scale` to multiply both σ_ε and the carried `eps_last` - that mechanism
  was never implemented, and the implemented behaviour is kept as the intended
  design, not a bug to fix. What is actually true: `eps_bar = c · eps_last`
  with `c = w · ρ(ages)` a function of the fitted autocorrelation alone, so
  `eps_scale` cannot reach `c` or `eps_last` at any value other than zero — the
  ε variance contribution scales as `eps_scale²` and the conditional mean is
  untouched. `eps_x0` still means "ε term off, mean and variance both", but for
  a different reason than the original text gave: at `eps_scale = 0`,
  `model.eps.sigma_bp == 0.0` gates the whole ε block off in `engine.py`
  (mean and variance together), rather than the scale having reached the mean
  channel directly. Away from zero this is the better-posed knob: it asks a
  single clean question ("is the fitted ε variance right?") instead of
  compounding a location shift with a variance change, and it leaves the
  mean/conditioning question to `eps_condition`, which already tests exactly
  that ("is the conditional-mean correction right?"). Implementing the
  original R1 would have muddled two orthogonal questions into one knob.
  Pinned by `tests/test_overrides_buildtime.py::test_eps_scale_is_variance_only_away_from_zero`.
- **R2 — `tail_scale` scales μ together with σ.** Otherwise the brief's own
  orthogonality assertion is false: `(y*/sd/e^c − μ)/σ` equals
  `(y*/sd − μ)/(σ e^c)` only when `μ = 0`. Scaling the standardised variable as
  a whole is also the more natural reading of "width in z-space".
- **R3 — the orthogonality test runs on `perp_single`.** The Jensen term makes
  `y*` depend on `Var_Y`, which breaks the `kappa_vol` ≡ `tail_scale`
  equivalence for the averaged kinds. `perp_single` has no Jensen term and no ε,
  which is exactly the "ε term off" condition the brief asks for.
- **R4 — "one `fair_value` entry point" is one public dispatcher.** The
  vectorised `fv.engine.evaluate` is worth about three orders of magnitude on
  the export and `tests/test_engine_matches_fairvalue.py` is what makes the
  duplication safe. One public `fair_value(...)` dispatches to the scalar or
  batch implementation; the equality test is extended to every override rather
  than the batch path being deleted.
- **R5 — `apply_overrides` is one build-time function plus one shared per-tick
  transform module.** See §4 above.
- **R6 — `ξ̄`** (the shrink target) is the forward curve evaluated from
  `params["registers"]["initial_value"]` — the unconditional bank at the same
  business time.
- **R7 — `D_k_seconds`** is business age in seconds: business days × 86400.
- **R8 — `basis_tracker` gains `"off"`**, so brief §4.3 does not delete the
  basis ablation (`Switches.use_basis`) for want of a selector.
- **R9 — `information_set` is kept as its own key.** The code's
  `market_observable` switch is *stronger* than the brief's `market_observable`
  variant: besides dropping reconstruction and alpha it shifts the quote origin
  back by the receive lag and prices off the last received print. That is the
  honest "no exchange feed" counterparty and is worth keeping; the variant then
  means what the code means.
- **R10 — brief §4.3 deletes:** `eps_mode="iid"`, `tail_mode="v2_twap"`,
  `use_blend`, `composed_weights`, `jensen`. None is selectable and none is
  production. A test asserts every remaining branch is reachable from an
  override key.
- **R11 — `w_spot` at its default `null` uses the shipped filter.** The per-`w`
  table is the *coarse* grid, so `w_spot: 0.6` gives τ = 0.7872 against the
  shipped fine-refined τ = 0.8447. `w_spot: 0.6` is therefore **not** baseline,
  and `variants/README.md` says so in as many words.

### 4.3 Moving `kappa_vol` (brief §4.2)

Today `kappa_vol` is applied to the register bank (`logv += 2κ` before
`forward_block`). It moves to the forward curve output (`log ξ_k += 2κ`). At
`κ = 0` both are no-ops, so the default is untouched; at `κ ≠ 0` the two differ,
because the register→curve link is non-linear, and that difference is expected
and documented.

The relocation buys the thing that makes this whole design fast: **the register
bank becomes variant-invariant.** See §6.

---

## 5. `filter_by_w.json`

`fit_filter` returns the whole `(w, τ, δ, p_late, rmse)` surface but
`print_model.json` keeps only the winner. The surface survives in
`cache/print_model.npz:surface_train` (17,238 rows). A one-off extraction step
reduces it to the 13-row argmin-per-`w` table and writes
`tables/filter_by_w.json`, which is small, readable and committed. The `.npz`
is not vendored.

---

## 6. Caching, and the warm start

`BatchV2.registers()` always seeds `run_bank` from
`params["registers"]["initial_value"]` at window index 0; there is no way to
resume a bank mid-stream. The burn-in must therefore be *in* the window. Perp
data starts 2025-09-01, so for a panel starting 2026-08-14 the 14-day burn-in is
free — no market is lost to it.

What must not happen is paying that sweep once per variant. Because §4.3 makes
the register bank variant-invariant, it is swept once and cached:

| layer | contents | invalidated by |
|---|---|---|
| L0 | aligned 1 s frame over `[t0 − 14 d, t1]` | window |
| L1 | `logv` (float32) and `act` at every quote second | `params_sha256`, window |
| L2 | print-model series `F, Fd, b, b_alt, eps_step, lxf` | `w_spot`, `basis_tracker`, filter constants |
| L3 | `runs/fair/<variant>.parquet` | any override |

L1 is the warm start. It is ~50 MB at float32 for the window and is the only
expensive, shared computation. Each variant then pays only `forward_block` →
variance → tail: an estimated 1–3 minutes for 300 horizons × ~4,900 markets,
dominated by the numba banded variance. `w_spot` and `basis_tracker` are the two
overrides that invalidate L2 and are correspondingly slower; the runner says so
when it starts.

---

## 7. The export and the block mapping

### 7.1 Algebra

`y*` is affine in the strike, so the export can be **strike-free** and the
blocks can do the strike work — which is precisely the split the harness's slot
contract asks for. Writing `K₀` for the strike at which `y* = 0`:

```
s     = p_ref · ( A_R + ω · ( 1 + ½·Var_Y + d̄ + m_Y + ε̄ ) )
sigma = p_ref · ω · sqrt(Var_Y)
```

then `y*(K) = (K − s) / (p_ref · ω)` and `y*/sd_Y = (K − s)/sigma = −z`, so

```
f(level)  = (level − strike) / sigma          exactly the harness's contract
link(z)   = 1 − F_t(−z; μ, σ_t) = F_t((z + μ)/σ_t)     by t-symmetry
```

`s` is the harness's `E[A]` and carries the units the spec assigns it (USD of
BTC). No approximation is introduced by the split.

### 7.2 Schema

`runs/fair/<variant>.parquet`, 301 rows per market (§7.3), ~1.5 M rows:

`market_id, open_ts, t_s, s, sigma, nu, mu, sigma_t, p_model, p_quoted, omega,
n_known, n_transit, m_Y, eps_bar, carry, var_eps, var_basis, z_business, p_ref,
ok`

plus a provenance footer: `model.json` version, overrides version, the full
non-default override set, and the window.

`p_model` / `p_quoted` are at the market's *real* strike and exist for the
scorecard; the blocks ignore them and go through `f`/`link`.

Rows the model cannot price (`ok = False`) carry NaN `s` and `sigma`. The
harness already treats absent `s` as absent — nothing is forward-filled — so no
special handling is needed, but the scorecard reports the excluded fraction
rather than dropping it silently.

### 7.3 Causality

The model's state at instant `t` is the price at instant `t` (per `fv.base`:
a 1 s bar stamped `t−1` covers `[t−1, t)` and its close is the price at `t`),
and Chainlink's 2 s receive lag is already inside the print filter. So the value
computed at `t` is knowable at `t` up to sub-second publication delay.

The harness's own convention is stricter, and this design adopts it rather than
arguing with it: a value carrying information through instant `t` becomes usable
one 100 ms bucket later, at

```
t_ms = (t − open_ts) · 1000 + 100
```

and holds until the next second's value supersedes it. To give bucket
`t_ms = 0` a value, the export runs `t` from `open_ts − 1` to `open_ts + 299`
inclusive — 301 rows per market — so every one of the 3,000 buckets is covered
and none is covered by a value from its own instant or later.

This is one function with one test, mirroring the harness spec's §1.4. The test
asserts that for every bucket `i`, the `t` behind its value satisfies
`(t − open_ts)·1000 + 100 <= i·100`.

### 7.4 The four blocks

`fair.py` and `vol.py` are `SignalBlock`s: `precompute(ep) -> s[N]` and
`-> sigma[N]`, read straight off the joined export. `f.py` is the
standardisation above. `link.py` is the fitted tail.

**One dependency on the harness team.** `link` is contracted as a pure
`link(z) -> p`, but `(ν, μ, σ_t)` are a function of business time left and are
therefore per-tick. `link` needs the same per-tick binding `f` already has. This
is a one-line contract extension and should be raised before the harness
freezes its block API.

The variant a run uses is selected by `FV_VARIANT` (default `baseline`); the
blocks share a `_fvexport.py` loader, whose name is deliberately not a slot.

---

## 8. Scorecard

Interim, until the execution harness lands, and deliberately smaller than the
brief's §3. It scores the export against the panel's **real** markets: real
strikes from `strikes_5m.parquet`, and realised settlement computed as the 60 s
Chainlink TWAP ending at `open_ts + 300` from `chainlink_1s.parquet` — which
also cross-checks the venue's own resolution.

- log-loss and Brier at the real strike, and on a strike grid at
  `{±0.5, ±1.0}` baseline sd (the only informative score at long tte, and
  placed on the *baseline's* sd so variants answer the same question)
- reliability by (tte bucket, moneyness bucket)
- `E[resid²/Var_Y]` and QLIKE excess by tte
- the near-strike last-30 s cell (`|moneyness| < 1σ`, tte ≤ 30 s) on its own line
- a trading proxy against the **real book mid** as `p_market`: buy the side
  where `|p_model − p_market| > edge` for edge ∈ {0.02, 0.05, 0.10}, sized
  ∝ (p_model − p_market), net of the verified fee model. Mean, sd, Sharpe-like
  ratio, and the fraction of PnL from the last 30 s.
- a one-line delta against `baseline` for every metric

`score/report_variants.py` emits `report_variants.html`: baseline plus every
shipped variant on the selection window, sorted by near-strike last-30 s
log-loss with pooled log-loss beside it, and a paragraph per variant on whether
the result matches its `variants/README.md` expectation.

### 8.1 Holdout

Selection window **2026-08-14 → 2026-08-25**; holdout **2026-08-26 →
2026-08-31**. `--holdout` is refused unless `--i-know` is also given, and every
holdout run appends variant, timestamp, git sha and metrics to a committed
`runs/holdout_log.tsv`. The report is generated on the selection window only.

---

## 9. Variants

`variants/<name>.yaml`: `{name, base_params: "model.json@fv-1.0.0", overrides,
notes}`. Shipped set, exactly as the brief specifies:

`baseline`; `market_observable` (`information_set: prints_only`,
`reconstruct_in_transit: false`, `alpha_scale: 0`); `no_alpha`, `alpha_x2`,
`alpha_half`; `vol_plus10` / `vol_minus10` (`kappa_vol ±0.095`);
`short_vol_minus20` (`kappa_vol_short −0.22`); `shrink_05` (`shrink_w 0.5`,
decay 60); `fat_tails` / `thin_tails` (`kappa_tail −0.5 / +0.5`), `normal_tail`;
`eps_x0`, `eps_x2`; `basis_alt`; `rho_off`.

A sweep spec is `{name_prefix, keys, grids, max_variants}`, expanded to the
product; `name_prefix` is required and the cap is enforced before anything runs.

`variants/README.md` gives, per override: sign convention, typical range, and
which metric it is expected to move — plus R11's warning that `w_spot: 0.6` is
not baseline.

---

## 10. Tests

**Captured before any refactor**, or it is worthless:

1. **Default identity.** Freeze the current model's `p_up, var_y, y_star, m_Y,
   eps_bar, tail` on a fixed quote sample to a golden `.npz`, then assert
   `np.array_equal` after consolidation. This is brief §1's "bit-identical"
   requirement and is the single load-bearing test.

Then:

2. `kappa_vol` relocation: identical at 0; differs at ≠ 0 (asserted, with the
   old behaviour recorded).
3. Orthogonality: `kappa_vol = c` ≡ `tail_scale = exp(c)` on `perp_single`
   (R2, R3).
4. `kappa_tail` changes ν and leaves `ν/(ν−2)·σ²` unchanged.
5. `temperature` leaves `p_model` unchanged and gives
   `p_quoted = σ(logit(p)/T)`.
6. `engine == fair_value` across every override (the existing test, extended).
7. Export causality: index `i` carries no information after its own instant.
8. Every shipped variant loads and runs on a short window.
9. Reachability: every surviving branch is selectable from an override key
   (R10).
10. Root directory contains exactly the four intended slot filenames.

---

## 11. Sequencing

1. Golden capture — freeze current outputs *first*.
2. Vendor code and tables; `model.json`, `config.py`, `CHANGELOG.md`;
   extract `filter_by_w.json`.
3. `Overrides` + `apply_overrides` + the per-tick transforms; move `kappa_vol`
   (§4.3); delete the dead branches (R10).
4. Export builder and the L0–L3 caches.
5. The four blocks, and raise the `link` binding question with the harness.
6. Variants and `variants/README.md`.
7. Scorecard and `report_variants.html`.

Steps 3 onward are TDD; step 1 gates all of them.
