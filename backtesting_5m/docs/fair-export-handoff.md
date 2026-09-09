# Fair-value export: handoff to the harness

**From** `investigations/2026-9-9_business_clock_t_dist` (the Chainlink fair-value
consolidation, model `fv-1.0.0` / overrides `ov-1.0.0`).
**To** whoever owns `backtesting_5m/harness`.
**Date** 2026-09-09.

This is the written statement `harness/core/episode.py` asks for, plus the two
things that block the harness from running these blocks today. Three sections:
the export, the causality contract, and the two asks.

---

## 1. The export

**Location.** `backtesting_5m/data/fair/<variant>.parquet`, with a provenance
sidecar `<variant>.json` beside it. `baseline.parquet` is the shipped default;
the block loader picks the variant from the `FV_VARIANT` environment variable
and defaults to `baseline`. Written by

    python -m export.build_export --variant baseline

run from the investigation folder. The path is the same constant the harness
already has (`harness/paths.py::FAIR_DIR`); the builder resolves it
independently and the two agree.

**Grain.** One row per (market, second). Every market gets exactly 301 rows,
`t_s` running from `-1` to `299` inclusive, priceable or not. The current
`baseline.parquet` is 1,552,558 rows over 5,158 markets.

**Schema.**

| column | type | what it is |
|---|---|---|
| `market_id` | str | joins to the panel and to `strikes_5m.parquet` |
| `open_ts` | i64 | market open, unix seconds |
| `t_s` | i32 | the information instant, seconds after `open_ts`; `-1 … 299` |
| `s` | f64 | the settlement level at which the model is indifferent, in USD. This is the harness's `E[A]`. **Strike-free** |
| `sigma` | f64 | settlement standard deviation, same units as `s` |
| `nu`, `mu`, `sigma_t` | f64 | the fitted Student-t settlement tail at this row's business time |
| `p_model` | f64 | P(up) at the market's **real venue strike** — for scorecards, not for the blocks |
| `p_quoted` | f64 | `p_model` after the quoting-layer temperature |
| `omega` | f64 | unknown fraction of the settlement average |
| `n_known`, `n_transit` | f64 | settlement components already received / stamped but not received |
| `m_Y`, `eps_bar`, `carry` | f64 | the location terms already folded into `s` |
| `var_eps`, `var_basis` | f64 | the two residual variance channels already folded into `sigma` |
| `z_business` | f64 | log business time remaining |
| `p_ref` | f64 | the reference price the linearisation is around |
| `ok` | bool | false where the model could not price the cell; `s` and `sigma` are then NaN |

Rows with `ok = False` carry NaN `s`/`sigma`. The harness already treats an
absent `s` as absent and forward-fills nothing, so no special handling is
needed.

**The blocks.** `fair.py`, `vol.py`, `f.py` and `link.py` in the investigation
folder are the four slot files; they read the export through a shared
`_fvexport.py` loader (deliberately not named after a slot). `f.py` is
`(level − strike) / sigma`; `link.py` is the tail. `s` being strike-free is what
lets the strike work live in the blocks — asserted by
`tests/test_export.py::test_s_and_sigma_reproduce_y_star`, which shifts the
strike and confirms `s` does not move.

---

## 2. The causality contract — CONFIRMED IN WRITING

`harness/core/episode.py::build_episode` says:

> A caller who has confirmed IN WRITING that the export is already decision
> aligned may pass `fair_is_causal=True` to opt out of the shift.

**This is that confirmation.** Precisely:

> A row of this export labelled `(market_id, t_s)` carries information through
> the instant `open_ts + t_s` and no later. It becomes usable at
>
>     t_ms = max(0, t_s * 1000 + 100)
>
> and holds until the next second's row supersedes it. The row is therefore
> already on the harness's decision grid — the 100 ms offset in that formula IS
> the harness's own "index `i` may only use buckets `k <= i − 1`" rule, applied
> at build time.

Consequences for the harness:

* **Pass `fair_is_causal=True` when building an Episode from this export.**
  The default path shifts `s` by another bucket, which double-shifts it: the
  strategy would be pricing off information it was entitled to 100 ms earlier.
  That is conservative rather than lookahead, so nothing is unsound — but it is
  silently wrong, it is not the model this investigation scored, and it costs
  100 ms of edge in a five-minute market where the last thirty seconds are the
  cell that matters.
* The shift is defined in **exactly one place**:
  `export/build_export.py::bucket_owner()`, which materialises, for each of the
  3,000 buckets, the `t_s` whose value it must use. Its inverse is
  `usable_t_ms(t_s)` in the same module. `_fvexport.for_episode` expands the
  one-second export onto the 100 ms grid by *gathering* through `bucket_owner()`,
  so the blocks restate nothing. If the convention ever changes, it changes
  there and nowhere else.
* `t_s = -1` exists solely so bucket 0 has an owner: it is the pre-open row,
  information through `open_ts − 1`. `t_s = 299` owns buckets 2991–2999 (nine of
  them, because 301 values cannot divide 3,000 buckets evenly).
* Tests on our side: `tests/test_export.py::test_bucket_owner_partitions_the_grid`
  and `::test_export_is_causal` assert that every bucket is served by a row whose
  information instant is **strictly** earlier than the bucket.

---

## 3. Both asks are WITHDRAWN

This section asked for two changes. Neither is needed: the harness had already
solved both by the time the model was rewritten against it, and the model now fits
the interface as it stands.

### 3.1 `link` per-tick binding — withdrawn

The original ask was for `link` to receive a tick index, because the settlement
tail's `(nu, mu, sigma_t)` vary with business time and `link(z)` is a pure function.

Resolved on our side instead. The tail is folded into the two channels the harness
already provides — a location shift on `s` and a scale on `sigma` — against one fixed
link shape per run. `models/chainlink_fv/tailfold.py` carries the derivation; the
cost is a median 0.07c and 95th-percentile 0.42c of probability in the quoting band,
measured on 100k real rows and held by a test.

`link` is now a one-argument function, exactly as `loop.py` and `quote.py` call it.

**If you ever do bind `link` per tick** — for another model, or because it is cheap —
tell us and the fold goes away: delete `tailfold.py`, hand the tail over whole, and
the concession with it. Nothing else in the model changes. But do not do it for us.

### 3.2 `paths.INVESTIGATIONS` — withdrawn

`backtest(investigation_dir=...)` takes an explicit path, so where the investigation
folder physically sits never comes up. The model itself lives in
`models/chainlink_fv/`, alongside `models/normal_qq/`, and is passed as `model=`.

## 4. What is NOT open

* The export's location, schema and grain are frozen for `fv-1.0.0`; a change
  bumps the version in the sidecar's `provenance`.
* `p_model` / `p_quoted` are for scorecards. The blocks ignore them and go
  through `f`/`link`, because `f` perturbs the level and the quote's `z` is not
  the export row's `z`.
* The variant is chosen by the `FV_VARIANT` environment variable, default
  `baseline`. The variant's quoting-layer temperature and tail family are
  per-variant constants read back from the sidecar's `overrides_non_default`,
  not parquet columns.
