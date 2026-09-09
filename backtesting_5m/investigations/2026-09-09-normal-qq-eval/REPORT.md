# normal_qq_basic — evaluation, 2026-09-09 (unified execution policy)

> ## RE-RUN 2026-09-09, warm-up on — READ THIS SECTION FIRST
>
> Four upstream defects have been fixed since every number below was measured:
> the spot panel was BTC/USDT rather than BTC/USD; `fair.py` now LEARNS the
> venue-to-oracle basis instead of trusting the capture's quote, off a Chainlink
> line gridded by RECEIPT rather than by the oracle's own stamp; `s` is shifted
> onto the decision grid like every other Episode array; and Episodes now carry
> a 900 s pre-open warm-up region, which this run turns on and `fair.py`'s
> 180 s basis halflife now depends on.
>
> **Sample changed too, and not cosmetically.** The run is now on
> `harness.paths.SPOT_ORACLE_WINDOW` (2026-08-17..21), the overlap of the book
> panel, the venue L1 capture and the Chainlink feed. `fair.py` returns NaN
> without an oracle, so markets with no oracle line are DROPPED rather than
> scored as $0.00 markets: 1,391 loaded, **1,102 scored**.
>
> ### The calibration finding is no longer what it was
>
> | | previous | **this run** |
> |---|---|---|
> | model Brier, unconditional | 0.2167 | **0.1345** |
> | book Brier, identical rows | 0.1331 | **0.1285** |
> | model − book | **+0.0836** | **+0.0060** |
> | p ≥ 0.999 fraction | 0.400 | **0.204** |
> | ...and its realised frequency | 0.773 | **0.972** |
> | worst decile gap | −0.332 | **−0.130** |
>
> The old text below — "the model is systematically overconfident", 40 % of
> observations pinned at p = 1.000 settling in the money 77 % of the time — was
> true of the model it measured and is **no longer true of this one**. The
> model has closed to within 0.006 Brier of the book. `calibration.png` for
> this run shows the at-fill-time deciles lying on the diagonal, where the
> previous run had the top decile at 0.98 predicted against 0.67 realised.
>
> **It is still not perfectly calibrated and this report does not claim it is.**
> The residual error has the same sign as before: 20.4 % of observations still
> sit at p ≥ 0.999 and settle in the money 97.2 % of the time, and the second
> highest decile predicts 0.952 against a realised 0.822.
>
> ### The loss did not go away
>
> | | previous run (00:28 today) | **this run** |
> |---|---|---|
> | markets | 1,553 | 1,102 |
> | fills | 54,717 | 38,473 |
> | c/share (`pnl_net`) | −1.483 | **−1.299** |
> | $/market (`pnl_net`) | −5.224 | **−4.534** |
> | day-blocked CI ($/mkt) | — | **[−6.717, −2.531]** |
> | gates | pass | **pass** |
>
> Per-period means −4.59 / −4.35 / −7.49 $/mkt, the CI excludes zero, and
> deleting the ten best markets makes it worse (−4.534 → −4.866). The two runs
> are on different samples and different day counts, so the levels are not
> strictly comparable; the sign, the robustness and the order of magnitude are.
> **Fixing the model's calibration did not make it profitable.** It is closer
> to the book's own prices, and being closer to the book's prices is exactly
> what leaves nothing to trade against after fees.
>
> Maker/taker split for this run (`split_headline.py`, markout-based, seed 0):
>
> | arm | fills | markets touched | mean markout (c/share) | mean fee (c/share) | net c/share | $/market (÷1,102) |
> |---|---|---|---|---|---|---|
> | maker | 14,039 | 1,087 | **−2.550** | −0.269 (paid to us) | −2.280 | −2.905 |
> | taker | 24,434 | 1,101 | **−0.491** | +1.254 (paid by us) | −1.745 | −3.870 |
> | overall | 38,473 | 1,101 | −1.242 | +0.698 | −1.941 | −6.775 |
>
> Latency ladder (450-market subsample, seed 20260909, single sweep seed 0),
> c/share: **−0.976 / −0.976 / −1.015 / −1.049 / −1.075** at 0 / 100 / 200 /
> 250 / 500 ms. Fill-arm range: optimistic −0.976, adverse_lag −0.976,
> penetration −1.041. Both ranges are narrow and every arm's sign matches.
>
> The sigma investigation next door
> (`../2026-09-09-vol-fixed/REPORT.md`, PART III) runs the same blocks with
> three vol arms on the same panel and finds the same thing from the other
> side, including a direct two-directional test of "better calibration, bigger
> loss" that the pattern passes.
>
> Everything below this box is the previous evaluation, kept verbatim for the
> before/after. Its calibration section is superseded by the table above.


Blocks copied from `Gambling104/investigations/2026-09-09_normal_qq_basic/`
(`fair.py`, `vol.py`, `f.py`, `link.py`), unchanged. Exact versions are pinned
by sha256 in each run's `manifest.json`.

**This supersedes the previous evaluation in this folder.** The execution
policy changed underneath it (see "Not comparable" below), so the earlier
maker/taker numbers are void and have been overwritten, not appended.

Run artefacts: `runs/2026-09-08T23-51-44__4e2601/` (headline, all plots),
`runs/2026-09-08T23-54-38__3fe78a/` (sweeps, base + 5 latency arms + 3 fill
arms), `runs/2026-09-08T23-57-19__8546de/` (tick emission for the 4 detail
markets). Re-run 2026-09-09 to add the `spot` tick column and the BTC-space
detail panel; the headline numbers below are unchanged (same model, same
policy, one extra diagnostic column). Headline run took 175 s, sweeps 161 s,
detail 0.5 s — total ~5.7 minutes end to end.

## What changed and why the old numbers are void

`ExecConfig.mode` no longer exists. Execution is a single pass that rests
**and** crosses in the same tick: a cross now only fires when
`book_ask <= eff_bid - taker_fee(book_ask)` (mirrored on the ask side), i.e.
the edge must clear the fee, not just beat a 1-cent threshold. Quote snapping
moved from `quote.py` into `execution.py`, so the maker rebate is applied
once, before the single snap to the cent grid, instead of being applied and
then annihilated by a second snap — it now moves the placed price a full
tick roughly 34.9% of the time near the money. `s` is now shifted onto the
decision grid like every other Episode array (previously the one column that
wasn't). None of this touches `fair.py`/`vol.py`/`f.py`/`link.py`, so the
calibration finding below is unaffected and reproduces exactly.

`eff_bid`/`eff_ask` in the ledger and tick frame are **theoretical
valuations**, not order prices — see `harness/core/ledger.py`'s docstring.
The actual placed/executed price is `price` on a ledger fill: for a maker
fill it is the fee-adjusted, snapped resting limit that fill traded at; for a
taker fill it is the crossing price paid. The per-market detail plots below
draw eff_bid/eff_ask as dotted "theoretical" lines and fill markers at the
real placed price, split by maker (filled marker) vs taker (hollow marker).

**Each per-market detail figure now has a fourth panel, in BTC dollar
space**, sharing the same seconds-to-expiry x-axis as the three panels above
it: venue BTC spot, the model's fair BTC estimate (`fair.py`'s `E[A]`, i.e.
`E[settling TWAP]`), the real Chainlink 60 s TWAP (`twap60`) from the RTDS
capture, and the Chainlink strike and settlement levels (horizontal lines,
solid and dashed). This is the probability panel's story told in the units
the model actually forecasts in: `E[A]` is a forecast of `twap60`
specifically, so the two sit on the same axis and the vertical gap between
them, at any point in the market, is the model's forecast error.

**Correction to an earlier draft of this note:** a previous version of this
report said no continuous Chainlink series existed and that only the two
boundary levels could be drawn. That was wrong — it was based on the
boundary-report capture (`chain_reports.parquet`) alone. A real, continuous,
1 Hz Chainlink feed does exist (`harness.paths.RTDS_BTC`, an external,
read-only capture living in the sibling `Gambling102` repo, not copied into
`data/`), and `twap60` on it is exactly the settlement variable this era's
markets pay out on. Its coverage window is **2026-08-14 through
2026-08-21 only**, which does not reach the back half of this evaluation's
six-day sample (08-19 → 08-24) — so the twap60 line is drawn only for
markets on 08-19/08-20, and the four per-market detail figures are
deliberately chosen from those two covered days (seeded, reproducible; see
`run.py`'s `RTDS_COVERED_DAYS`) so all four BTC panels are populated. The
strike and settlement horizontals are unaffected by this coverage gap — they
come from the boundary reports, independently verified against signed
on-chain `ReportVerified` reports at median **and** max difference of $0.00
across 1,992 markets — and are drawn on every detail figure regardless of
day.

On the two markets inspected, `E[A]` tracks venue spot closely throughout
(as the model's own theory predicts for τ > 60 s, where `E[A] = S_t`), but
tracks the *actual* Chainlink `twap60` only loosely for most of the window —
the two series can sit tens of dollars apart in the middle of a market — and
converge only in the closing ~60 s, exactly where `fair.py`'s blend toward
the realised-TWAP term takes over. That convergence-only-near-expiry pattern
is the BTC-space version of the same overconfidence story the calibration
section documents below: mid-market, the model is confidently reporting a
spot-tracking number as if it were the settlement forecast, when Chainlink's
own TWAP is materially smoother and often on the other side of the strike.

## Headline

> **Re-run 2026-09-09 after the taker-lifecycle fixes.** An unfilled marketable
> order is now expired rather than resting forever (it had been holding per-side
> cap room and suppressing the resting quote), marketable orders can no longer be
> cancelled inside the venue's 250 ms lock, and the stale-book cancel no longer
> waits for the requote cadence. Net effect: slightly MORE fills (35,828 ->
> 36,835) and a slightly smaller loss per market, because the simulation had been
> dodging crosses it should have taken. All three data inputs are now fingerprinted
> in the manifest. Conclusion unchanged.


Sample: **1,553 markets, 2026-08-19 → 08-24** (6 days, `require_spot=True`),
seeds (0, 1, 2). Quote parameters `e_p=0.01, rpl_p=0.0005, max_pos=50,
shares=10` — **not optimised, not tuned in response to these results.**

One run now makes and takes simultaneously. The official headline (from
`markets.pnl_net`, the harness's own mark-to-market/settlement PnL) is:

| | n_markets | n_fills | c/share | $/market | day-blocked CI ($/mkt) | gates |
|---|---|---|---|---|---|---|
| **overall (harness pnl_net)** | 1,553 | 36,835 | **−1.420** | **−3.369** | [−4.136, −2.469] | **pass** |

"Gates pass" = the loss is robust: sign survives all 3 calendar periods
(−1.97, −4.13, −4.39 $/mkt), the day-blocked CI excludes zero, and deleting
the ten best markets does not flip the sign (−3.42 → −3.67).

`pnl_net` cannot be split by liquidity after the fact — it is whole-market
mark-to-market, mixing every fill. To report maker/taker separately, the
ledger was split on `liquidity` and each fill's fee-inclusive economics were
summed directly (`delta_quality_c` markout, net of `fee_usd`; see
`split_headline.py`). This is a different, markout-based quantity — it
totals **−$6,438** vs pnl_net's **−$5,309** (a ~17% gap, expected: markout is
a 10s-forward-or-settlement proxy per fill, not the path-dependent
mark-to-market the official number is) — but it is internally consistent and
is the only way to see the two liquidity types apart:

| arm | fills | markets touched | c/share (markout, net of fee) | $/market (÷1,553 all) | mean markout c/share | mean fee c/share |
|---|---|---|---|---|---|---|
| maker | 7,370 | 1,377 | −2.070 | −0.982 | −2.29 | **−0.22 (paid to us)** |
| taker | 28,466 | 1,553 | −1.726 | −3.161 | −0.50 | **+1.23 (paid by us)** |

**Fills did not collapse.** 28,466 of 35,828 fills (79%) are taker crosses,
even though the threshold now requires clearing the fee — the model's
average disagreement with the book is large enough that fee-aware crossing
still fires constantly. This says the mispricing (below) is large relative
to the fee (peaks at 1.75 c/share), not that the new policy is inert.

## The cause: the model is not calibrated, and the book is (carried forward, unchanged)

This is a property of `fair.py`/`vol.py`/`f.py`/`link.py`, not of execution,
so it is unaffected by the policy rewrite and is carried forward verbatim
from the previous evaluation. Reproduced by rerunning `python
calibration_uncond.py` just now:

```
episodes: 1553
observations: 7761

--- MODEL fair_p, unconditional ---
 pred  real    n    gap
0.000 0.013  777  0.013
0.033 0.138  776  0.105
0.292 0.228  776 -0.064
0.573 0.392  776 -0.181
0.824 0.553  776 -0.271
0.973 0.642  776 -0.332
1.000 0.773 3104 -0.227
Brier: 0.2167

--- BOOK mid, same rows (benchmark) ---
 pred  real    n    gap
0.004 0.001  818 -0.003
0.064 0.079  768  0.015
0.206 0.219  744  0.013
0.345 0.327  779 -0.018
0.471 0.410  846 -0.060
0.571 0.588  767  0.017
0.691 0.657  732 -0.034
0.835 0.856  806  0.021
0.976 0.981 1331  0.005
0.999 1.000  170  0.001
Brier: 0.1331
```

Model Brier 0.2167 against the book's 0.1331 on identical rows; worst decile
gap −0.332; **40% of observations (3,104 / 7,761) are pinned at exactly
p = 1.000 and settle in-the-money only 77.3% of the time.** The book is
close to perfectly calibrated (every decile within ±0.06); the model is
systematically overconfident. The per-fill calibration plot
(`calibration.png`, deciles of `fair_p` at fill time vs realised outcome on
the *traded* subset) shows the identical pattern — decile means run below
the 45° line throughout, worst at the top decile (0.98 predicted vs 0.67
realised, n=21,495) — so the traded subset does not disagree with the
unconditional table; it confirms it. **This evaluation does not repeat the
prior error of declaring the model well calibrated** — it is not, in either
measurement.

## Mean markout by liquidity

| arm | mean markout (delta_quality_c, c/share) | mean fee (c/share) |
|---|---|---|
| maker | −2.29 | −0.22 (rebate received) |
| taker | −0.50 | +1.23 (fee paid) |

Maker markout is more negative than taker's — consistent with the original
mechanism (resting on the side the market is about to move against, and the
harness's cancel-race model fills those quotes hardest exactly when the book
is moving). Taker's smaller markout magnitude is offset by paying the fee
outright on every fill, which the maker side does not (it is paid instead).

## Latency ladder and fill-arm range (sweeps, 450-market subsample, seed 20260909, single sweep seed 0)

Latency ladder (`place_ms = cancel_ms = take_ms` swept together), c/share:

| latency (ms) | 0 | 100 | 200 | 250 | 500 |
|---|---|---|---|---|---|
| c/share | −2.413 | −2.413 | −2.352 | −2.340 | −2.354 |

Fill-arm range (optimism about the cancel race), c/share:

| arm | optimistic (cancel=0) | adverse_lag (as configured) | penetration (queue proxy) |
|---|---|---|---|
| c/share | −2.413 | −2.413 | −2.497 |

Both ranges are narrow (≈0.16 c/share peak to trough) and every arm's sign
matches the headline. The loss is not an artefact of a particular latency or
fill-optimism assumption.

## Not comparable to the previous run folders

The previous evaluation in this folder ran two separate arms
(`ExecConfig(mode="maker")` and `ExecConfig(mode="taker")`) against a taker
threshold that crossed on a flat 1-cent edge regardless of the fee, and with
the maker-rebate/tick-snap bug described above (rebate applied, then
annihilated by a second snap). Both defects are fixed here. The unified run
is a different execution model end to end — a single ledger with a
`liquidity` column, fee-aware crossing, and a rebate that actually moves the
quote — so its c/share and $/market numbers are not the same measurement as
the old maker/taker split and must not be diffed against it. Only the
calibration finding (a model property, not an execution property) is safe to
carry forward, and it does, unchanged.

## Standing caveats — these bind every number above

1. **The 100 ms replay grid flatters results by ≈ $0.13/market**, measured
   twice on two independent models. Results here are *before* that haircut.
2. **Every maker fill is modelled, not measured.** This dataset has no
   depth, no trade tape and no queue. The maker number is a statement about
   the fill block as much as about the market; the fill-arm sweep above is
   what bounds it.
3. **The venue↔panel clock offset is a configured 0.0, not a measurement.**
   It cannot be measured from these two streams — they carry different
   events. The repo's own same-book drift measurements bound the plausible
   range at 0–74 ms.
4. **Six days, one instrument, one regime.** 2026-08-18 is absent
   (source-archive schema defect). Nothing here survives a regime change.

## What this does NOT establish

- Not a verdict on the model's design as such — it is one parameterisation,
  unoptimised, on six days, and the failure is upstream of execution.
- The quote parameters (`e_p=0.01, rpl_p=0.0005`) were not tuned in response
  to any of these numbers, per instruction; a hill-climbed number from six
  days would be overfitting.
- The maker/taker split above is a markout-based decomposition for reporting
  purposes, not the harness's own headline metric (`pnl_net`) — it agrees in
  sign and order of magnitude but not exactly, as noted above.

## Suggested next step

Unchanged from the previous evaluation: fix calibration before touching
execution further. The cheapest decisive test is still to plot `sigma` from
`vol.py` against realised |ln(S_T/S_t)| by τ bucket — saturation at the
bound (40% of observations pinned at p=1.000) is the signature of a `sigma`
that is too small, which points at `vol.py`'s `tau_eff` term structure. This
evaluation is one command to rerun (`python run.py` then `python
plot_report.py`) once that is fixed — the investigation folder and its
blocks are frozen and re-runnable.

Reproduce the calibration table with `python calibration_uncond.py`.
Reproduce the maker/taker split with `python split_headline.py
runs/2026-09-08T23-51-44__4e2601 0`.
