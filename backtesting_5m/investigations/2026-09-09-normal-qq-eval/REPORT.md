# normal_qq_basic — evaluation, 2026-09-09

Blocks copied from `Gambling104/investigations/2026-09-09_normal_qq_basic/`
(`fair.py`, `vol.py`, `f.py`, `link.py`). Exact versions are pinned by sha256 in
each run's `manifest.json`.

Run artefacts: `runs/2026-09-08T18-49-31__ae871a/` (maker + all plots),
`runs/2026-09-08T18-52-01__8f325e/` (taker), plus nine sweep-arm folders.

## Headline

Sample: **1,553 markets, 2026-08-19 → 08-24** (6 days), every market with spot
coverage. Quote parameters `e_p=0.01, rpl_p=0.0005, max_pos=50, shares=10` —
**not optimised, and deliberately not tuned.**

| arm | markets | fills | c/share | $/market | day-blocked CI ($/mkt) | gates |
|---|---|---|---|---|---|---|
| maker | 1,553 | 6,922 | **−2.417** | −1.077 | [−1.778, −0.472] | pass |
| taker | 1,553 | 34,049 | **−1.254** | −2.750 | [−3.383, −2.000] | pass |

"Gates pass" here means the **loss** is robust: its sign survives every calendar
period, the day-blocked CI excludes zero, and deleting the ten best markets does
not flip it. This is a reliable loss, not noise.

## The cause: the model is not calibrated, and the book is

Unconditional calibration — five snapshots per market (τ = 270, 210, 150, 90,
30 s), 7,761 observations, model against the book's own mid on **identical rows**:

| | model `fair_p` | book mid |
|---|---|---|
| **Brier** | 0.2167 | **0.1331** |
| worst decile gap | **−0.332** | +0.021 |
| decile at ~0.97 | realises 0.642 | realises 0.981 |

The book is close to perfectly calibrated — every decile within ±0.06, most
within ±0.02. The model is not, and the error is one-directional:
**systematic overconfidence.**

**40 % of observations (3,104 of 7,761) are pinned at exactly `p = 1.000`, and
they settle in-the-money only 77.3 % of the time.**

Saturation at the bound is the signature of a `sigma` that is too small: `f.py`
computes `d2 = (ln(S/K) − ½σ²)/σ`, so an under-estimated total-vol-to-expiry
drives `z` to extremes and `Φ` pins at 0 or 1. That points at `vol.py`'s
`tau_eff` term structure — the TWAP-averaging correction is aggressive
(−90 % at τ = 10 s), and the EWMA level may compound it — rather than at the
QQ correction in `link.py`, which is trying to fatten tails that are being
crushed upstream of it.

## Why each mode loses, specifically

**Taker.** 21.9 fills per market, mean fee **1.15 c/share**, mean disagreement
with the book at fill time **12.4 cents**, mean 10 s markout **−0.46 c/share**.
The model crosses the spread on a 12-cent conviction, pays the fee, and the book
then moves *against* it. A 12-cent average disagreement with a well-calibrated
counterparty is not an edge — it is the miscalibration above, converted into
trades. Note also that `e_p = 0.01` is below the taker fee's 1.75 c/share peak,
so the threshold was never economic; but fixing the threshold cannot rescue a
signal whose sign is wrong.

**Maker.** Quoting a 1-cent edge around a fair that is systematically wrong
means resting on the side the market is about to take. The harness's cancel-race
model then fills those quotes hardest exactly when the book is moving, which is
what the −2.4 c/share reflects.

## What this does NOT establish

- **Not a verdict on the model's design.** This is one parameterisation on six
  days. The failure mode is specific and fixable, and it is upstream of the
  quoting.
- **The quote parameters are unoptimised.** Two configurations were run, not a
  search. A tuned number from six days would be overfitting.
- Per-fill calibration (`calibration.png`) is measured on the *traded* subset,
  which is selected for disagreement. The unconditional table above is the
  honest measurement; it agrees, which is why the conclusion stands.

## Standing caveats — these bind every number above

1. **The 100 ms replay grid flatters results by ≈ $0.13/market**, measured twice
   on two independent models (CI [+0.053, +0.239] and [+0.025, +0.229]). Results
   here are *before* that haircut.
2. **Every maker fill is modelled, not measured.** This dataset has no depth, no
   trade tape and no queue. The maker number is a statement about the fill block
   as much as about the market; the fill-arm sweep is what bounds it.
3. **The venue↔panel clock offset is a configured 0.0, not a measurement.** It
   cannot be measured from these two streams — they carry different events. The
   repo's own same-book drift measurements (0–74 ms) bound the plausible range.
4. **Six days, one instrument, one regime.** 2026-08-18 is absent (source-archive
   schema defect). Nothing here survives a regime change.

## Suggested next step

Fix calibration before touching execution. The cheapest decisive test: plot
`sigma` from `vol.py` against realised |ln(S_T/S_t)| by τ bucket. If the model's
σ is low, the whole chain follows. Reruning this evaluation afterwards is one
command — the investigation folder and its blocks are frozen and re-runnable.

Reproduce the calibration table with `python calibration_uncond.py`.
