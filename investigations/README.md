# Investigations

One folder per question. A folder is a set of block overrides plus a `run.py`.

`_template/` is the worked example and the only one kept: copy it, rename it,
and change what you need. It runs the benchmark model in `models/normal_qq`
over one day, sweeps a single quote parameter across three runs, and draws the
cumulative-PnL comparison -- which is the whole shape of an investigation. Run
it as-is first; it should select several hundred markets and write a PNG.

The exploratory investigations that came before it are in git history, not
here.

## How blocks resolve

A block slot is a filename. The harness looks in **this folder first**, then in
the shared model directory you passed as `model=`, then in
`harness/blocks/defaults/`. Nothing is registered and nothing is named --
dropping `link.py` here IS the override, and it shadows only that one slot.
The other three keep coming from the model, so forking one block stays a
one-file act.

Slots: `fair` `vol` `f` `link` `quote` `execution` `fill` `fees`

## What a run leaves behind

`runs/<timestamp>__<confighash>/` containing `manifest.json` (every slot's path
and sha256, plus the harness commit), `blocks/` (frozen copies of the files
actually used), `ledger.parquet`, `markets.parquet`, `summary.json`,
optionally `ticks.parquet`, and `cum_pnl.png`.

Two runs with the same model, params and data share a config hash. If yours
does not match one you expected, something changed.

## Before believing a number

`summary.json` carries two caveats on every run and they are not decoration:

* the 100 ms grid flatters results by ~$0.13/market, measured on two models
* every maker fill is a model, not a measurement -- report a range across fill
  optimism, never a single maker number
