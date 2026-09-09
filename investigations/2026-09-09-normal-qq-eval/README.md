# normal-qq-eval

Evaluation of the "normal QQ" fair-value model: `fair` (E[settling 60 s TWAP]
in USD from the venue spot panel), `vol` (EWMA realised-vol term structure
folding in the TWAP-averaging effect), `f` (d2 log-moneyness), `link` (normal
CDF with a quintile QQ correction).

## Provenance

`fair.py`, `vol.py`, `f.py`, `link.py` were **copied, unmodified**, on
2026-09-09 from:

```
C:/Users/kaima/Github2/Gambling104/investigations/2026-09-09_normal_qq_basic/
```

That folder is owned by a separate, live session and was not otherwise read,
written, or run as part of this work. Only these four files were copied out
of it.

Each run's `manifest.json` records the sha256 of every block file it actually
used (see `harness/core/provenance.py`), so the exact version of these blocks
evaluated by any given run in this investigation is pinned there, independent
of anything that happens to the source folder afterwards.

## Slots left at harness defaults

`quote`, `execution`, `fill`, `fees` are not overridden here -- this folder
only replaces the forecasting model, not the policy or the market mechanics.
