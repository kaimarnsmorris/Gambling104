---
title: A stream standard, a registry, and a usable backtesting module
date: 2026-09-09
status: design — approved in brainstorming, not yet implemented
supersedes_in_part: 2026-09-09-backtesting-harness-design.md
related:
  - 2026-09-09-backtesting-harness-design.md
---

# Stream registry and module design

## The question this answers

**How does an investigation reference tick data — including streams recorded
after the harness was written — without editing the harness?**

Today adding a data source means editing three places: a constant in
`paths.py`, a join in `episodes.py`, and a field on the `Episode` dataclass.
That friction is the reason five spot-panel constants, three copies of
`fair.py`, and a stale spec exist.

## Scope

This is **sub-project 1 of 3**, chosen to go first because the other two build
on it:

1. **This spec** — the stream standard, the registry, model directories, one
   public API, and a bounded cleanup.
2. **Event replay** — a pluggable time base so grid and event modes are one
   engine. Designed for here, built later.
3. **Raw NDJSON → parquet** — extends coverage to 2026-08-26 and unlocks
   trades, depth diffs, klines, force orders. Independent.

**Out:** event replay itself, the NDJSON converter, any change to the fill or
fee models, any tuning of quote parameters.

---

## 1. The stream standard — self-describing parquets

A conforming stream file explains itself, so the registry needs no external
declaration. This mostly **codifies what the recorder already does**: `recv_ns`
is already the receipt column in `l1_events`, `l1_mid_10ms`, `trades` and
`chain_reports`, and `l1_events` already carries `exch_ns` beside it.

### Reserved columns

| column | type | required | meaning |
|---|---|---|---|
| `recv_ns` | int64 | **yes** | Epoch nanoseconds, when **this host received it** |
| `src_ns` | int64 | when available | The source's own stamp (`exch_ns`, `oracle_ms`, …) |
| `venue` | string | no | For multi-venue streams |
| `seq` | int64 | no | Sequence number, for gap detection |

**Every other column is a value column.** There is no metadata map of value
columns: the schema is authoritative and cannot drift from itself.

### The one rule that carries most of the value

**Alignment is always `recv_ns`, never `src_ns`.**

A source stamp can *lead* receipt. `rtds_btc`'s `oracle_ms` runs roughly 1.5 s
ahead of `px_first_recv_ns`, so aligning on it feeds a model prices before it
was told them — lookahead, silently, in the alpha column. That bug was found
and fixed by hand on 2026-09-09. Making receipt the only alignable column means
it cannot be written again.

`src_ns` is **required when the source provides one**. It is free to record and
is the only route to ever *measuring* transport lag rather than assuming it —
the same gap that left the venue↔panel clock offset unmeasurable.

### File-level metadata

Parquet key-value metadata, identity and semantics only:

```
stream.schema        = "1"
stream.name          = "venue_l1_events"
stream.asset         = "BTC"
stream.causal        = "true"
stream.recorder      = "london_recorder"
stream.created_utc   = "..."
stream.span_start_ns = "..."
stream.span_end_ns   = "..."
```

Metadata lives in the parquet footer and is parsed once per file at open, never
per row — it has no effect on read throughput. It is also robust across writer
versions: pyarrow 19 reads these files' metadata even where it cannot read
their data pages.

### Layout

`<stream_root>/date=YYYY-MM-DD/*.parquet`, matching `Z:\parquet` today.

### Applies to derived parquets, not raw captures

Raw NDJSON stays verbatim — losslessly whatever the exchange sent — and the
converter normalises to this standard. One enforcement point (the writer), not
two, and a capture bug can never be masked by normalisation at capture time.

### Tooling

- `harness.streams.write_stream(df, root, spec)` — emits conforming files with
  metadata assembled correctly, so the recorder never hand-rolls it.
- `harness.streams.validate_stream(path)` — fails loudly on a missing `recv_ns`
  or absent metadata. Belongs in the recorder's own tests so a non-conforming
  file never reaches the archive.

---

## 2. The registry and the Episode accessor

### Resolution

Streams resolve **investigation-first, catalog-second**, mirroring how blocks
already resolve. A shared catalog holds the standard streams (`book`,
`strikes`, `spot`, `chainlink`); an investigation registers new ones locally and
may shadow a catalog entry by name.

```python
register("my_capture", "/path/to/root")          # conforming: nothing declared
```

**The catalog lives at `harness/streams/catalog.py`** — one module of `register()`
calls for the standard streams, importable and readable in one screen. An
investigation adds its own by calling `register()` in its `run.py` before
`backtest()`, or in a `streams.py` beside it if there are several.

**The registered name is the local alias and always wins.** A conforming file's
`stream.name` metadata is informational — it records what the recorder called
it, which is useful when the same capture is registered under different names in
different investigations. Where they differ, the run manifest records both.

**`causal` comes from metadata for conforming files and from the adapter for
legacy ones.** If a conforming file's metadata lacks `stream.causal`,
registration fails rather than defaulting: whether a series may be carried
forward onto a decision grid is not a safe thing to guess.

### Legacy adapter

Non-conforming files — everything recorded before this standard — need an
explicit adapter. This is the fallback, not the normal path:

```python
register("chainlink", RTDS_PATH, adapter=Stream(
    time_col  = "px_first_recv_ns",
    time_kind = TimeKind.RECEIPT,     # REQUIRED, no default
    time_unit = "ns",
    causal    = True,
))
```

`time_kind` has no default. Declaring `RECEIPT` is free; declaring
`SOURCE_STAMP` additionally requires a transport offset, which the run records
as a declared assumption — the same treatment the clock offset gets. A stream
that cannot say which it is cannot be registered.

### The accessor

Core fields (`bid`, `ask`, `mid`, `strike`, `settle`, `t_ms`) stay statically
typed and unchanged, so existing blocks keep working untouched. Registered
streams come through an explicit accessor:

```python
cl = ep.stream("chainlink")
cl.px          # decision-aligned array
cl.twap60
cl.age_ms      # ms since KNOWABLE, not since stamped
cl.has         # bool
```

An unregistered name raises immediately and lists what is registered, so a typo
fails at the first tick rather than surfacing as NaN deep in a sweep.

Registered streams route through the same `shift_to_decision_grid` as
everything else when `causal=True`, so a new stream inherits the lookahead
guarantee rather than re-earning it. They participate in warm-up automatically,
so a new stateful estimator gets burn-in for free.

### Time base — designed now, exercised later

`ep.t_ms[i]` is the decision time of index `i`: today exactly `i * 100`, later
whatever an event stream supplies. Blocks and the engine ask *"what time is
index i"* rather than assuming 100 ms buckets.

This is the whole forward-compatibility story for sub-project 2. Event replay
then supplies a different `t_ms` array and converts index arithmetic to
`searchsorted` in three places — `latency.delay_idx`, `ledger.markout`, and the
loop — with no change to stream declarations, blocks, or the accessor.

---

## 3. Model directories

An investigation names a **model directory** instead of copying blocks into
itself. Resolution order becomes: investigation folder → model directory →
`harness/blocks/defaults/`.

Copying was the right call for a one-off point-in-time evaluation and is why
`fair.py` now exists in three places that have already drifted; the 2026-09-09
fair fix had to be applied twice by hand.

**Provenance is unaffected.** Every run already freezes the resolved block files
into `runs/<id>/blocks/` with sha256s in the manifest, so a run folder still
answers *what exactly was this model?* byte-for-byte. Referencing loses nothing
that copying provided.

Shadowing still works: a local `link.py` overrides the model's, so forking one
block remains a one-file act.

---

## 4. The public API

```python
from harness import backtest, Sample, QuoteParams, ExecConfig, Output

result = backtest(
    model   = "models/normal_qq",
    streams = ["book", "strikes", "spot", "chainlink"],
    quote   = QuoteParams(e_p=0.01, rpl_p=0.0005, max_pos=50, shares=10),
    execn   = ExecConfig(),
    sample  = Sample(days=(...), require=("spot", "chainlink")),
    output  = Output(seeds=(0, 1, 2), emit_ticks=True, plots=True),
)
```

`backtest()` returns a result object carrying `summary`, `ledger`, `markets`,
`ticks` and `run_dir`, so plotting and analysis work off the return value rather
than re-reading parquet by path.

**`Sample(require=(...))` generalises `require_spot`.** That flag ignored
Chainlink and silently scored **237 unpriceable markets as $0.00**, which
entered the mean as real observations. Requiring streams by name makes that
class of bug unrepresentable.

`require=("spot", "chainlink")` means: **exclude any market for which a named
stream has no usable observation inside the window.** Excluded means absent from
the run entirely — not present with NaN, and never scored as zero. The count of
markets dropped per stream is reported in `summary.json`, so a require clause
that silently halves the sample is visible rather than inferred from a market
count that looks lower than expected.

---

## 5. Cleanup

**In scope:**

- Collapse five spot constants (`SPOT`, `SPOT_USD`, `SPOT_ORACLE_WINDOW`,
  `SPOT_LEGACY_USDT`, `SPOT_LONDON`) into catalog entries. Old run folders still
  reproduce: the manifest fingerprints inputs by sha256.
- Fix `load_episodes`'s O(markets × rows) scan — a full boolean pass over a
  4.6 M-row frame per market. Groupby once.
- **Sweep the dead surface.** `apply_daily_minimum` is implemented, tested and
  wired to nothing. The final whole-branch review flagged this as a *pattern*
  (three such functions), so audit rather than fixing one at a time.
- Remove the orphan `data/spot_5m_100ms.parquet.sha256`.
- Reconcile `2026-09-09-backtesting-harness-design.md` with the code. It still
  describes `mode="maker"|"taker"|"both"` (removed), snapping in `quote.py`
  (moved to `execution.py`), a *measured* clock offset (superseded — it is a
  declared assumption), and `Sample.split` (deleted). A stale spec is worse than
  no spec.
- Update `investigations/_template` to the new API.

**Explicitly out of scope:** `Gambling104/investigations/2026-09-09_normal_qq_basic`
is owned by a concurrent session and is not to be moved or edited. `models/normal_qq`
is created from this module's copies; that session repoints when convenient.

**Migration is additive.** The registry and `harness.api` land alongside the
existing surface; `paths.py` constants become thin wrappers over catalog
entries; investigations port one at a time; the old surface is deleted only once
nothing imports it. Chosen because a second session is committing to the same
branch and because the first useful capability — declaring a new stream — works
before the cleanup finishes.

---

## Decisions and why

| decision | rationale |
|---|---|
| Self-describing parquets over external declarations | Data that explains itself cannot drift from its description. Codifies existing `recv_ns` practice. |
| `recv_ns` the only alignable column | A source stamp can lead receipt by ~1.5 s; aligning on it is lookahead by construction. |
| `src_ns` required when available | The only route to measuring transport lag instead of assuming it. |
| No value-column map in metadata | The schema is authoritative; a map is a second source of truth that can disagree. |
| Standard applies to derived, not raw | One enforcement point; capture bugs cannot be masked by normalisation. |
| Typed core + `ep.stream()` accessor | Keeps autocomplete where it matters; zero harness edits for a new stream; typos fail at the first tick. |
| Model dirs over copied blocks | Provenance already comes from the run folder's frozen copies, so copying only buys drift. |
| Time-based interface now | Makes event replay additive rather than a rework of every declaration and block. |
| Additive migration | A second session shares the branch. |

## Known limits

1. **Event replay improves signal timing, not fill realism.** The Polymarket
   book is a 10 Hz state sampler; no event-level CLOB capture exists and nothing
   can backfill it. Event mode makes decisions fire at the true instant a venue
   moved; it does not make fills more real.
2. **`SPOT_LONDON` is BTC/USDT, uncorrected.** `panel_100ms.parquet` carries no
   `usdt_basis`, so that panel is ~+$43–56 against BTC/USD. It is usable only
   because `fair.py` learns the basis against the oracle. Any block assuming USD
   would be wrong by that much; the catalog entry must say so.
3. **The clock offset remains a declared assumption**, not a measurement. It
   cannot be measured from streams carrying different events.
4. **The 100 ms grid flatters results** by ≈$0.13/market, still an inherited
   constant rather than one measured on this harness.
