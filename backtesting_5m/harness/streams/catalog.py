"""The standard streams, in one readable screen.

An investigation registers its own by calling `register()` before `backtest()`,
and shadows any of these by registering the same name.
"""
from harness import paths
from harness.streams.registry import register, registered
from harness.streams.spec import Stream, TimeKind


def _install(name, path, adapter=None):
    """Register a standard entry ONLY if nothing has registered `name`.

    `install()` runs as `backtest()`'s FIRST statement -- which is to say
    after the investigation's own `register()` calls in its `run.py`, because
    the spec tells it to register before calling `backtest()`. Registering
    unconditionally here therefore reverted the caller's shadow: episodes were
    still built from the shadowed file, but the manifest fingerprinted the
    catalog's path, sha256 and all. A confidently wrong manifest is worse than
    an empty one, because it will be believed.

    `register()` itself keeps overwrite-wins semantics untouched -- that IS
    the shadowing feature. Only the standard entries defer.
    """
    if name in registered():
        return
    register(name, path, adapter=adapter)


def _spot_adapter(name):
    """Spot panels predate the standard: they key on (open_ts, t_ms), not
    recv_ns, and are ALREADY on the decision grid.

    `pre_gridded=True` is load-bearing. `grid_stream` treats its time column as
    an absolute epoch; `t_ms` is milliseconds since the market open, so
    gridding one of these would compute a hugely negative offset and silently
    drop every observation. These entries exist for discovery and to put the
    currency warnings in one place -- `load_episodes` reads the panels by its
    existing path.
    """
    return Stream(name=name, time_col="t_ms", time_kind=TimeKind.RECEIPT,
                  time_unit="ms", causal=True, pre_gridded=True)


def install():
    """Register the standard streams. Idempotent, and NON-CLOBBERING.

    A name an investigation already registered is left alone: see `_install`.
    """
    #: PRE-GRIDDED. The book and spot panels are already keyed on
    #: (open_ts, t_ms) and are already on the decision grid. They are
    #: registered for discovery and documentation; `load_episodes` reads them
    #: by its existing path and never routes them through `grid_stream`.
    _install("book", paths.PANEL, adapter=Stream(
        name="book", time_col="t_ms", time_kind=TimeKind.RECEIPT,
        time_unit="ms", causal=True, pre_gridded=True))

    _install("spot", paths.SPOT, adapter=Stream(
        name="spot", time_col="t_ms", time_kind=TimeKind.RECEIPT,
        time_unit="ms", causal=True, pre_gridded=True))

    #: `px_first_recv_ns`, NOT `oracle_ms`. The oracle's own stamp runs ~1.5 s
    #: ahead of arrival on this vantage, so aligning on it is lookahead.
    #:
    #: ONE RECEIPT PER FIELD. This file stamps px, twap30 and twap60 each with
    #: its OWN arrival, and they disagree: twap60_first_recv_ns runs later
    #: than px_first_recv_ns on 70.8 % of 568,261 rows, median +46.6 ms, past
    #: a whole bucket on 27.5 %. Bucketing all three at the price's receipt
    #: put a not-yet-arrived twap60 on 138 of 2,992 buckets (4.6 %) of one
    #: real window, off by up to $0.72 -- and twap60 is the settlement
    #: variable `fair.py`'s E[A] forecasts, so that is the answer, not
    #: optimism.
    #:
    #: Naming the three pairs also declares that this stream has THREE value
    #: columns. The other ten -- `oracle_ms` and the per-field
    #: *_first_recv_ns / *_last_recv_ns / *_n_copies -- are bookkeeping and
    #: stay off the grid: a receipt is float64-lossy past ~256 ns at these
    #: magnitudes and is already reported as `age_ms`, a copy count is a
    #: property of the recorder, and `oracle_ms` is the source stamp the
    #: module docstring of harness.streams.spec forbids aligning on. Gridding
    #: it as a causal float leaves it one attribute access from being used
    #: that way.
    _install("chainlink", paths.RTDS_BTC, adapter=Stream(
        name="chainlink", time_col="px_first_recv_ns",
        time_kind=TimeKind.RECEIPT, time_unit="ns", causal=True,
        value_time_cols=(("px", "px_first_recv_ns"),
                         ("twap30", "twap30_first_recv_ns"),
                         ("twap60", "twap60_first_recv_ns"))))

    #: PRE-GRIDDED, one row per market rather than per bucket: market_id,
    #: open_ts, strike. Registered because the stream-registry spec names it
    #: among the standard streams and its own example passes it -- and a name
    #: nothing registers is skipped silently by `_fingerprinted_inputs`, so a
    #: caller following that example got a manifest missing the file that
    #: supplies `strike`, hence `winner_up`, hence every settled PnL in the
    #: run, while looking complete.
    #:
    #: `pre_gridded=True` is load-bearing: `open_ts` is the market open, and
    #: `grid_stream` would read it as an observation receipt and drop every
    #: row. `load_episodes` reads this file by its existing path.
    _install("strikes", paths.STRIKES, adapter=Stream(
        name="strikes", time_col="open_ts", time_kind=TimeKind.RECEIPT,
        time_unit="s", causal=True, pre_gridded=True))

    #: BTC/USD. The default. Venue mid less the capture's usdt_basis; sits
    #: +$4.50 (sd 7.44) against the Chainlink oracle, versus +$43.17 (sd
    #: 16.36) raw. The raw bias was also time-varying by day, which is why no
    #: fitted intercept could ever have absorbed it; the panel had to be
    #: rebuilt, not regressed away.
    _install("spot_usd", paths.SPOT, adapter=_spot_adapter("spot_usd"))

    #: BTC/USD, rebuilt over the oracle overlap 2026-08-17..08-21 so a
    #: basis-learning fair block has a Chainlink line for the whole window.
    _install("spot_oracle_window", paths.SPOT_ORACLE_WINDOW,
             adapter=_spot_adapter("spot_oracle_window"))

    #: BTC/USDT, UNCORRECTED -- +$43.17 (median +47.05, sd 16.36) against the
    #: BTC/USD oracle these markets settle on, measured over 1,654,313 buckets
    #: on 2026-08-19..21 by `harness/build/spot_5m_100ms.py`, and time-varying
    #: by day. panel_100ms carries no usdt_basis column, so no correction is
    #: possible here. Usable ONLY with a fair block that learns the basis
    #: against the oracle itself. A USD-assuming consumer is $43 wrong -- some
    #: 0.17 of a 300 s sigma, ~7 c of probability bias toward UP near the
    #: money; the name says usdt for that reason.
    _install("spot_london_usdt", paths.SPOT_LONDON_USDT,
             adapter=_spot_adapter("spot_london_usdt"))

    #: Superseded BTC/USDT panel. Registered only so a historical run folder
    #: can be reproduced against the data it actually used. Do not build on
    #: it. Its bias against the Chainlink oracle was +$43.17 (sd 16.36) --
    #: versus +$4.50 (sd 7.44) on the corrected panel -- and time-varying by
    #: day, which is why it had to be rebuilt rather than regressed away.
    _install("spot_legacy_usdt", paths.SPOT_LEGACY_USDT,
             adapter=_spot_adapter("spot_legacy_usdt"))
