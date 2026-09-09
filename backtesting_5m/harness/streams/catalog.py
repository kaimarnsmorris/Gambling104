"""The standard streams, in one readable screen.

An investigation registers its own by calling `register()` before `backtest()`,
and shadows any of these by registering the same name.
"""
from harness import paths
from harness.streams.registry import register
from harness.streams.spec import Stream, TimeKind


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
    """Register the standard streams. Idempotent."""
    #: PRE-GRIDDED. The book and spot panels are already keyed on
    #: (open_ts, t_ms) and are already on the decision grid. They are
    #: registered for discovery and documentation; `load_episodes` reads them
    #: by its existing path and never routes them through `grid_stream`.
    register("book", paths.PANEL, adapter=Stream(
        name="book", time_col="t_ms", time_kind=TimeKind.RECEIPT,
        time_unit="ms", causal=True, pre_gridded=True))

    register("spot", paths.SPOT, adapter=Stream(
        name="spot", time_col="t_ms", time_kind=TimeKind.RECEIPT,
        time_unit="ms", causal=True, pre_gridded=True))

    #: `px_first_recv_ns`, NOT `oracle_ms`. The oracle's own stamp runs ~1.5 s
    #: ahead of arrival on this vantage, so aligning on it is lookahead.
    register("chainlink", paths.RTDS_BTC, adapter=Stream(
        name="chainlink", time_col="px_first_recv_ns",
        time_kind=TimeKind.RECEIPT, time_unit="ns", causal=True))

    #: BTC/USD. The default. Venue mid less the capture's usdt_basis; sits
    #: +$4.50 (sd 7.44) against the Chainlink oracle, versus +$43.17 (sd
    #: 16.36) raw. The raw bias was also time-varying -- day means drifting
    #: from roughly $40 to roughly $10 -- which is why no fitted intercept
    #: could ever have absorbed it; the panel had to be rebuilt, not
    #: regressed away.
    register("spot_usd", paths.SPOT, adapter=_spot_adapter("spot_usd"))

    #: BTC/USD, rebuilt over the oracle overlap 2026-08-17..08-21 so a
    #: basis-learning fair block has a Chainlink line for the whole window.
    register("spot_oracle_window", paths.SPOT_ORACLE_WINDOW,
             adapter=_spot_adapter("spot_oracle_window"))

    #: BTC/USDT, UNCORRECTED -- roughly +$56 against the BTC/USD oracle these
    #: markets settle on. panel_100ms carries no usdt_basis column, so no
    #: correction is possible here. Usable ONLY with a fair block that learns
    #: the basis against the oracle itself. A USD-assuming consumer is $56
    #: wrong; the name says usdt for that reason.
    register("spot_london_usdt", paths.SPOT_LONDON_USDT,
             adapter=_spot_adapter("spot_london_usdt"))

    #: Superseded BTC/USDT panel. Registered only so a historical run folder
    #: can be reproduced against the data it actually used. Do not build on
    #: it. Its bias against the Chainlink oracle was +$43.17 (sd 16.36) --
    #: versus +$4.50 (sd 7.44) on the corrected panel -- and time-varying
    #: (day means roughly $40 to roughly $10), which is why it had to be
    #: rebuilt rather than regressed away.
    register("spot_legacy_usdt", paths.SPOT_LEGACY_USDT,
             adapter=_spot_adapter("spot_legacy_usdt"))
