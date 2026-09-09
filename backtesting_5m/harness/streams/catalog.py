"""The standard streams, in one readable screen.

An investigation registers its own by calling `register()` before `backtest()`,
and shadows any of these by registering the same name.
"""
from harness import paths
from harness.streams.registry import register
from harness.streams.spec import Stream, TimeKind


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
