"""Backtesting harness for the BTC up/down 5 m book."""
__version__ = "0.1.0"

from harness.core.api import backtest, BacktestResult
from harness.core.config import ExecConfig, Output, QuoteParams, Sample
from harness.core.grid import (Arm, default_workers, grid_search,
                               linear_grid)
from harness.streams import Stream, TimeKind, register, registered, resolve

__all__ = ["Arm", "backtest", "BacktestResult", "default_workers",
           "ExecConfig", "grid_search", "linear_grid", "Output",
           "QuoteParams", "Sample", "Stream", "TimeKind", "register",
           "registered", "resolve", "__version__"]
