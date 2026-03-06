"""Backtest package."""

from .engine import SimpleBacktestEngine, load_market_bars
from .interface import (
    BacktestResult,
    Forecast,
    MarketBar,
    Signal,
    Trade,
    bps_to_decimal,
)
from .strategies import EqualWeightSignalStrategy

__all__ = [
    "BacktestResult",
    "EqualWeightSignalStrategy",
    "Forecast",
    "MarketBar",
    "Signal",
    "SimpleBacktestEngine",
    "Trade",
    "bps_to_decimal",
    "load_market_bars",
]
