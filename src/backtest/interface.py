"""Interfaces and core data structures for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Protocol, Sequence


class Signal(Enum):
    """Trading signal direction."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(frozen=True)
class MarketBar:
    """Single bar market data for one symbol."""

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Forecast:
    """Model output aligned to a timestamp and symbol."""

    timestamp: datetime
    symbol: str
    predicted_return: float


@dataclass
class Trade:
    """Executed order summary."""

    timestamp: datetime
    symbol: str
    quantity: float
    price: float
    fee: float
    slippage: float


@dataclass
class BacktestResult:
    """Container for simulation outputs."""

    equity_curve: List[Any]  # List of (date, float) tuples
    returns: List[float]
    trades: List[Trade]
    metadata: Dict[str, Any]


class Strategy(Protocol):
    """Maps forecasts to target weights at each rebalance timestamp."""

    def target_weights(self, as_of: datetime, forecasts: Sequence[Forecast]) -> Dict[str, float]:
        ...


def bps_to_decimal(value_bps: float) -> float:
    """Converts basis points to decimal representation."""

    return value_bps / 10_000.0
