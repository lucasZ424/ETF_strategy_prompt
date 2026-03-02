"""Interfaces and core data structures for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Protocol, Sequence


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
class BacktestConfig:
    """Backtest simulation configuration."""

    initial_cash: float = 1_000_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 3.0
    max_gross_leverage: float = 1.0


@dataclass
class BacktestResult:
    """Container for simulation outputs."""

    equity_curve: List[float]
    returns: List[float]
    trades: List[Trade]
    metadata: Dict[str, float]


class Strategy(Protocol):
    """Maps forecasts to target weights at each rebalance timestamp."""

    def target_weights(self, as_of: datetime, forecasts: Sequence[Forecast]) -> Dict[str, float]:
        ...


class BacktestEngine(Protocol):
    """Runs the backtest loop with market bars and forecast inputs."""

    def run(
        self,
        bars: Sequence[MarketBar],
        forecasts: Sequence[Forecast],
        strategy: Strategy,
        config: BacktestConfig,
    ) -> BacktestResult:
        ...


def bps_to_decimal(value_bps: float) -> float:
    """Converts basis points to decimal representation."""

    return value_bps / 10_000.0
