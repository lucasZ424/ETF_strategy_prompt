"""Concrete strategy implementations for backtesting."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Sequence

from .interface import Forecast, Signal

_SIGNAL_MAP = {"long": Signal.LONG, "short": Signal.SHORT, "flat": Signal.FLAT}


class EqualWeightSignalStrategy:
    """Equal-weight allocation across active (non-FLAT) signals.

    LONG signals get positive weight, SHORT signals get negative weight.
    Total gross exposure is capped at max_gross_leverage (default 1.0).
    """

    def __init__(self, signal_lookup: Dict[tuple, Signal], max_gross_leverage: float = 1.0):
        self._signal_lookup = signal_lookup
        self._max_leverage = max_gross_leverage

    def target_weights(self, as_of: datetime, forecasts: Sequence[Forecast]) -> Dict[str, float]:
        longs = []
        shorts = []

        for f in forecasts:
            key = (as_of.date() if hasattr(as_of, "date") else as_of, f.symbol)
            sig = self._signal_lookup.get(key, Signal.FLAT)
            if sig == Signal.LONG:
                longs.append(f.symbol)
            elif sig == Signal.SHORT:
                shorts.append(f.symbol)

        weights: Dict[str, float] = {}
        n_active = len(longs) + len(shorts)
        if n_active == 0:
            return weights

        per_position = self._max_leverage / n_active
        for sym in longs:
            weights[sym] = per_position
        for sym in shorts:
            weights[sym] = -per_position

        return weights
