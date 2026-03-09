"""Tests for backtest engine, strategy, and metrics."""

from __future__ import annotations

import sys
from datetime import datetime, date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.interface import Forecast, MarketBar, Signal
from src.backtest.strategies import EqualWeightSignalStrategy
from src.backtest.engine import SimpleBacktestEngine
from src.config import BacktestConfig
from src.evaluation.backtest_metrics import compute_backtest_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _bar(d, sym, open_, close):
    return MarketBar(
        timestamp=datetime(2024, 1, d),
        symbol=sym,
        open=open_, high=max(open_, close),
        low=min(open_, close), close=close,
        volume=1_000_000,
    )


def _forecast(d, sym, pred):
    return Forecast(timestamp=datetime(2024, 1, d), symbol=sym, predicted_return=pred)


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestEqualWeightSignalStrategy:
    def test_long_only(self):
        lookup = {
            (date(2024, 1, 2), "A"): Signal.LONG,
            (date(2024, 1, 2), "B"): Signal.FLAT,
        }
        strat = EqualWeightSignalStrategy(lookup, max_gross_leverage=1.0)
        weights = strat.target_weights(
            datetime(2024, 1, 2),
            [_forecast(2, "A", 0.01), _forecast(2, "B", 0.001)],
        )
        assert "A" in weights
        assert weights["A"] == pytest.approx(1.0)
        assert "B" not in weights

    def test_long_short(self):
        lookup = {
            (date(2024, 1, 2), "A"): Signal.LONG,
            (date(2024, 1, 2), "B"): Signal.SHORT,
        }
        strat = EqualWeightSignalStrategy(lookup, max_gross_leverage=1.0)
        weights = strat.target_weights(
            datetime(2024, 1, 2),
            [_forecast(2, "A", 0.02), _forecast(2, "B", -0.02)],
        )
        assert weights["A"] == pytest.approx(0.5)
        assert weights["B"] == pytest.approx(-0.5)

    def test_all_flat_returns_empty(self):
        lookup = {(date(2024, 1, 2), "A"): Signal.FLAT}
        strat = EqualWeightSignalStrategy(lookup)
        weights = strat.target_weights(
            datetime(2024, 1, 2), [_forecast(2, "A", 0.0)],
        )
        assert weights == {}


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

class TestSimpleBacktestEngine:
    def _make_config(self, **kwargs):
        defaults = dict(initial_cash=100_000, fee_bps=0.0, slippage_bps=0.0,
                        max_gross_leverage=1.0, risk_free_rate=0.02)
        defaults.update(kwargs)
        return BacktestConfig(**defaults)

    def test_positive_gap_makes_money(self):
        """If prev close=10 and today open=11 (10% gap), LONG should profit."""
        bars = [
            _bar(1, "A", 9.5, 10.0),   # day 1: close=10
            _bar(2, "A", 11.0, 10.5),   # day 2: open=11 (gap = 11/10 - 1 = 10%)
        ]
        forecasts = [_forecast(2, "A", 0.10)]
        lookup = {(date(2024, 1, 2), "A"): Signal.LONG}
        strat = EqualWeightSignalStrategy(lookup, max_gross_leverage=1.0)
        cfg = self._make_config(fee_bps=0.0, slippage_bps=0.0)

        result = SimpleBacktestEngine().run(bars, forecasts, strat, cfg)

        assert len(result.equity_curve) == 1
        final = result.equity_curve[0][1]
        assert final > 100_000  # should profit from 10% gap

    def test_negative_gap_loses_money_on_long(self):
        """If gap is negative and we're LONG, should lose."""
        bars = [
            _bar(1, "A", 9.5, 10.0),
            _bar(2, "A", 9.0, 9.5),   # gap = 9/10 - 1 = -10%
        ]
        forecasts = [_forecast(2, "A", 0.05)]
        lookup = {(date(2024, 1, 2), "A"): Signal.LONG}
        strat = EqualWeightSignalStrategy(lookup, max_gross_leverage=1.0)
        cfg = self._make_config()

        result = SimpleBacktestEngine().run(bars, forecasts, strat, cfg)

        final = result.equity_curve[0][1]
        assert final < 100_000

    def test_fees_reduce_pnl(self):
        """Same gap, but with fees should give less profit."""
        bars = [
            _bar(1, "A", 9.5, 10.0),
            _bar(2, "A", 11.0, 10.5),
        ]
        forecasts = [_forecast(2, "A", 0.10)]
        lookup = {(date(2024, 1, 2), "A"): Signal.LONG}
        strat = EqualWeightSignalStrategy(lookup, max_gross_leverage=1.0)

        no_fee = SimpleBacktestEngine().run(
            bars, forecasts, strat, self._make_config(fee_bps=0.0))
        with_fee = SimpleBacktestEngine().run(
            bars, forecasts, strat, self._make_config(fee_bps=50.0))

        assert with_fee.equity_curve[0][1] < no_fee.equity_curve[0][1]

    def test_empty_forecasts_returns_empty(self):
        bars = [_bar(1, "A", 10.0, 10.0)]
        result = SimpleBacktestEngine().run(bars, [], EqualWeightSignalStrategy({}), self._make_config())
        assert result.equity_curve == []
        assert result.metadata["final_value"] == 100_000


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestBacktestMetrics:
    def test_empty_equity(self):
        m = compute_backtest_metrics(
            equity_curve=[], returns=[],
            metadata={"initial_cash": 100_000, "final_value": 100_000,
                      "n_trading_days": 0, "total_trades": 0,
                      "total_fees": 0, "total_slippage": 0},
        )
        assert m.total_return_pct == 0.0
        assert m.sharpe_ratio == 0.0

    def test_positive_return(self):
        ec = [(date(2024, 1, i), 100_000 + i * 100) for i in range(1, 11)]
        rets = [100 / (100_000 + (i - 1) * 100) for i in range(1, 10)]
        m = compute_backtest_metrics(
            equity_curve=ec, returns=rets,
            metadata={"initial_cash": 100_000, "final_value": 101_000,
                      "n_trading_days": 10, "total_trades": 10,
                      "total_fees": 50, "total_slippage": 30},
        )
        assert m.total_return_pct > 0
        assert m.sharpe_ratio > 0
        assert m.max_drawdown_pct == 0.0  # monotonically increasing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
