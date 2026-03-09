"""Performance metrics for backtest results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np

TRADING_DAYS_PER_YEAR = 244  # A-share market


@dataclass
class BacktestMetrics:
    """Summary statistics for a backtest run."""

    initial_cash: float
    final_value: float
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    win_rate_pct: float
    total_trades: int
    total_fees: float
    total_slippage: float
    n_trading_days: int

    def to_dict(self):
        return asdict(self)


def compute_backtest_metrics(
    equity_curve: List[Tuple],
    returns: List[float],
    metadata: dict,
    risk_free_rate: float = 0.02,
) -> BacktestMetrics:
    """Compute performance metrics from backtest results."""
    initial = metadata["initial_cash"]
    final = metadata.get("final_value", initial)
    n_days = metadata.get("n_trading_days", 0)

    if not equity_curve or n_days == 0:
        return BacktestMetrics(
            initial_cash=initial, final_value=initial,
            total_return_pct=0.0, cagr_pct=0.0, sharpe_ratio=0.0,
            max_drawdown_pct=0.0, max_drawdown_duration_days=0,
            win_rate_pct=0.0, total_trades=int(metadata.get("total_trades", 0)),
            total_fees=metadata.get("total_fees", 0.0),
            total_slippage=metadata.get("total_slippage", 0.0),
            n_trading_days=0,
        )

    total_return = (final / initial - 1) * 100

    # CAGR
    years = n_days / TRADING_DAYS_PER_YEAR
    if years > 0 and final > 0:
        cagr = ((final / initial) ** (1 / years) - 1) * 100
    else:
        cagr = 0.0

    # Sharpe ratio (annualized)
    ret_arr = np.array(returns)
    if len(ret_arr) > 1:
        daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        excess = ret_arr - daily_rf
        sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        sharpe = 0.0

    # Max drawdown
    equity_values = [v for _, v in equity_curve]
    peak = equity_values[0]
    max_dd = 0.0
    dd_start = 0
    max_dd_duration = 0
    current_dd_start = 0

    for i, v in enumerate(equity_values):
        if v >= peak:
            peak = v
            duration = i - current_dd_start
            if duration > max_dd_duration and max_dd > 0:
                max_dd_duration = duration
            current_dd_start = i
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # Check final drawdown duration
    if equity_values[-1] < peak:
        final_duration = len(equity_values) - 1 - current_dd_start
        if final_duration > max_dd_duration:
            max_dd_duration = final_duration

    # Win rate
    if len(ret_arr) > 0:
        win_rate = float(np.mean(ret_arr > 0)) * 100
    else:
        win_rate = 0.0

    return BacktestMetrics(
        initial_cash=initial,
        final_value=round(final, 2),
        total_return_pct=round(total_return, 4),
        cagr_pct=round(cagr, 4),
        sharpe_ratio=round(sharpe, 4),
        max_drawdown_pct=round(max_dd * 100, 4),
        max_drawdown_duration_days=max_dd_duration,
        win_rate_pct=round(win_rate, 2),
        total_trades=int(metadata["total_trades"]),
        total_fees=round(metadata["total_fees"], 2),
        total_slippage=round(metadata["total_slippage"], 2),
        n_trading_days=n_days,
    )
