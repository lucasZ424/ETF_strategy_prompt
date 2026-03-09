"""Simple daily backtest engine aligned to gap-return target.

Execution model (matching open_to_prev_close_log_return target):
- Signal for day T is known before T's open (features use data up to T-1).
- Entry: previous close (adj_close_{T-1}) — conceptually, position taken EOD T-1.
- Exit: day T's open — the gap is realized.
- PnL per trade = open_T / adj_close_{T-1} - 1  (matches model target).
- Slippage applied to both entry (close) and exit (open).
- Fees deducted from cash on each trade.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .interface import (
    BacktestResult,
    Forecast,
    MarketBar,
    Trade,
    bps_to_decimal,
)
from src.config import BacktestConfig
from .strategies import EqualWeightSignalStrategy

logger = logging.getLogger(__name__)


def _group_by_date(items, date_fn):
    """Group items by date, returning dict[date, list]."""
    groups = defaultdict(list)
    for item in items:
        d = date_fn(item)
        groups[d].append(item)
    return groups


def _bar_date(b):
    return b.timestamp.date() if isinstance(b.timestamp, datetime) else b.timestamp


def _forecast_date(f):
    return f.timestamp.date() if isinstance(f.timestamp, datetime) else f.timestamp


class SimpleBacktestEngine:
    """Gap-return backtest engine.

    For each forecast date T with a signal:
      1. Enter at close_{T-1} + slippage  (previous day's close).
      2. Exit at open_T - slippage         (day T's open).
      3. PnL captures exactly the overnight gap the model predicts.

    Each day's positions are fully closed — no carry between days.
    """

    def run(
        self,
        bars: Sequence[MarketBar],
        forecasts: Sequence[Forecast],
        strategy: EqualWeightSignalStrategy,
        config: BacktestConfig,
    ) -> BacktestResult:
        initial_cash = config.initial_cash
        fee_rate = bps_to_decimal(config.fee_bps)
        slip_rate = bps_to_decimal(config.slippage_bps)

        bars_by_date = _group_by_date(bars, _bar_date)
        forecasts_by_date = _group_by_date(forecasts, _forecast_date)

        all_bar_dates = sorted(bars_by_date.keys())
        forecast_dates = sorted(forecasts_by_date.keys())

        if not forecast_dates or len(all_bar_dates) < 2:
            logger.warning("Insufficient data for backtest.")
            return BacktestResult(
                equity_curve=[], returns=[], trades=[],
                metadata={"initial_cash": initial_cash, "final_value": initial_cash,
                          "total_trades": 0, "total_fees": 0.0, "total_slippage": 0.0,
                          "fee_bps": config.fee_bps, "slippage_bps": config.slippage_bps,
                          "n_trading_days": 0},
            )

        # Map each bar date to its previous bar date
        prev_bar_map = {}
        for i in range(1, len(all_bar_dates)):
            prev_bar_map[all_bar_dates[i]] = all_bar_dates[i - 1]

        cash = initial_cash
        all_trades: List[Trade] = []
        equity_curve: List[Tuple[date, float]] = []
        total_fees = 0.0
        total_slippage = 0.0

        for T in forecast_dates:
            if T not in prev_bar_map:
                continue
            T_prev = prev_bar_map[T]

            if T not in bars_by_date or T_prev not in bars_by_date:
                continue

            bars_T = {b.symbol: b for b in bars_by_date[T]}
            bars_prev = {b.symbol: b for b in bars_by_date[T_prev]}
            day_forecasts = forecasts_by_date[T]

            # Strategy decides weights based on forecasts for day T
            dt = datetime.combine(T, datetime.min.time())
            target_weights = strategy.target_weights(dt, day_forecasts)

            port_value = cash
            day_pnl = 0.0
            day_fees = 0.0
            day_slippage = 0.0

            for sym, w in target_weights.items():
                if sym not in bars_prev or sym not in bars_T:
                    continue
                if abs(w) < 1e-9:
                    continue

                prev_close = bars_prev[sym].close
                today_open = bars_T[sym].open

                if prev_close <= 0 or today_open <= 0:
                    continue

                # Position size (notional)
                notional = w * port_value
                direction = 1 if notional > 0 else -1

                # Entry at prev close + slippage
                entry_price = prev_close * (1 + direction * slip_rate)
                # Exit at today's open - slippage
                exit_price = today_open * (1 - direction * slip_rate)

                quantity = notional / entry_price

                # Trade PnL
                trade_pnl = quantity * (exit_price - entry_price)
                entry_fee = abs(notional) * fee_rate
                exit_fee = abs(quantity * exit_price) * fee_rate
                trade_fee = entry_fee + exit_fee
                trade_slip = (abs(quantity) * prev_close * slip_rate
                              + abs(quantity) * today_open * slip_rate)

                day_pnl += trade_pnl - trade_fee
                day_fees += trade_fee
                day_slippage += trade_slip

                all_trades.append(Trade(
                    timestamp=dt,
                    symbol=sym,
                    quantity=quantity,
                    price=entry_price,
                    fee=trade_fee,
                    slippage=trade_slip,
                ))

            cash += day_pnl
            total_fees += day_fees
            total_slippage += day_slippage
            equity_curve.append((T, cash))

        # Compute daily returns
        equity_values = [v for _, v in equity_curve]
        returns = []
        for i in range(1, len(equity_values)):
            if equity_values[i - 1] != 0:
                returns.append(equity_values[i] / equity_values[i - 1] - 1)
            else:
                returns.append(0.0)

        final_value = equity_values[-1] if equity_values else initial_cash
        metadata = {
            "initial_cash": initial_cash,
            "final_value": final_value,
            "total_trades": len(all_trades),
            "total_fees": total_fees,
            "total_slippage": total_slippage,
            "fee_bps": config.fee_bps,
            "slippage_bps": config.slippage_bps,
            "n_trading_days": len(equity_curve),
        }
        logger.info(
            "Backtest complete: %d days, %d trades, final=%.2f",
            len(equity_curve), len(all_trades), final_value,
        )

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            trades=all_trades,
            metadata=metadata,
        )


def load_market_bars(raw_dir, symbols: List[str], start_date=None, end_date=None) -> List[MarketBar]:
    """Load raw CSVs into MarketBar objects.

    Loads one extra day before start_date so prev-close is available for the
    first forecast date.
    """
    from pathlib import Path

    raw_path = Path(raw_dir) / "china_etfs"
    bars = []

    for sym in symbols:
        csv_path = raw_path / f"{sym}.csv"
        if not csv_path.exists():
            logger.warning("No data file for %s at %s", sym, csv_path)
            continue

        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # Include one extra day before start_date for prev-close entry
        if start_date:
            ts = pd.Timestamp(start_date)
            first_valid_idx = df["Date"].searchsorted(ts)
            load_from = max(0, first_valid_idx - 1)
            df = df.iloc[load_from:]
        if end_date:
            df = df[df["Date"] <= pd.Timestamp(end_date)]

        for _, row in df.iterrows():
            bars.append(MarketBar(
                timestamp=row["Date"].to_pydatetime(),
                symbol=sym,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            ))

    logger.info("Loaded %d market bars for %d symbols", len(bars), len(symbols))
    return bars
