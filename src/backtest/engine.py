"""Simple daily backtest engine."""

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
from .strategies import EqualWeightSignalStrategy

logger = logging.getLogger(__name__)


def _group_by_date(items, date_fn):
    """Group items by date, returning dict[date, list]."""
    groups = defaultdict(list)
    for item in items:
        d = date_fn(item)
        groups[d].append(item)
    return groups


class SimpleBacktestEngine:
    """Daily rebalance backtest engine.

    Execution model:
    - Each day, the strategy produces target weights from forecasts.
    - Trades execute at the day's Open price + slippage.
    - Fees are deducted from cash on each trade.
    - Portfolio is marked-to-market at Close prices.
    """

    def run(
        self,
        bars: Sequence[MarketBar],
        forecasts: Sequence[Forecast],
        strategy: EqualWeightSignalStrategy,
        initial_cash: float,
        fee_bps: float,
        slippage_bps: float,
    ) -> BacktestResult:
        fee_rate = bps_to_decimal(fee_bps)
        slip_rate = bps_to_decimal(slippage_bps)

        bars_by_date: Dict[date, List[MarketBar]] = _group_by_date(
            bars, lambda b: b.timestamp.date() if isinstance(b.timestamp, datetime) else b.timestamp
        )
        forecasts_by_date: Dict[date, List[Forecast]] = _group_by_date(
            forecasts, lambda f: f.timestamp.date() if isinstance(f.timestamp, datetime) else f.timestamp
        )

        sorted_dates = sorted(set(bars_by_date.keys()) & set(forecasts_by_date.keys()))
        if not sorted_dates:
            logger.warning("No overlapping dates between bars and forecasts.")
            return BacktestResult(
                equity_curve=[], returns=[], trades=[],
                metadata={"initial_cash": initial_cash},
            )

        cash = initial_cash
        positions: Dict[str, float] = {}  # symbol -> quantity
        all_trades: List[Trade] = []
        equity_curve: List[Tuple[date, float]] = []
        total_fees = 0.0
        total_slippage = 0.0

        for d in sorted_dates:
            day_bars = {b.symbol: b for b in bars_by_date[d]}
            day_forecasts = forecasts_by_date[d]

            # Compute target weights
            dt = datetime.combine(d, datetime.min.time())
            target_weights = strategy.target_weights(dt, day_forecasts)

            # Current portfolio value at open prices (for sizing)
            port_value = cash
            for sym, qty in positions.items():
                if sym in day_bars:
                    port_value += qty * day_bars[sym].open

            # Compute target positions (in quantity)
            target_qty: Dict[str, float] = {}
            for sym, w in target_weights.items():
                if sym in day_bars and day_bars[sym].open > 0:
                    target_qty[sym] = (w * port_value) / day_bars[sym].open

            # Determine trades needed
            current_syms = set(positions.keys()) | set(target_qty.keys())
            for sym in current_syms:
                cur = positions.get(sym, 0.0)
                tgt = target_qty.get(sym, 0.0)
                delta = tgt - cur

                if abs(delta) < 1e-6:
                    continue
                if sym not in day_bars:
                    continue

                bar = day_bars[sym]
                # Execute at open + slippage
                direction = 1 if delta > 0 else -1
                exec_price = bar.open * (1 + direction * slip_rate)
                trade_value = abs(delta * exec_price)
                fee = trade_value * fee_rate
                slippage_cost = abs(delta) * bar.open * slip_rate

                cash -= delta * exec_price + fee
                positions[sym] = positions.get(sym, 0.0) + delta
                total_fees += fee
                total_slippage += slippage_cost

                all_trades.append(Trade(
                    timestamp=dt,
                    symbol=sym,
                    quantity=delta,
                    price=exec_price,
                    fee=fee,
                    slippage=slippage_cost,
                ))

            # Clean near-zero positions
            positions = {s: q for s, q in positions.items() if abs(q) > 1e-9}

            # Mark-to-market at close
            port_value_close = cash
            for sym, qty in positions.items():
                if sym in day_bars:
                    port_value_close += qty * day_bars[sym].close

            equity_curve.append((d, port_value_close))

        # Compute daily returns
        equity_values = [v for _, v in equity_curve]
        returns = []
        for i in range(1, len(equity_values)):
            if equity_values[i - 1] != 0:
                returns.append(equity_values[i] / equity_values[i - 1] - 1)
            else:
                returns.append(0.0)

        metadata = {
            "initial_cash": initial_cash,
            "final_value": equity_values[-1] if equity_values else initial_cash,
            "total_trades": len(all_trades),
            "total_fees": total_fees,
            "total_slippage": total_slippage,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "n_trading_days": len(sorted_dates),
        }
        logger.info(
            "Backtest complete: %d days, %d trades, final=%.2f",
            len(sorted_dates), len(all_trades), metadata["final_value"],
        )

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            trades=all_trades,
            metadata=metadata,
        )


def load_market_bars(raw_dir, symbols: List[str], start_date=None, end_date=None) -> List[MarketBar]:
    """Load raw CSVs into MarketBar objects."""
    from pathlib import Path

    raw_path = Path(raw_dir) / "china_etfs"
    bars = []

    for sym in symbols:
        csv_path = raw_path / f"{sym}.csv"
        if not csv_path.exists():
            logger.warning("No data file for %s at %s", sym, csv_path)
            continue

        df = pd.read_csv(csv_path, parse_dates=["Date"])
        if start_date:
            df = df[df["Date"] >= pd.Timestamp(start_date)]
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
