"""Threshold-based signal generation using backtest.interface types."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.backtest.interface import Forecast, Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalConfig:
    """Result of threshold optimization."""

    long_threshold: float
    short_threshold: float
    criterion_name: str
    criterion_value: float


def optimize_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: List[float],
    criterion: str = "directional_accuracy",
) -> SignalConfig:
    """Find the symmetric threshold that maximizes the chosen criterion.

    Only evaluates non-FLAT predictions (|pred| > threshold).

    Parameters
    ----------
    criterion : str
        "directional_accuracy" — fraction of active predictions with correct sign.
        "edge" — mean(y_true * sign(y_pred)) for active predictions (PnL proxy).
    """
    best_val = -np.inf
    best_t = thresholds[0]

    for t in thresholds:
        long_mask = y_pred > t
        short_mask = y_pred < -t
        active_mask = long_mask | short_mask

        n_active = int(active_mask.sum())
        if n_active < 10:
            continue

        signals = np.where(long_mask, 1, np.where(short_mask, -1, 0))
        active_signals = signals[active_mask]
        active_true = y_true[active_mask]

        if criterion == "directional_accuracy":
            correct = np.sign(active_signals) == np.sign(active_true)
            val = float(np.mean(correct))
        elif criterion == "edge":
            val = float(np.mean(active_true * active_signals))
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        if val > best_val:
            best_val = val
            best_t = t

    logger.info("Best threshold=%.5f, %s=%.4f", best_t, criterion, best_val)

    return SignalConfig(
        long_threshold=best_t,
        short_threshold=-best_t,
        criterion_name=criterion,
        criterion_value=best_val,
    )


def predict_to_signal(predicted_return: float, config: SignalConfig) -> Signal:
    """Map a single predicted return to a Signal."""
    if predicted_return > config.long_threshold:
        return Signal.LONG
    elif predicted_return < config.short_threshold:
        return Signal.SHORT
    return Signal.FLAT


def generate_forecasts(
    df: pd.DataFrame,
    predictions: np.ndarray,
    signal_config: SignalConfig,
) -> Tuple[List[Forecast], pd.DataFrame]:
    """Generate Forecast objects and a summary DataFrame from predictions.

    Returns
    -------
    forecasts : list of Forecast
    summary_df : DataFrame with [date, symbol, predicted_return, signal, target]
    """
    forecasts: List[Forecast] = []
    records = []

    for i in range(len(df)):
        row = df.iloc[i]
        pred = float(predictions[i])
        sig = predict_to_signal(pred, signal_config)

        forecast = Forecast(
            timestamp=pd.Timestamp(row["date"]).to_pydatetime(),
            symbol=row["symbol"],
            predicted_return=pred,
        )
        forecasts.append(forecast)

        records.append({
            "date": row["date"],
            "symbol": row["symbol"],
            "predicted_return": pred,
            "signal": sig.value,
            "target": row.get("target", np.nan),
        })

    summary_df = pd.DataFrame(records)
    logger.info(
        "Signals: LONG=%d, SHORT=%d, FLAT=%d",
        (summary_df["signal"] == "long").sum(),
        (summary_df["signal"] == "short").sum(),
        (summary_df["signal"] == "flat").sum(),
    )

    return forecasts, summary_df
