"""Orchestrate feature computation in the correct dependency order."""

from __future__ import annotations

import logging
# from pathlib import Path
from typing import List

import pandas as pd

from src.features.price_features import (
    add_calendar_features,
    add_gap_features,
    add_mean_reversion_features,
    add_momentum_features,
    add_volatility_scaling_features,
    add_volume_features,
    compute_target,
)
from src.features.screening import select_top_k_features
from src.features.volatility import add_volatility_features

logger = logging.getLogger(__name__)

_NON_FEATURE_COLS = {"date", "symbol", "target", "open", "high", "low", "close", "volume", "adj_close"}


def build_all_features(
    china_df: pd.DataFrame,
    cross_market_aligned: pd.DataFrame,
    lookback_windows: List[int],
    top_k_features: int | None = None,
) -> pd.DataFrame:
    """Build all features and target in correct dependency order.

    Steps:
      1  Target        (open, adj_close)
      2  Momentum      (adj_close)         → ret_1d needed by steps 9–10
      3  Volatility    (adj_close, high, low) → rolling_std, atr needed by step 10
      4  Mean reversion (adj_close)
      5  Volume        (volume)
      6  Gap           (open, adj_close)
      7  Calendar      (date)
      8  Cross-market merge
      9  Relative returns (ret_1d − cross-market ETF returns)
     10  Volatility scaling (ret_1d / atr, ret_1d / rolling_std)
     11  Drop warmup NaN rows
     12  GRA-GRG feature screening (optional, if top_k_features is set)
    """
    df = china_df.copy()

    df = compute_target(df)
    df = add_momentum_features(df, periods=[1, 3, 5, 10])
    df = add_volatility_features(df, windows=lookback_windows)
    df = add_mean_reversion_features(df, windows=lookback_windows)
    df = add_volume_features(df, windows=lookback_windows)
    df = add_gap_features(df)
    df = add_calendar_features(df)

    df = df.merge(cross_market_aligned, on="date", how="left")

    # Relative returns: China 1d return minus each cross-market ETF return
    for xm in ["spy", "qqq", "ieur"]:
        ret_col = f"{xm}_ret"
        if ret_col in df.columns:
            df[f"rel_ret_{xm}"] = df["ret_1d"] - df[ret_col]

    df = add_volatility_scaling_features(df, windows=lookback_windows)

    # Drop warmup NaN rows (max_window + shifts)
    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    pre_drop = len(df)
    df = df.dropna(subset=["target"] + feature_cols).reset_index(drop=True)
    logger.info("Dropped %d warmup rows. Final: %d rows, %d features", pre_drop - len(df), len(df), len(feature_cols))

    # GRA-GRG feature screening (development phase)
    if top_k_features is not None:
        selected, grades = select_top_k_features(
            df, feature_cols, top_k=top_k_features,
        )
        feature_cols = selected

    # Stacked format: group by date, then symbol within each date
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    output_cols = ["date", "symbol"] + sorted(feature_cols) + ["target"]
    return df[[c for c in output_cols if c in df.columns]]
