"""Clean raw OHLCV data: drop NaN rows for China, forward-fill for cross-market."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def clean_china_etfs(df: pd.DataFrame) -> pd.DataFrame:
    """Clean China ETF data — DROP any rows with NaN or invalid OHLCV.

    Never impute core Chinese market data.
    """

    initial = len(df)

    # Drop NaN in OHLCV
    nan_mask = df[_OHLCV_COLS].isna().any(axis=1)
    if (n := nan_mask.sum()) > 0:
        logger.warning("Dropping %d rows with NaN in OHLCV", n)
        df = df[~nan_mask].copy()

    # Sanity checks
    bad = (
        (df["open"] <= 0)
        | (df["close"] <= 0)
        | (df["high"] < df["low"])
        | (df["volume"] < 0)
    )
    if (n := bad.sum()) > 0:
        logger.warning("Dropping %d rows with invalid OHLCV values", n)
        df = df[~bad].copy()

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info("China cleaning: %d → %d rows", initial, len(df))
    return df


def clean_cross_market(df: pd.DataFrame) -> pd.DataFrame:
    """Clean cross-market data — forward-fill NaN per symbol (fills holiday gaps)."""

    initial_nans = int(df[_OHLCV_COLS].isna().sum().sum())
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df[_OHLCV_COLS] = df.groupby("symbol")[_OHLCV_COLS].ffill()

    remaining = int(df[_OHLCV_COLS].isna().sum().sum())
    if remaining > 0:
        df = df.dropna(subset=_OHLCV_COLS).reset_index(drop=True)

    logger.info(
        "Cross-market cleaning: filled %d NaN, dropped %d remaining",
        initial_nans - remaining,
        remaining,
    )
    return df
