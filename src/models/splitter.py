"""Chronological train/val/test split on unique dates."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitResult:
    """Container for the three data subsets."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_dates: pd.Index
    val_dates: pd.Index
    test_dates: pd.Index


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    date_col: str = "date",
) -> SplitResult:
    """Split dataframe by unique dates chronologically.

    All ETF rows for a given date land in the same subset.
    Ratios are applied to the count of unique dates, not rows.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame")

    unique_dates = sorted(df[date_col].unique())
    n = len(unique_dates)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    train_df = df[df[date_col].isin(set(train_dates))].reset_index(drop=True)
    val_df = df[df[date_col].isin(set(val_dates))].reset_index(drop=True)
    test_df = df[df[date_col].isin(set(test_dates))].reset_index(drop=True)

    logger.info(
        "Split: %d train dates (%d rows), %d val dates (%d rows), %d test dates (%d rows)",
        len(train_dates), len(train_df),
        len(val_dates), len(val_df),
        len(test_dates), len(test_df),
    )

    return SplitResult(
        train=train_df,
        val=val_df,
        test=test_df,
        train_dates=pd.Index(train_dates),
        val_dates=pd.Index(val_dates),
        test_dates=pd.Index(test_dates),
    )
