"""Dashboard targets: 1d / 3d / 5d forward raw close prices.

    y_close_Hd = close_{t+H}

Output columns: [date, symbol, close, y_close_1d, y_close_3d, y_close_5d]
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def build_dashboard_targets(
    backbone_df: pd.DataFrame,
    horizons: List[int] | None = None,
) -> pd.DataFrame:
    """Build multi-horizon raw close-price targets.

    Parameters
    ----------
    backbone_df : DataFrame with [date, symbol, close] at minimum.
    horizons : list of forward horizons in trading days (default [1, 3, 5]).

    Returns
    -------
    DataFrame [date, symbol, close, y_close_1d, y_close_3d, y_close_5d],
    sorted by [date, symbol]. Rows where any target is NaN are dropped.
    """
    if horizons is None:
        horizons = [1, 3, 5]

    df = backbone_df[["date", "symbol", "close"]].copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    close_cols = []
    for h in horizons:
        col = f"y_close_{h}d"
        df[col] = df.groupby("symbol")["close"].shift(-h)
        close_cols.append(col)

    pre_drop = len(df)
    df = df.dropna(subset=close_cols).reset_index(drop=True)
    dropped = pre_drop - len(df)

    result = df[["date", "symbol", "close"] + close_cols].sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)

    logger.info(
        "Dashboard targets (horizons=%s): %d rows, dropped %d tail rows.",
        horizons, len(result), dropped,
    )
    for col in close_cols:
        logger.info(
            "  %s: mean=%.4f, std=%.4f",
            col, result[col].mean(), result[col].std(),
        )

    return result
