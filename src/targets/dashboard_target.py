"""Dashboard targets: 1d / 3d / 5d forward price ratios.

    y_ratio_Hd = close_{t+H} / close_t

The model trains on scale-invariant ratios (centred around 1.0).
At inference the predicted ratio is multiplied by current close to
recover the raw price for Grafana display.

Output columns: [date, symbol, close, y_ratio_1d, y_ratio_3d, y_ratio_5d]
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
    """Build multi-horizon price-ratio targets.

    Parameters
    ----------
    backbone_df : DataFrame with [date, symbol, close] at minimum.
    horizons : list of forward horizons in trading days (default [1, 3, 5]).

    Returns
    -------
    DataFrame [date, symbol, close, y_ratio_1d, y_ratio_3d, y_ratio_5d],
    sorted by [date, symbol]. Rows where any target is NaN are dropped.
    """
    if horizons is None:
        horizons = [1, 3, 5]

    df = backbone_df[["date", "symbol", "close"]].copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    ratio_cols = []
    for h in horizons:
        col = f"y_ratio_{h}d"
        future_close = df.groupby("symbol")["close"].shift(-h)
        df[col] = future_close / df["close"]
        ratio_cols.append(col)

    pre_drop = len(df)
    df = df.dropna(subset=ratio_cols).reset_index(drop=True)
    dropped = pre_drop - len(df)

    result = df[["date", "symbol", "close"] + ratio_cols].sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)

    logger.info(
        "Dashboard targets (horizons=%s): %d rows, dropped %d tail rows.",
        horizons, len(result), dropped,
    )
    for col in ratio_cols:
        logger.info(
            "  %s: mean=%.6f, std=%.6f",
            col, result[col].mean(), result[col].std(),
        )

    return result
