"""Alpha target: forward 10-day adjusted log return.

Definition:
    y_alpha_t = log(adj_close_{t+10} / adj_close_t)

Recomputed daily so the model sees a daily-updated 10-day opportunity
score rather than a blind fixed holding period.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_alpha_target(
    backbone_df: pd.DataFrame,
    horizon: int = 10,
) -> pd.DataFrame:
    """Build the forward adjusted log-return alpha target.

    Parameters
    ----------
    backbone_df : DataFrame with [date, symbol, adj_close] at minimum.
    horizon : forward return horizon in trading days (default 10).

    Returns
    -------
    DataFrame with columns [date, symbol, y_alpha], sorted by [date, symbol].
    Rows where the forward return cannot be computed (tail) are dropped.
    """
    df = backbone_df[["date", "symbol", "adj_close"]].copy()

    # Sort by [symbol, date] for correct per-symbol shift
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Forward log return per symbol
    df["_fwd_adj"] = df.groupby("symbol")["adj_close"].shift(-horizon)
    df["y_alpha"] = np.log(df["_fwd_adj"] / df["adj_close"])

    # Drop rows where forward return is unavailable (last `horizon` dates per symbol)
    pre_drop = len(df)
    df = df.dropna(subset=["y_alpha"]).reset_index(drop=True)
    dropped = pre_drop - len(df)

    # Re-sort to stacked [date, symbol] format
    result = df[["date", "symbol", "y_alpha"]].sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)

    logger.info(
        "Alpha target (horizon=%d): %d rows, dropped %d tail rows. "
        "y_alpha mean=%.4f, std=%.4f",
        horizon, len(result), dropped,
        result["y_alpha"].mean(), result["y_alpha"].std(),
    )

    return result
