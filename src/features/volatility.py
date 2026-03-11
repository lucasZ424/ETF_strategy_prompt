"""Volatility features for the v3 trade branch.

All features on row T use only data from T-1 and earlier.
Closing-price dynamics use adj_close per trade-branch convention.

Produces:
  rv5_adj   — rolling std of 1d adj log return over 5 bars (lagged)
  rv10_adj  — rolling std over 10 bars (lagged)
  rv20_adj  — rolling std over 20 bars (lagged)
  atr14_over_adj — ATR(14) / adj_close (lagged, normalized)
  hl_range_adjproxy — (high - low) / adj_close (lagged)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import talib

logger = logging.getLogger(__name__)


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all v3 volatility features per symbol (all lagged by 1 day).

    Parameters
    ----------
    df : DataFrame with columns: symbol, date, adj_close, high, low, open, volume.
    """

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # 1d adj-close log return, then lag by 1
    df["_log_ret"] = df.groupby("symbol")["adj_close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    df["_lr_lag"] = df.groupby("symbol")["_log_ret"].shift(1)

    # Rolling std of lagged 1d returns at 5, 10, 20 windows
    for w in [5, 10, 20]:
        df[f"rv{w}_adj"] = df.groupby("symbol")["_lr_lag"].transform(
            lambda s, _w=w: s.rolling(_w, min_periods=_w).std()
        )

    # ATR(14) via TA-Lib: compute per symbol on raw OHLC, then shift by 1 and normalize
    atr_parts = []
    for _, g in df.groupby("symbol"):
        atr_vals = talib.ATR(
            g["high"].values, g["low"].values,
            g["adj_close"].values, timeperiod=14,
        )
        atr_parts.append(pd.Series(atr_vals, index=g.index))
    df["_atr14_raw"] = pd.concat(atr_parts).sort_index()

    lagged_adj = df.groupby("symbol")["adj_close"].shift(1)
    df["atr14_over_adj"] = df.groupby("symbol")["_atr14_raw"].shift(1) / lagged_adj

    # High-low range normalized by adj_close (lagged by 1)
    lagged_high = df.groupby("symbol")["high"].shift(1)
    lagged_low = df.groupby("symbol")["low"].shift(1)
    df["hl_range_adjproxy"] = (lagged_high - lagged_low) / lagged_adj

    # Cleanup temporary columns
    df = df.drop(columns=["_log_ret", "_lr_lag", "_atr14_raw"])

    return df
