"""Volatility features: squared return, abs return, rolling std, ATR, realised vol.

Every feature on row T uses only data from T-1 and earlier.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import talib


def add_volatility_features(
    df: pd.DataFrame,
    windows: List[int],
) -> pd.DataFrame:
    """Add volatility features per symbol (all lagged by 1 day)."""

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Adj-close-to-adj-close log return, then lag it by 1
    df["_log_ret"] = df.groupby("symbol")["adj_close"].transform(
        lambda s: np.log(s / s.shift(1))
    )
    df["_lr_lag"] = df.groupby("symbol")["_log_ret"].shift(1)

    df["ret_sq"] = df["_lr_lag"] ** 2
    df["ret_abs"] = df["_lr_lag"].abs()

    for w in windows:
        df[f"rolling_std_{w}"] = df.groupby("symbol")["_lr_lag"].transform(
            lambda s, _w=w: s.rolling(_w, min_periods=_w).std()
        )
        df[f"realized_vol_{w}"] = df.groupby("symbol")["_lr_lag"].transform(
            lambda s, _w=w: (s**2).rolling(_w, min_periods=_w).sum().pipe(np.sqrt)
        )

    # ATR via TA-Lib: compute on raw OHLC per symbol, then shift by 1
    for w in windows:
        atr_parts = []
        for _, g in df.groupby("symbol"):
            atr_vals = talib.ATR(
                g["high"].values, g["low"].values,
                g["adj_close"].values, timeperiod=w,
            )
            atr_parts.append(pd.Series(atr_vals, index=g.index))
        df[f"_atr_raw_{w}"] = pd.concat(atr_parts).sort_index()
        df[f"atr_{w}"] = df.groupby("symbol")[f"_atr_raw_{w}"].shift(1)
        df = df.drop(columns=[f"_atr_raw_{w}"])

    df = df.drop(columns=["_log_ret", "_lr_lag"])
    return df
