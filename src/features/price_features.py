"""Price, volume, gap, and calendar features — all OHLCV-only, no TA-Lib.

Every feature on row T uses only data from T-1 and earlier (shift-then-roll pattern).

Consolidates: momentum, mean_reversion, volume, gap, calendar, volatility_scaling.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``target`` = log(open_T / adj_close_{T-1}). First row per symbol → NaN."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["target"] = np.log(df["open"] / df.groupby("symbol")["adj_close"].shift(1))
    return df


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def add_momentum_features(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """ret_{n}d on row T = log(adj_close_{T-1} / adj_close_{T-1-n})."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    for n in periods:
        df[f"ret_{n}d"] = df.groupby("symbol")["adj_close"].transform(
            lambda s, _n=n: np.log(s.shift(1) / s.shift(1 + _n))
        )
    return df


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------

def add_mean_reversion_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """price_minus_ma, zscore, price_over_ma — all using adj_close lagged by 1."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    lagged = df.groupby("symbol")["adj_close"].shift(1)
    for w in windows:
        ma = df.groupby("symbol")["adj_close"].transform(
            lambda s, _w=w: s.shift(1).rolling(_w, min_periods=_w).mean()
        )
        std = df.groupby("symbol")["adj_close"].transform(
            lambda s, _w=w: s.shift(1).rolling(_w, min_periods=_w).std()
        )
        df[f"price_minus_ma_{w}"] = lagged - ma
        df[f"zscore_{w}"] = (lagged - ma) / std
        df[f"price_over_ma_{w}"] = lagged / ma
    return df


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def add_volume_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """vol_change, vol_zscore, vol_ma_ratio — lagged by 1."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    lagged_vol = df.groupby("symbol")["volume"].shift(1)
    prev2_vol = df.groupby("symbol")["volume"].shift(2).replace(0, float("nan"))
    df["vol_change"] = (lagged_vol / prev2_vol).replace([float("inf"), float("-inf")], float("nan")).fillna(1.0)
    for w in windows:
        ma = df.groupby("symbol")["volume"].transform(
            lambda s, _w=w: s.shift(1).rolling(_w, min_periods=_w).mean()
        )
        std = df.groupby("symbol")["volume"].transform(
            lambda s, _w=w: s.shift(1).rolling(_w, min_periods=_w).std()
        )
        df[f"vol_zscore_{w}"] = (lagged_vol - ma) / std
        df[f"vol_ma_ratio_{w}"] = lagged_vol / ma
    return df


# ---------------------------------------------------------------------------
# Gap
# ---------------------------------------------------------------------------

def add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """prev_gap and mean_gap_3d — lagged by 1."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    gap = np.log(df["open"] / df.groupby("symbol")["adj_close"].shift(1))
    df["prev_gap"] = gap.groupby(df["symbol"]).shift(1)
    df["mean_gap_3d"] = gap.groupby(df["symbol"]).transform(
        lambda s: s.shift(1).rolling(3, min_periods=3).mean()
    )
    return df


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """dow, month, quarter — no lookahead, trading date is known in advance."""
    dt = pd.to_datetime(df["date"])
    df["dow"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    return df


# ---------------------------------------------------------------------------
# Volatility scaling (depends on ret_1d, atr_*, rolling_std_*)
# ---------------------------------------------------------------------------

def add_volatility_scaling_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """ret_over_atr and ret_over_std — requires momentum + volatility features."""
    for w in windows:
        if f"atr_{w}" in df.columns:
            df[f"ret_over_atr_{w}"] = df["ret_1d"] / df[f"atr_{w}"].replace(0, float("nan"))
        if f"rolling_std_{w}" in df.columns:
            df[f"ret_over_std_{w}"] = df["ret_1d"] / df[f"rolling_std_{w}"].replace(0, float("nan"))
    return df
