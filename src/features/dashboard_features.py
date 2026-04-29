"""Dashboard-specific feature builder for price-ratio prediction.

Design goals:
1. All features are normalised / stationary — the model predicts
   scale-invariant price ratios (close_{t+H} / close_t ≈ 1.0).
2. Prefer dynamics: returns, MA/EMA ratios, volatility, candle shape,
   volume state.  No absolute-price anchors.
3. Avoid direct price-path level lags that leak absolute scale into
   a scale-invariant target.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """OLS slope of series over a rolling window (normalised by window)."""

    def _slope(arr):
        n = len(arr)
        if n < 2 or np.isnan(arr).any():
            return np.nan
        x = np.arange(n, dtype=float)
        x -= x.mean()
        return np.dot(x, arr) / np.dot(x, x)

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


def build_dashboard_features(
    china_df: pd.DataFrame,
    cross_market_aligned: pd.DataFrame,
) -> pd.DataFrame:
    """Build dashboard features for raw close-price prediction.

    Parameters
    ----------
    china_df : Cleaned China ETF DataFrame with
               [date, symbol, open, high, low, close, volume].
    cross_market_aligned : Cross-market features aligned to China dates
                           (spy_ret_lag1, qqq_ret_lag1, etc.)

    Returns
    -------
    DataFrame [date, symbol, close, <features>], sorted by [date, symbol],
    warmup NaNs dropped.
    """
    df = china_df[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    g = df.groupby("symbol")
    eps = 1e-8

    # === Block 1: Return dynamics ===
    df["log_close"] = np.log(df["close"])

    df["ret1_close"] = g["log_close"].diff(1)
    df["ret3_close"] = g["log_close"].diff(3)
    df["ret5_close"] = g["log_close"].diff(5)
    df["ret10_close"] = g["log_close"].diff(10)

    df["ret1_lag1"] = g["ret1_close"].shift(1)
    df["ret1_lag2"] = g["ret1_close"].shift(2)
    df["ret1_lag3"] = g["ret1_close"].shift(3)
    df["ret1_lag5"] = g["ret1_close"].shift(5)
    df["ret3_lag1"] = g["ret3_close"].shift(1)
    df["ret5_lag1"] = g["ret5_close"].shift(1)

    df["ret_accel_1v5"] = df["ret1_close"] - (df["ret5_close"] / 5.0)
    df["ret_accel_3v10"] = (df["ret3_close"] / 3.0) - (df["ret10_close"] / 10.0)

    # === Block 2: Trend-shape ===
    df["_ma5"] = g["close"].transform(lambda x: x.rolling(5, min_periods=5).mean())
    df["_ma10"] = g["close"].transform(lambda x: x.rolling(10, min_periods=10).mean())
    df["_ma20"] = g["close"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    df["_ema5"] = g["close"].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df["_ema20"] = g["close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())

    df["close_over_ma5"] = df["close"] / (df["_ma5"] + eps)
    df["close_over_ma10"] = df["close"] / (df["_ma10"] + eps)
    df["close_over_ma20"] = df["close"] / (df["_ma20"] + eps)
    df["ema5_over_ema20"] = df["_ema5"] / (df["_ema20"] + eps)
    df["ma5_over_ma20"] = df["_ma5"] / (df["_ma20"] + eps)

    df["slope5_close"] = g["log_close"].transform(lambda x: _rolling_slope(x, 5))
    df["slope10_close"] = g["log_close"].transform(lambda x: _rolling_slope(x, 10))

    # === Block 3: Volatility + candle shape ===
    df["rv5_close"] = g["ret1_close"].transform(lambda x: x.rolling(5, min_periods=5).std())
    df["rv10_close"] = g["ret1_close"].transform(lambda x: x.rolling(10, min_periods=10).std())
    rv20 = g["ret1_close"].transform(lambda x: x.rolling(20, min_periods=20).std())
    df["rv5_over_rv20"] = df["rv5_close"] / (rv20 + eps)

    # ATR(14) / close
    df["_tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            np.abs(df["high"] - g["close"].shift(1)),
            np.abs(df["low"] - g["close"].shift(1)),
        ),
    )
    df["atr14_over_close"] = (
        g["_tr"].transform(lambda x: x.rolling(14, min_periods=14).mean()) / (df["close"] + eps)
    )
    df["hl_range_close"] = (df["high"] - df["low"]) / (df["close"] + eps)
    df["gap_oc_prev_close"] = np.log(df["open"] / (g["close"].shift(1) + eps))

    df["oc_body_pct"] = (df["close"] - df["open"]) / (df["close"] + eps)
    df["upper_shadow_pct"] = (
        df["high"] - df[["open", "close"]].max(axis=1)
    ) / (df["close"] + eps)
    df["lower_shadow_pct"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    ) / (df["close"] + eps)

    # === Block 4: Volume state ===
    vol_ma20 = g["volume"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    vol_std20 = g["volume"].transform(lambda x: x.rolling(20, min_periods=20).std())
    df["vol_z20"] = (df["volume"] - vol_ma20) / (vol_std20 + eps)
    df["vol_ma_ratio20"] = df["volume"] / (vol_ma20 + eps)
    df["vol_chg1"] = np.log((df["volume"] + 1.0) / (g["volume"].shift(1) + 1.0))

    # === Block 5: Cross-market context ===
    df = df.merge(cross_market_aligned, on="date", how="left")

    # === Block 6: Calendar (cyclical) ===
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek  # 0=Mon, 4=Fri
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["dow_sin"] = np.sin(2.0 * np.pi * df["dow"] / 5.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * df["dow"] / 5.0)
    df["month_sin"] = np.sin(2.0 * np.pi * (df["month"] - 1.0) / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * (df["month"] - 1.0) / 12.0)

    # === Cleanup: drop temp columns, identify feature columns ===
    temp_cols = ["_ma5", "_ma10", "_ma20", "_ema5", "_ema20", "_tr", "log_close"]
    non_feature = {"date", "symbol", "open", "high", "low", "close", "volume"} | set(temp_cols)
    feature_cols = sorted([c for c in df.columns if c not in non_feature])

    df = df.drop(columns=[c for c in temp_cols if c in df.columns])

    # Drop warmup NaN rows
    pre_drop = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    dropped = pre_drop - len(df)

    # Keep date, symbol, close (needed downstream), and all features
    keep_cols = ["date", "symbol", "close"] + feature_cols
    result = df[keep_cols].sort_values(["date", "symbol"]).reset_index(drop=True)

    logger.info(
        "Dashboard features: %d rows (%d dropped), %d features: %s",
        len(result), dropped, len(feature_cols), feature_cols,
    )

    return result
