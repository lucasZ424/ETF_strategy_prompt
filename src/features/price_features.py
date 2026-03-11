"""Trade-branch feature engineering — adj-close based (blocks 1-8).

All features on row T use only data from T-1 and earlier (no lookahead).
All closing-price features use adj_close per the trade-branch convention.

Produces features for blocks:
  1. Core returns (ret1_adj, ret3_adj, ret5_adj, ret10_adj, ret20_adj)
  2. Mean reversion (zscore5/10/20_adj, adj_over_ma10, adj_over_ma20)
  3. Trend / normalized momentum (ret10_over_rv10, ret20_over_rv20, ema/ma ratios, slopes)
  4. Volatility extras (abs_ret1_adj, gap_open_prev_adj)
  5. Volume (vol_z20, vol_ma_ratio20, log_volume)
  6. Relative strength (rel10_spy, rel10_ieur)
  7. Cross-sectional ranking (zscore_ret10d_cross, rank_volume_cross)

Gate-specific features (block 9) are in gate_features.py.
Regime date-level features (block 10) are in regime_features.py.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core returns (adj-close based, lagged by 1)
# ---------------------------------------------------------------------------

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """ret{N}_adj on row T = log(adj_close_{T-1} / adj_close_{T-1-N}).

    Produces: ret1_adj, ret3_adj, ret5_adj, ret10_adj, ret20_adj
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    lagged = df.groupby("symbol")["adj_close"].shift(1)

    for n in [1, 3, 5, 10, 20]:
        lagged_n = df.groupby("symbol")["adj_close"].shift(1 + n)
        df[f"ret{n}_adj"] = np.log(lagged / lagged_n)

    return df


# ---------------------------------------------------------------------------
# Mean reversion (adj-close based)
# ---------------------------------------------------------------------------

def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Horizon-aligned mean reversion features.

    Produces: zscore5_adj, zscore10_adj, zscore20_adj, adj_over_ma10, adj_over_ma20
    Uses adj_close lagged by 1 to avoid lookahead.
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    lagged = df.groupby("symbol")["adj_close"].shift(1)

    # Z-scores at 5, 10, 20 windows
    for w in [5, 10, 20]:
        ma = df.groupby("symbol")["adj_close"].transform(
            lambda s, _w=w: s.shift(1).rolling(_w, min_periods=_w).mean()
        )
        std = df.groupby("symbol")["adj_close"].transform(
            lambda s, _w=w: s.shift(1).rolling(_w, min_periods=_w).std()
        )
        df[f"zscore{w}_adj"] = (lagged - ma) / std

        # Store MA for reuse (adj_over_ma and regime)
        df[f"_ma{w}_adj"] = ma

    # Distance-to-MA ratios
    df["adj_over_ma10"] = lagged / df["_ma10_adj"]
    df["adj_over_ma20"] = lagged / df["_ma20_adj"]

    return df


# ---------------------------------------------------------------------------
# Trend / normalized momentum
# ---------------------------------------------------------------------------

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trend features using adj-close dynamics.

    Requires: ret10_adj, ret20_adj from add_return_features
    Requires: rv10_adj, rv20_adj from volatility module

    Produces: ret10_over_rv10, ret20_over_rv20, ema10_over_ema20_adj,
              ma10_over_ma20_adj, slope10_adj, slope5_adj
    """
    eps = 1e-8
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Vol-scaled momentum (requires rv10_adj, rv20_adj from volatility)
    if "rv10_adj" in df.columns and "ret10_adj" in df.columns:
        df["ret10_over_rv10"] = df["ret10_adj"] / (df["rv10_adj"] + eps)
    if "rv20_adj" in df.columns and "ret20_adj" in df.columns:
        df["ret20_over_rv20"] = df["ret20_adj"] / (df["rv20_adj"] + eps)

    # EMA ratio: ema10 / ema20 of lagged adj_close
    ema10 = df.groupby("symbol")["adj_close"].transform(
        lambda s: s.shift(1).ewm(span=10, adjust=False).mean()
    )
    ema20 = df.groupby("symbol")["adj_close"].transform(
        lambda s: s.shift(1).ewm(span=20, adjust=False).mean()
    )
    df["ema10_over_ema20_adj"] = ema10 / ema20

    # MA ratio: ma10 / ma20 (uses precomputed _ma10_adj, _ma20_adj from mean reversion)
    if "_ma10_adj" in df.columns and "_ma20_adj" in df.columns:
        df["ma10_over_ma20_adj"] = df["_ma10_adj"] / df["_ma20_adj"]

    # Rolling slope of log(adj_close) over 10 and 5 bars
    log_adj = np.log(df.groupby("symbol")["adj_close"].shift(1))
    df["slope10_adj"] = log_adj.groupby(df["symbol"]).transform(
        _rolling_slope, window=10
    )
    df["slope5_adj"] = log_adj.groupby(df["symbol"]).transform(
        _rolling_slope, window=5
    )

    return df


def _rolling_slope(s: pd.Series, window: int = 10) -> pd.Series:
    """OLS slope of a series over a rolling window."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(vals: np.ndarray) -> float:
        if len(vals) < window or np.any(np.isnan(vals)):
            return np.nan
        y_mean = vals.mean()
        return float(np.sum((x - x_mean) * (vals - y_mean)) / x_var)

    return s.rolling(window, min_periods=window).apply(_slope, raw=True)


# ---------------------------------------------------------------------------
# Volatility extras (beyond the core rv/atr in volatility.py)
# ---------------------------------------------------------------------------

def add_volatility_extras(df: pd.DataFrame) -> pd.DataFrame:
    """Additional volatility features: abs_ret1_adj, gap_open_prev_adj.

    Produces: abs_ret1_adj, gap_open_prev_adj
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    if "ret1_adj" in df.columns:
        df["abs_ret1_adj"] = df["ret1_adj"].abs()

    # Overnight gap: log(Open_t / AdjClose_{t-1}), lagged by 1
    lagged_open = df.groupby("symbol")["open"].shift(1)
    lagged_adj_prev = df.groupby("symbol")["adj_close"].shift(2)
    df["gap_open_prev_adj"] = np.log(lagged_open / lagged_adj_prev)

    return df


# ---------------------------------------------------------------------------
# Volume features
# ---------------------------------------------------------------------------

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """vol_z20, vol_ma_ratio20, log_volume — lagged by 1.

    Produces: vol_z20, vol_ma_ratio20, log_volume
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    lagged_vol = df.groupby("symbol")["volume"].shift(1)

    ma20 = df.groupby("symbol")["volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean()
    )
    std20 = df.groupby("symbol")["volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).std()
    )

    df["vol_z20"] = (lagged_vol - ma20) / std20
    df["vol_ma_ratio20"] = lagged_vol / ma20
    df["log_volume"] = np.log(lagged_vol + 1)

    return df


# ---------------------------------------------------------------------------
# Relative strength vs cross-market
# ---------------------------------------------------------------------------

def add_relative_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """rel10_spy and rel10_ieur — horizon-aligned relative returns.

    Requires: ret10_adj, spy_ret10d_lag1, ieur_ret10d_lag1 already merged.
    """
    if "spy_ret10d_lag1" in df.columns and "ret10_adj" in df.columns:
        df["rel10_spy"] = df["ret10_adj"] - df["spy_ret10d_lag1"]
    if "ieur_ret10d_lag1" in df.columns and "ret10_adj" in df.columns:
        df["rel10_ieur"] = df["ret10_adj"] - df["ieur_ret10d_lag1"]
    return df


# ---------------------------------------------------------------------------
# Cross-sectional ranking features (same-date across core ETFs)
# ---------------------------------------------------------------------------

def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-ETF relative features computed per date.

    zscore_ret10d_cross: z-score of ret10_adj across core ETFs on same date.
    rank_volume_cross:   percentile rank of vol_z20 across core ETFs on same date.
    """
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    if "ret10_adj" in df.columns:
        mean_r = df.groupby("date")["ret10_adj"].transform("mean")
        std_r = df.groupby("date")["ret10_adj"].transform("std")
        df["zscore_ret10d_cross"] = (df["ret10_adj"] - mean_r) / std_r.replace(0, np.nan)

    if "vol_z20" in df.columns:
        df["rank_volume_cross"] = df.groupby("date")["vol_z20"].transform(
            lambda s: s.rank(pct=True)
        )

    return df


