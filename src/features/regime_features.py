"""Regime-only market-state features (block 10).

These are date-level features computed by aggregating ETF-level data across
the universe on each date. They are inputs to the regime model which selects
optimizer policy (aggressive / balanced / defensive).

Produces (one row per date):
  universe_ret10_mean       — mean(ret10_adj) across ETFs
  universe_ret10_dispersion — std(ret10_adj) across ETFs
  universe_breadth_ma10     — fraction of ETFs with adj_over_ma10 > 1
  universe_breadth_pos10    — fraction of ETFs with ret10_adj > 0
  universe_vol20_mean       — mean(rv20_adj) across ETFs
  universe_corr20           — rolling average pairwise correlation
  universe_volume_stress    — mean(vol_z20) across ETFs
  macro_stress_score        — composite of vix/dxy/us10y
  cross_market_trend_score  — composite of SPY/QQQ/IEUR lagged trends
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_regime_features(
    backbone_df: pd.DataFrame,
    corr_lookback: int = 20,
) -> pd.DataFrame:
    """Build date-level regime features from the ETF-level backbone.

    Parameters
    ----------
    backbone_df : DataFrame with [date, symbol, <ETF-level features>].
                  Must already have ret10_adj, adj_over_ma10, rv20_adj,
                  vol_z20, ret1_adj, and cross-market/macro columns.
    corr_lookback : rolling window for pairwise correlation.

    Returns
    -------
    DataFrame keyed by [date] with regime-only columns.
    """
    df = backbone_df.sort_values(["date", "symbol"]).copy()
    dates = sorted(df["date"].unique())

    regime_rows = []

    # Pre-compute per-date cross-sectional aggregations
    for dt in dates:
        day = df[df["date"] == dt]
        row = {"date": dt}

        # Universe 10d return statistics
        if "ret10_adj" in day.columns:
            r10 = day["ret10_adj"].dropna()
            row["universe_ret10_mean"] = r10.mean() if len(r10) > 0 else np.nan
            row["universe_ret10_dispersion"] = r10.std() if len(r10) > 1 else np.nan
            row["universe_breadth_pos10"] = (r10 > 0).mean() if len(r10) > 0 else np.nan

        # Breadth above MA10
        if "adj_over_ma10" in day.columns:
            am = day["adj_over_ma10"].dropna()
            row["universe_breadth_ma10"] = (am > 1).mean() if len(am) > 0 else np.nan

        # Average vol 20
        if "rv20_adj" in day.columns:
            rv = day["rv20_adj"].dropna()
            row["universe_vol20_mean"] = rv.mean() if len(rv) > 0 else np.nan

        # Volume stress
        if "vol_z20" in day.columns:
            vz = day["vol_z20"].dropna()
            row["universe_volume_stress"] = vz.mean() if len(vz) > 0 else np.nan

        regime_rows.append(row)

    regime_df = pd.DataFrame(regime_rows)

    # Rolling average pairwise correlation across ETF returns
    regime_df = _add_universe_corr(df, regime_df, lookback=corr_lookback)

    # Macro stress score (composite of cross-market risk indicators)
    regime_df = _add_macro_stress_score(df, regime_df)

    # Cross-market trend score
    regime_df = _add_cross_market_trend_score(df, regime_df)

    regime_df = regime_df.sort_values("date").reset_index(drop=True)
    logger.info("Built %d regime date-rows with %d columns.", len(regime_df), len(regime_df.columns))
    return regime_df


def _add_universe_corr(
    etf_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """Rolling average pairwise correlation of ret1_adj across ETFs."""
    if "ret1_adj" not in etf_df.columns:
        regime_df["universe_corr20"] = np.nan
        return regime_df

    # Pivot to wide: dates x symbols
    pivot = etf_df.pivot_table(index="date", columns="symbol", values="ret1_adj")
    pivot = pivot.sort_index()

    # Rolling pairwise correlation mean
    corr_series = pivot.rolling(lookback, min_periods=lookback).apply(
        lambda _: np.nan, raw=True  # placeholder
    ).iloc[:, 0]  # just for index

    # Compute properly: rolling corr matrix mean off-diagonal
    avg_corr = []
    symbols = pivot.columns.tolist()
    n_sym = len(symbols)

    for i in range(len(pivot)):
        if i < lookback - 1:
            avg_corr.append(np.nan)
            continue
        window = pivot.iloc[i - lookback + 1: i + 1]
        if window.dropna(axis=1, how="all").shape[1] < 2:
            avg_corr.append(np.nan)
            continue
        cm = window.corr()
        # Mean of upper triangle (off-diagonal)
        mask = np.triu(np.ones(cm.shape, dtype=bool), k=1)
        vals = cm.values[mask]
        vals = vals[np.isfinite(vals)]
        avg_corr.append(float(np.mean(vals)) if len(vals) > 0 else np.nan)

    corr_result = pd.DataFrame({"date": pivot.index, "universe_corr20": avg_corr})
    regime_df = regime_df.merge(corr_result, on="date", how="left")
    return regime_df


def _add_macro_stress_score(
    etf_df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """Composite macro stress: standardized sum of vix_chg, dxy_ret, us10y_chg."""
    # Extract one row per date (cross-market features are date-level)
    macro_cols = ["vix_chg_lag1", "dxy_ret_lag1", "us10y_chg_lag1"]
    available = [c for c in macro_cols if c in etf_df.columns]

    if not available:
        regime_df["macro_stress_score"] = np.nan
        return regime_df

    # Take first row per date (macro features are identical across ETFs)
    daily = etf_df.groupby("date")[available].first().reset_index()

    # Z-score each component then sum
    for col in available:
        mean_val = daily[col].mean()
        std_val = daily[col].std()
        if std_val > 0:
            daily[f"_z_{col}"] = (daily[col] - mean_val) / std_val
        else:
            daily[f"_z_{col}"] = 0.0

    z_cols = [f"_z_{c}" for c in available]
    daily["macro_stress_score"] = daily[z_cols].sum(axis=1)

    regime_df = regime_df.merge(daily[["date", "macro_stress_score"]], on="date", how="left")
    return regime_df


def _add_cross_market_trend_score(
    etf_df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """Composite cross-market trend: standardized sum of SPY/QQQ/IEUR 10d returns."""
    trend_cols = ["spy_ret10d_lag1", "qqq_ret_lag1", "ieur_ret10d_lag1"]
    available = [c for c in trend_cols if c in etf_df.columns]

    if not available:
        regime_df["cross_market_trend_score"] = np.nan
        return regime_df

    daily = etf_df.groupby("date")[available].first().reset_index()

    for col in available:
        mean_val = daily[col].mean()
        std_val = daily[col].std()
        if std_val > 0:
            daily[f"_z_{col}"] = (daily[col] - mean_val) / std_val
        else:
            daily[f"_z_{col}"] = 0.0

    z_cols = [f"_z_{c}" for c in available]
    daily["cross_market_trend_score"] = daily[z_cols].sum(axis=1)

    regime_df = regime_df.merge(daily[["date", "cross_market_trend_score"]], on="date", how="left")
    return regime_df
