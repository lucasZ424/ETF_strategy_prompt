"""Shared trading-branch feature builder with 3 dataset views.

Computes the full feature backbone once, then emits:
  1. alpha dataset — ETF-level [date, symbol, alpha-tagged features]
  2. gate dataset  — ETF-level [date, symbol, gate-tagged features]
  3. regime dataset — date-level [date, regime-tagged features]

Dependency chain (10-day target horizon):
  1  Returns       → ret1/3/5/10/20_adj
  2  Volatility    → rv5/10/20_adj, atr14_over_adj, hl_range_adjproxy
  3  Mean reversion → zscore5/10/20_adj, adj_over_ma10/20
  4  Trend         → ret10_over_rv10, ret20_over_rv20, ema/ma ratios, slopes
  5  Vol extras    → abs_ret1_adj, gap_open_prev_adj
  6  Volume        → vol_z20, vol_ma_ratio20, log_volume
  7  Cross-market merge
  8  Relative strength → rel10_spy, rel10_ieur
  9  Cross-sectional   → zscore_ret10d_cross, rank_volume_cross
 10  Gate overlay      → downside_semivol_10, max_drawdown_10, etc.
 11  Drop warmup NaN rows
 12  Regime date-level aggregation
 13  Two-stage feature selection (per dataset view)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import pandas as pd

from src.config import FeatureSelectionConfig
from src.features.gate_features import add_gate_features
from src.features.price_features import (
    add_return_features,
    add_mean_reversion_features,
    add_trend_features,
    add_volatility_extras,
    add_volume_features,
    add_relative_strength_features,
    add_cross_sectional_features,
)
from src.features.regime_features import build_regime_features
from src.features.screening import run_feature_selection
from src.features.tags import FEATURE_TAGS, features_for
from src.features.volatility import add_volatility_features

logger = logging.getLogger(__name__)

# Columns that are NOT features (metadata, raw OHLCV, internal temps)
_NON_FEATURE_COLS = {
    "date", "symbol", "target",
    "open", "high", "low", "close", "volume", "adj_close",
    "_ma5_adj", "_ma10_adj", "_ma20_adj",
}


@dataclass
class TradeDatasets:
    """Container for the 3 downstream dataset views."""

    alpha: pd.DataFrame      # ETF-level [date, symbol, alpha features]
    gate: pd.DataFrame       # ETF-level [date, symbol, gate features]
    regime: pd.DataFrame     # date-level [date, regime features]
    backbone: pd.DataFrame   # full ETF-level backbone (for target builders)


def build_trade_features(
    china_df: pd.DataFrame,
    cross_market_aligned: pd.DataFrame,
    feature_selection: FeatureSelectionConfig | None = None,
    seed: int = 42,
    cost_threshold: float = 0.0016,
    gap_threshold: float = 0.005,
) -> TradeDatasets:
    """Build the full trading feature backbone and emit 3 dataset views.

    Parameters
    ----------
    china_df : cleaned China ETF DataFrame (pooled, all symbols).
    cross_market_aligned : cross-market features aligned to China dates.
    feature_selection : two-stage selection config. None = skip selection.
    seed : random seed for XGBoost importance stage.
    cost_threshold : round-trip cost for gate cost_buffer_10.

    Returns
    -------
    TradeDatasets with .alpha, .gate, .regime, .backbone
    """
    df = china_df.copy()

    # --- Shared backbone computation (steps 1-10) ---

    # 1. Returns
    logger.info("Step 1/10: Computing return features...")
    df = add_return_features(df)

    # 2. Volatility (must come before trend)
    logger.info("Step 2/10: Computing volatility features...")
    df = add_volatility_features(df)

    # 3. Mean reversion (produces _ma10_adj, _ma20_adj)
    logger.info("Step 3/10: Computing mean-reversion features...")
    df = add_mean_reversion_features(df)

    # 4. Trend (depends on returns + volatility + MAs)
    logger.info("Step 4/10: Computing trend features...")
    df = add_trend_features(df)

    # 5. Volatility extras (depends on ret1_adj)
    logger.info("Step 5/10: Computing volatility extras...")
    df = add_volatility_extras(df)

    # 6. Volume
    logger.info("Step 6/10: Computing volume features...")
    df = add_volume_features(df)

    # 7. Cross-market merge
    logger.info("Step 7/10: Merging cross-market features...")
    df = df.merge(cross_market_aligned, on="date", how="left")

    # 8. Relative strength (depends on ret10_adj + cross-market 10d)
    logger.info("Step 8/10: Computing relative strength features...")
    df = add_relative_strength_features(df)

    # 9. Cross-sectional ranking (depends on ret10_adj, vol_z20)
    logger.info("Step 9/10: Computing cross-sectional features...")
    df = add_cross_sectional_features(df)

    # 10. Gate overlay features
    logger.info("Step 10/10: Computing gate overlay features...")
    df = add_gate_features(df, cost_threshold=cost_threshold, gap_threshold=gap_threshold)

    # --- Identify all feature columns and drop warmup ---
    all_feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]

    pre_drop = len(df)
    df = df.dropna(subset=all_feature_cols).reset_index(drop=True)
    logger.info(
        "Dropped %d warmup rows. Remaining: %d rows, %d features.",
        pre_drop - len(df), len(df), len(all_feature_cols),
    )

    # Clean up internal temp columns
    for tmp_col in ["_ma5_adj", "_ma10_adj", "_ma20_adj"]:
        if tmp_col in df.columns:
            df = df.drop(columns=[tmp_col])

    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Recompute available feature cols after cleanup
    available = [c for c in df.columns if c not in _NON_FEATURE_COLS]

    # --- Build regime date-level features ---
    logger.info("Building regime date-level features...")
    regime_df = build_regime_features(df)

    # --- Feature selection per view ---
    alpha_cols = [c for c in features_for("alpha") if c in available]
    gate_cols = [c for c in features_for("gate") if c in available]

    if feature_selection is not None:
        proxy_target = "ret10_adj"
        if proxy_target in df.columns:
            df_sel = df.copy()
            df_sel["target"] = df_sel[proxy_target]

            # Alpha selection
            alpha_selected, alpha_meta = run_feature_selection(
                df_sel, alpha_cols, target_col="target",
                correlation_threshold=feature_selection.correlation_threshold,
                importance_top_k=feature_selection.importance_top_k,
                protected_prefixes=list(feature_selection.protected_prefixes),
                seed=seed,
            )
            logger.info(
                "Alpha selection: %d → %d features. Pearson dropped: %s",
                len(alpha_cols), len(alpha_selected),
                alpha_meta.get("pearson_dropped", []),
            )
            alpha_cols = alpha_selected

            # Gate selection (separate pass)
            gate_selected, gate_meta = run_feature_selection(
                df_sel, gate_cols, target_col="target",
                correlation_threshold=feature_selection.correlation_threshold,
                importance_top_k=feature_selection.importance_top_k,
                protected_prefixes=list(feature_selection.protected_prefixes),
                seed=seed,
            )
            logger.info(
                "Gate selection: %d → %d features. Pearson dropped: %s",
                len(gate_cols), len(gate_selected),
                gate_meta.get("pearson_dropped", []),
            )
            gate_cols = gate_selected
        else:
            logger.warning("No proxy target '%s' for feature selection, skipping.", proxy_target)

    # --- Emit 3 dataset views ---
    alpha_df = df[["date", "symbol"] + sorted(alpha_cols)].copy()
    gate_df = df[["date", "symbol"] + sorted(gate_cols)].copy()

    logger.info(
        "Dataset views: alpha=%d cols, gate=%d cols, regime=%d cols.",
        len(alpha_cols), len(gate_cols), len(regime_df.columns) - 1,
    )

    return TradeDatasets(
        alpha=alpha_df,
        gate=gate_df,
        regime=regime_df,
        backbone=df,
    )
