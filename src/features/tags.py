"""Feature usage-tag registry for the alpha / gate / regime pipeline.

Each feature is tagged with its eligible downstream consumers:
  alpha  = candidate input to the alpha regressor
  gate   = candidate input to the tradability gate
  regime = candidate input to the regime dataset (may be aggregated to date-level)

The builder computes all features once, then filters by tag to emit
three separate datasets.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tag definitions: feature_name -> frozenset of tags
# ---------------------------------------------------------------------------

# Block 1: Core returns
_CORE_RETURNS = {
    "ret1_adj":  frozenset({"alpha", "gate", "regime"}),
    "ret3_adj":  frozenset({"alpha", "gate", "regime"}),
    "ret5_adj":  frozenset({"alpha", "gate", "regime"}),
    "ret10_adj": frozenset({"alpha", "gate", "regime"}),
    "ret20_adj": frozenset({"alpha", "gate", "regime"}),
}

# Block 2: Mean reversion
_MEAN_REVERSION = {
    "zscore5_adj":    frozenset({"alpha", "gate", "regime"}),
    "zscore10_adj":   frozenset({"alpha", "gate", "regime"}),
    "zscore20_adj":   frozenset({"alpha", "gate", "regime"}),
    "adj_over_ma10":  frozenset({"alpha", "gate", "regime"}),
    "adj_over_ma20":  frozenset({"alpha", "gate", "regime"}),
}

# Block 3: Trend / normalized momentum
_TREND = {
    "ret10_over_rv10":      frozenset({"alpha", "gate", "regime"}),
    "ret20_over_rv20":      frozenset({"alpha", "gate", "regime"}),
    "ema10_over_ema20_adj": frozenset({"alpha", "gate", "regime"}),
    "ma10_over_ma20_adj":   frozenset({"alpha", "gate", "regime"}),
    "slope10_adj":          frozenset({"alpha", "gate", "regime"}),
    "slope5_adj":           frozenset({"alpha", "gate", "regime"}),
}

# Block 4: Volatility / risk
_VOLATILITY = {
    "rv5_adj":           frozenset({"alpha", "gate", "regime"}),
    "rv10_adj":          frozenset({"alpha", "gate", "regime"}),
    "rv20_adj":          frozenset({"alpha", "gate", "regime"}),
    "atr14_over_adj":    frozenset({"alpha", "gate", "regime"}),
    "hl_range_adjproxy": frozenset({"alpha", "gate", "regime"}),
    "abs_ret1_adj":      frozenset({"alpha", "gate", "regime"}),
    "gap_open_prev_adj": frozenset({"alpha", "gate", "regime"}),
}

# Block 5: Volume
_VOLUME = {
    "vol_z20":        frozenset({"alpha", "gate", "regime"}),
    "vol_ma_ratio20": frozenset({"alpha", "gate", "regime"}),
    "log_volume":     frozenset({"alpha", "gate", "regime"}),
}

# Block 6: Cross-market and macro
_CROSS_MARKET = {
    "spy_ret_lag1":     frozenset({"alpha", "gate", "regime"}),
    "qqq_ret_lag1":     frozenset({"alpha", "gate", "regime"}),
    "ieur_ret_lag1":    frozenset({"alpha", "gate", "regime"}),
    "spy_ret10d_lag1":  frozenset({"alpha", "gate", "regime"}),
    "ieur_ret10d_lag1": frozenset({"alpha", "gate", "regime"}),
    "vix_chg_lag1":     frozenset({"alpha", "gate", "regime"}),
    "us10y_chg_lag1":   frozenset({"alpha", "gate", "regime"}),
    "dxy_ret_lag1":     frozenset({"alpha", "gate", "regime"}),
}

# Block 7: Relative strength
_RELATIVE_STRENGTH = {
    "rel10_spy":  frozenset({"alpha", "gate", "regime"}),
    "rel10_ieur": frozenset({"alpha", "gate", "regime"}),
}

# Block 8: Cross-sectional ETF
_CROSS_SECTIONAL = {
    "zscore_ret10d_cross": frozenset({"alpha", "gate"}),
    "rank_volume_cross":   frozenset({"alpha", "gate"}),
}

# Block 9: Gate-specific tradability overlay
_GATE_OVERLAY = {
    "downside_semivol_10": frozenset({"gate", "regime"}),
    "max_drawdown_10":     frozenset({"gate", "regime"}),
    "gap_freq_10":         frozenset({"gate"}),
    "whipsaw_5":           frozenset({"gate"}),
    "cost_buffer_10":      frozenset({"gate"}),
}

# Block 10: Regime-only market-state (date-level, computed separately)
_REGIME_ONLY = {
    "universe_ret10_mean":       frozenset({"regime"}),
    "universe_ret10_dispersion": frozenset({"regime"}),
    "universe_breadth_ma10":     frozenset({"regime"}),
    "universe_breadth_pos10":    frozenset({"regime"}),
    "universe_vol20_mean":       frozenset({"regime"}),
    "universe_corr20":           frozenset({"regime"}),
    "universe_volume_stress":    frozenset({"regime"}),
    "macro_stress_score":        frozenset({"regime"}),
    "cross_market_trend_score":  frozenset({"regime"}),
}

# ---------------------------------------------------------------------------
# Combined registry
# ---------------------------------------------------------------------------

FEATURE_TAGS: dict[str, frozenset[str]] = {
    **_CORE_RETURNS,
    **_MEAN_REVERSION,
    **_TREND,
    **_VOLATILITY,
    **_VOLUME,
    **_CROSS_MARKET,
    **_RELATIVE_STRENGTH,
    **_CROSS_SECTIONAL,
    **_GATE_OVERLAY,
    **_REGIME_ONLY,
}


def features_for(tag: str) -> list[str]:
    """Return sorted list of feature names that carry the given tag."""
    return sorted(f for f, tags in FEATURE_TAGS.items() if tag in tags)


def tags_for(feature: str) -> frozenset[str]:
    """Return the usage tags for a given feature name."""
    return FEATURE_TAGS.get(feature, frozenset())
