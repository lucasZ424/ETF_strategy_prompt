"""CLI entry point: python scripts/build_targets.py [--config path/to/config.toml]

Builds all trading targets from the backbone parquet:
  1. y_alpha   — forward 10-day adjusted log return
  2. barrier   — triple-barrier ternary gate labels {-1, 0, +1}
  3. regime    — rule-based regime labels {0=defensive, 1=balanced, 2=aggressive}

Requires: run scripts/run_pipeline.py first to produce backbone.parquet
          and regime_features.parquet.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.targets.alpha_target import build_alpha_target  # noqa: E402
from src.targets.dashboard_target import build_dashboard_targets  # noqa: E402
from src.targets.regime_labels import build_regime_labels  # noqa: E402
from src.targets.triple_barrier import build_barrier_labels  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _class_distribution(series: pd.Series) -> dict:
    """Serialize class counts in a JSON-safe and type-stable form."""
    counts = series.value_counts().sort_index()
    items = [{"label": int(k), "count": int(v)} for k, v in counts.items()]
    return {
        # Backward-compatible map (JSON object keys are always strings)
        "distribution": {str(int(k)): int(v) for k, v in counts.items()},
        # Type-stable structure to avoid int-vs-str key mismatches
        "distribution_items": items,
        "distribution_labels": [it["label"] for it in items],
        "distribution_counts": [it["count"] for it in items],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build trading targets from backbone.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "china_open_universe_minimal.template.toml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = PROJECT_ROOT / config.processed_dir

    # --- Load inputs ---
    backbone_path = processed_dir / "backbone.parquet"
    regime_path = processed_dir / "regime_features.parquet"

    if not backbone_path.exists():
        logger.error("Backbone not found at %s. Run scripts/run_pipeline.py first.", backbone_path)
        sys.exit(1)
    if not regime_path.exists():
        logger.error("Regime features not found at %s. Run scripts/run_pipeline.py first.", regime_path)
        sys.exit(1)

    backbone_df = pd.read_parquet(backbone_path)
    regime_df = pd.read_parquet(regime_path)
    logger.info("Loaded backbone: %d rows, %d cols", len(backbone_df), len(backbone_df.columns))
    logger.info("Loaded regime features: %d rows, %d cols", len(regime_df), len(regime_df.columns))

    # --- 1. Alpha target ---
    logger.info("=== Building alpha target ===")
    alpha_targets = build_alpha_target(backbone_df, horizon=config.barrier.horizon)

    # --- 2. Barrier labels ---
    logger.info("=== Building barrier labels ===")
    barrier_labels = build_barrier_labels(backbone_df, config=config.barrier)

    # --- 3. Regime labels ---
    logger.info("=== Building regime labels ===")
    regime_labels = build_regime_labels(regime_df)

    # --- 4. Dashboard targets (1d / 3d / 5d price ratios) ---
    logger.info("=== Building dashboard targets ===")
    dashboard_horizons = config.dashboard_target.horizons
    dashboard_targets = build_dashboard_targets(backbone_df, horizons=dashboard_horizons)

    # --- Save ---
    alpha_path = processed_dir / "alpha_targets.parquet"
    barrier_path = processed_dir / "barrier_labels.parquet"
    regime_label_path = processed_dir / "regime_labels.parquet"
    dashboard_path = processed_dir / "dashboard_targets.parquet"

    alpha_targets.to_parquet(alpha_path, index=False)
    barrier_labels.to_parquet(barrier_path, index=False)
    regime_labels.to_parquet(regime_label_path, index=False)
    dashboard_targets.to_parquet(dashboard_path, index=False)

    # CSV copies for inspection
    alpha_targets.to_csv(processed_dir / "alpha_targets.csv", index=False)
    barrier_labels.to_csv(processed_dir / "barrier_labels.csv", index=False)
    regime_labels.to_csv(processed_dir / "regime_labels.csv", index=False)
    dashboard_targets.to_csv(processed_dir / "dashboard_targets.csv", index=False)

    logger.info("Saved alpha targets:     %s (%d rows)", alpha_path, len(alpha_targets))
    logger.info("Saved barrier labels:    %s (%d rows)", barrier_path, len(barrier_labels))
    logger.info("Saved regime labels:     %s (%d rows)", regime_label_path, len(regime_labels))
    logger.info("Saved dashboard targets: %s (%d rows)", dashboard_path, len(dashboard_targets))

    # --- Metadata ---
    barrier_dist_meta = _class_distribution(barrier_labels["barrier_label"])
    regime_dist_meta = _class_distribution(regime_labels["regime_label"])

    metadata = {
        "alpha_target": {
            "horizon": config.barrier.horizon,
            "rows": len(alpha_targets),
            "y_alpha_mean": float(alpha_targets["y_alpha"].mean()),
            "y_alpha_std": float(alpha_targets["y_alpha"].std()),
            "symbols": sorted(alpha_targets["symbol"].unique().tolist()),
        },
        "barrier_labels": {
            "horizon": config.barrier.horizon,
            "upper_multiplier": config.barrier.upper_multiplier,
            "lower_multiplier": config.barrier.lower_multiplier,
            "vol_lookback": config.barrier.vol_lookback,
            "target_scaling": config.barrier.target_scaling,
            "rows": len(barrier_labels),
            **barrier_dist_meta,
        },
        "regime_labels": {
            "rows": len(regime_labels),
            **regime_dist_meta,
            # Keep labels as string keys in JSON-safe map.
            "states": {"0": "defensive", "1": "balanced", "2": "aggressive"},
        },
    }

    # Dashboard target stats (ratio targets)
    dash_ratio_cols = [c for c in dashboard_targets.columns if c.startswith("y_ratio_")]
    dash_stats = {}
    for col in dash_ratio_cols:
        dash_stats[col] = {
            "mean": float(dashboard_targets[col].mean()),
            "std": float(dashboard_targets[col].std()),
        }
    metadata["dashboard_targets"] = {
        "horizons": dashboard_horizons,
        "rows": len(dashboard_targets),
        "target_columns": dash_ratio_cols,
        "target_type": "price_ratio",
        "stats": dash_stats,
        "symbols": sorted(dashboard_targets["symbol"].unique().tolist()),
    }

    meta_path = processed_dir / "target_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved target metadata: %s", meta_path)

    print("\nTargets built successfully:")
    print(f"  Alpha:     {len(alpha_targets):,} rows → {alpha_path}")
    print(f"  Barrier:   {len(barrier_labels):,} rows → {barrier_path}")
    print(f"  Regime:    {len(regime_labels):,} rows → {regime_label_path}")
    print(f"  Dashboard: {len(dashboard_targets):,} rows → {dashboard_path}")


if __name__ == "__main__":
    main()
