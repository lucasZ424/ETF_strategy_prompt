"""End-to-end data processing pipeline: raw CSV → 4 processed datasets."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import PipelineConfig
from src.data.cleaner import clean_china_etfs, clean_cross_market
from src.data.cross_market import align_cross_market_to_china
from src.data.loader import load_china_etfs, load_cross_market_etfs
from src.features.builder import TradeDatasets, build_trade_features
from src.features.dashboard_features import build_dashboard_features

logger = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig, project_root: Path) -> dict[str, Path]:
    """Execute the full data processing pipeline.

    1. Load raw CSVs  2. Clean  3. Align cross-market
    4. Build shared backbone  5. Emit 3 dataset views  6. Save

    Returns
    -------
    dict mapping dataset name → parquet path
    """

    raw_dir = project_root / config.raw_dir
    processed_dir = project_root / config.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    logger.info("=== Loading China ETFs ===")
    china_raw = load_china_etfs(raw_dir, config.universe_core, config.universe_optional)

    logger.info("=== Loading cross-market ETFs ===")
    cross_raw = load_cross_market_etfs(raw_dir, config.cross_market)

    # 2. Clean
    logger.info("=== Cleaning China ETFs ===")
    china_clean = clean_china_etfs(china_raw)

    logger.info("=== Cleaning cross-market ETFs ===")
    cross_clean = clean_cross_market(cross_raw)

    # 3. Align cross-market to China calendar
    logger.info("=== Aligning cross-market to China calendar ===")
    china_dates = china_clean["date"].drop_duplicates().sort_values()
    cross_aligned = align_cross_market_to_china(
        china_dates, cross_clean, config.cross_market, raw_dir=raw_dir
    )

    # 4-5. Build shared backbone → 3 dataset views
    logger.info("=== Building trade feature backbone ===")
    datasets: TradeDatasets = build_trade_features(
        china_clean,
        cross_aligned,
        feature_selection=config.feature_selection,
        seed=config.seed,
        cost_threshold=config.gate.cost_threshold,
        gap_threshold=config.gate.gap_threshold,
    )

    # 6. Build dashboard features (raw close-based)
    logger.info("=== Building dashboard features ===")
    dashboard_df = build_dashboard_features(china_clean, cross_aligned)

    # 7. Save all datasets
    output_paths: dict[str, Path] = {}

    for name, df in [
        ("alpha_features", datasets.alpha),
        ("gate_features", datasets.gate),
        ("regime_features", datasets.regime),
        ("backbone", datasets.backbone),
        ("dashboard_features", dashboard_df),
    ]:
        parquet_path = processed_dir / f"{name}.parquet"
        csv_path = processed_dir / f"{name}.csv"
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        logger.info(
            "Saved %s: %s (%d rows, %d cols)",
            name, parquet_path, len(df), len(df.columns),
        )
        output_paths[name] = parquet_path

    return output_paths
