"""End-to-end data processing pipeline: raw CSV → processed parquet."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import PipelineConfig
from src.data.cleaner import clean_china_etfs, clean_cross_market
from src.data.cross_market import align_cross_market_to_china
from src.data.loader import load_china_etfs, load_cross_market_etfs
from src.features.builder import build_all_features

logger = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig, project_root: Path) -> Path:
    """Execute the full data processing pipeline.

    1. Load raw CSVs  2. Clean  3. Align cross-market  4. Build features  5. Save
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

    # 4. Build features
    logger.info("=== Building features ===")
    processed = build_all_features(
        china_clean, cross_aligned, config.lookback_windows,
        top_k_features=config.top_k_features,
        cross_symbols=config.cross_market,
    )

    # 5. Save
    parquet_path = processed_dir / "features_v1.parquet"
    processed.to_parquet(parquet_path, index=False)
    logger.info("Saved parquet: %s (%d rows, %d cols)", parquet_path, len(processed), len(processed.columns))

    csv_path = processed_dir / "features_v1.csv"
    processed.to_csv(csv_path, index=False)
    logger.info("Saved CSV: %s", csv_path)

    return parquet_path
