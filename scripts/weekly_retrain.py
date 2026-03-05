"""Weekly automated retrain: fetch latest data, rebuild features, retrain model.

Usage:
    python scripts/weekly_retrain.py [--config path/to/config.toml]

Workflow:
    1. Fetch latest ETF data from Yahoo Finance (extends existing CSVs)
    2. Run data pipeline (clean, align, feature engineering)
    3. Train XGBoost with Optuna
    4. Generate plots

Schedule this via cron (Linux/Mac) or Task Scheduler (Windows):
    # Every Sunday at 18:00 CST (before Monday market open)
    0 18 * * 0 cd /path/to/ETF_strategy_prompt && python scripts/weekly_retrain.py
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)


def _fetch_and_append(ticker: str, csv_path: Path, end_date: str) -> int:
    """Fetch new data for ticker and append to existing CSV. Returns rows added."""
    if csv_path.exists():
        existing = pd.read_csv(csv_path, parse_dates=["Date"])
        last_date = existing["Date"].max()
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = "2015-01-01"
        existing = pd.DataFrame()

    if start_date >= end_date:
        logger.info("  %s: already up to date (last=%s)", ticker, start_date)
        return 0

    logger.info("  %s: fetching %s to %s", ticker, start_date, end_date)
    new_data = yf.download(ticker, start=start_date, end=end_date,
                           auto_adjust=False, progress=False)
    if new_data.empty:
        logger.info("  %s: no new data returned", ticker)
        return 0

    if isinstance(new_data.columns, pd.MultiIndex):
        new_data.columns = new_data.columns.get_level_values(0)
    new_data = new_data.reset_index()
    new_data.columns = [str(c).strip() for c in new_data.columns]

    if not existing.empty:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date"], keep="last")
        combined = combined.sort_values("Date").reset_index(drop=True)
    else:
        combined = new_data

    combined.to_csv(csv_path, index=False)
    n_new = len(combined) - len(existing)
    logger.info("  %s: added %d new rows (total=%d)", ticker, n_new, len(combined))
    return n_new


def fetch_latest_data(config, end_date: str) -> int:
    """Fetch latest data for all tickers. Returns total new rows."""
    raw_dir = PROJECT_ROOT / config.raw_dir
    total_new = 0

    # China ETFs
    china_dir = raw_dir / "china_etfs"
    china_dir.mkdir(parents=True, exist_ok=True)
    all_china = list(config.universe_core) + list(config.universe_optional)
    for ticker in all_china:
        total_new += _fetch_and_append(ticker, china_dir / f"{ticker}.csv", end_date)

    # Cross-market ETFs
    cross_dir = raw_dir / "cross_market"
    cross_dir.mkdir(parents=True, exist_ok=True)
    for ticker in config.cross_market:
        total_new += _fetch_and_append(ticker, cross_dir / f"{ticker}.csv", end_date)

    # Global risk proxies (VIX, TNX, DXY)
    risk_tickers = {"^VIX": "VIX.csv", "^TNX": "TNX.csv", "DX-Y.NYB": "DXY.csv"}
    for ticker, fname in risk_tickers.items():
        total_new += _fetch_and_append(ticker, cross_dir / fname, end_date)

    return total_new


def run_subprocess(script: str, config_path: Path) -> None:
    """Run a Python script as subprocess, streaming output."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script),
           "--config", str(config_path)]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed with return code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly automated retrain pipeline")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "china_open_universe_minimal.template.toml",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date for data fetch (default: today)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")

    # Step 1: Fetch latest data
    logger.info("=== Step 1: Fetching latest data (up to %s) ===", end_date)
    n_new = fetch_latest_data(config, end_date)
    logger.info("Total new rows fetched: %d", n_new)

    if n_new == 0:
        logger.info("No new data. Skipping pipeline and training.")
        return

    # Step 2: Run data pipeline
    logger.info("=== Step 2: Running data pipeline ===")
    run_subprocess("run_pipeline.py", args.config)

    # Step 3: Train model
    logger.info("=== Step 3: Training XGBoost ===")
    run_subprocess("train_xgboost.py", args.config)

    # Step 4: Generate plots
    logger.info("=== Step 4: Generating plots ===")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "plot_results.py")],
        cwd=str(PROJECT_ROOT),
    )

    logger.info("=== Weekly retrain complete ===")


if __name__ == "__main__":
    main()
