"""Daily dashboard inference: predict 1d/3d/5d raw close prices for all eligible ETFs.

Loads trained dashboard model bundles, builds features from the latest bars
in the cloud DB, predicts forward close prices, and writes results to the
``prediction_snapshots`` table for Grafana consumption.

Usage::

    # Predict for all eligible ETFs (default)
    python -m scripts.predict_dashboard

    # Predict for specific symbols only
    python -m scripts.predict_dashboard --symbols 510050.SS 159915.SZ

    # Dry run - show eligible ETFs and feature stats without writing
    python -m scripts.predict_dashboard --dry-run

    # Override model directory
    python -m scripts.predict_dashboard --model-dir models

Environment:
    ETF_DB_URL - cloud PostgreSQL connection string (required)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import text
from src.config import load_config
from src.data.cleaner import clean_china_etfs, clean_cross_market
from src.data.cross_market import align_cross_market_to_china
from src.data.db import get_engine, get_session_factory, resolve_db_url
from src.data.repository import BarRepository, PredictionRepository
from src.features.dashboard_features import build_dashboard_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))



warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Inference thresholds ---
MIN_RECENT_BARS = 30        # minimum bars needed for feature warmup
FRESHNESS_CALENDAR_DAYS = 45  # bars must span within this window

HORIZONS = ["y_close_1d", "y_close_3d", "y_close_5d"]


def _load_models(model_dir: Path) -> tuple[dict, list[str], str]:
    """Load dashboard model bundles.

    Returns (models_dict, feature_cols, model_version).
    """
    models = {}
    for horizon in HORIZONS:
        path = model_dir / f"dashboard_{horizon}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        models[horizon] = joblib.load(path)
        logger.info("Loaded model: %s", path.name)

    manifest_path = model_dir / "dashboard_feature_manifest.json"
    with open(manifest_path) as f:
        feature_cols = json.load(f)
    logger.info("Feature manifest: %d features", len(feature_cols))

    # Derive model version from metadata
    meta_path = model_dir / "dashboard_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        model_version = f"dashboard_v1_nf{meta.get('n_features', len(feature_cols))}"
    else:
        model_version = "dashboard_v1"

    return models, feature_cols, model_version


def _get_eligible_symbols(engine, symbols_override: list[str] | None) -> list[str]:
    """Return symbols eligible for prediction based on freshness criteria."""
    cutoff_date = date.today() - timedelta(days=FRESHNESS_CALENDAR_DAYS)

    if symbols_override:
        # Validate overrides exist and meet freshness
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT symbol, COUNT(*) AS n_bars, MAX(trade_date) AS last_date
                FROM daily_bars
                WHERE symbol = ANY(:symbols)
                  AND trade_date >= :cutoff
                GROUP BY symbol
                HAVING COUNT(*) >= :min_bars
            """), {
                "symbols": symbols_override,
                "cutoff": cutoff_date,
                "min_bars": MIN_RECENT_BARS,
            }).fetchall()

        eligible = [r[0] for r in rows]
        skipped = set(symbols_override) - set(eligible)
        if skipped:
            logger.warning("Skipped (insufficient recent data): %s", sorted(skipped))
        return eligible

    # All active china_etf instruments meeting freshness criteria
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT b.symbol, COUNT(*) AS n_bars, MAX(b.trade_date) AS last_date
            FROM daily_bars b
            JOIN instrument_master i ON b.symbol = i.symbol
            WHERE i.asset_type = 'china_etf'
              AND i.is_active = TRUE
              AND b.trade_date >= :cutoff
            GROUP BY b.symbol
            HAVING COUNT(*) >= :min_bars
        """), {
            "cutoff": cutoff_date,
            "min_bars": MIN_RECENT_BARS,
        }).fetchall()

    symbols = [r[0] for r in rows]
    logger.info(
        "Eligible ETFs: %d (of those in instrument_master with >= %d bars since %s)",
        len(symbols), MIN_RECENT_BARS, cutoff_date,
    )
    return symbols


def _load_bars_for_inference(
    repo: BarRepository,
    symbols: list[str],
    cross_market_symbols: list[str],
    macro_symbols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load recent bars for china ETFs, cross-market, and macro series."""
    # Load enough history for feature warmup (30 bars + buffer)
    lookback_start = date.today() - timedelta(days=90)

    china_df = repo.load_bars(symbols, start_date=lookback_start)
    logger.info("China ETF bars loaded: %d rows, %d symbols", len(china_df), china_df["symbol"].nunique())

    cross_df = repo.load_bars(cross_market_symbols, start_date=lookback_start)
    logger.info("Cross-market bars loaded: %d rows", len(cross_df))

    # Build macro loader that reads from DB
    def macro_loader(symbol: str) -> pd.DataFrame:
        return repo.load_macro_series(symbol)

    return china_df, cross_df, {"macro_loader": macro_loader}


def _build_features_and_predict(
    china_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    macro_kwargs: dict,
    cross_market_symbols: list[str],
    models: dict,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build features and run prediction for the latest date per symbol."""
    # Clean
    china_clean = clean_china_etfs(china_df)
    cross_clean = clean_cross_market(cross_df)

    # Align cross-market to china dates
    china_dates = china_clean["date"].drop_duplicates().sort_values()
    cross_aligned = align_cross_market_to_china(
        china_dates, cross_clean, cross_market_symbols,
        macro_loader=macro_kwargs.get("macro_loader"),
    )

    # Build dashboard features
    dash_features = build_dashboard_features(china_clean, cross_aligned)
    logger.info("Dashboard features built: %d rows, %d cols", len(dash_features), len(dash_features.columns))

    # Take only the latest date per symbol for prediction
    latest_date = dash_features["date"].max()
    latest = dash_features[dash_features["date"] == latest_date].copy()
    logger.info("Predicting for date: %s, %d symbols", latest_date, len(latest))

    if latest.empty:
        logger.warning("No data for latest date after feature build")
        return pd.DataFrame()

    # Check feature availability
    missing_features = [f for f in feature_cols if f not in latest.columns]
    if missing_features:
        logger.error("Missing features in data: %s", missing_features)
        raise ValueError(f"Missing features: {missing_features}")

    X = latest[feature_cols].values

    # Predict
    result = latest[["date", "symbol", "close"]].copy()
    result = result.rename(columns={"close": "current_close"})
    for horizon in HORIZONS:
        result[f"{horizon}_pred"] = models[horizon].predict(X)
        logger.info("  %s: predicted %d values", horizon, len(X))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily dashboard inference - predict 1d/3d/5d close prices."
    )
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "china_open_universe_minimal.template.toml",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=PROJECT_ROOT / "models",
        help="Directory containing trained dashboard model bundles.",
    )
    parser.add_argument(
        "--symbols", nargs="*", default=None,
        help="Predict only for these symbols (default: all eligible).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show eligible ETFs and feature stats without writing to DB.",
    )
    args = parser.parse_args()

    # --- Config & DB ---
    config = load_config(args.config)
    db_url = resolve_db_url(config.database.url, config.database.url_env)
    engine = get_engine(db_url)
    session_factory = get_session_factory(engine)
    repo = BarRepository(session_factory)
    pred_repo = PredictionRepository(session_factory)

    # --- Load models ---
    models, feature_cols, model_version = _load_models(args.model_dir)
    logger.info("Model version: %s", model_version)

    # --- Find eligible symbols ---
    eligible = _get_eligible_symbols(engine, args.symbols)
    if not eligible:
        logger.warning("No eligible ETFs found. Exiting.")
        return

    logger.info("Eligible ETFs: %d symbols", len(eligible))
    if args.dry_run:
        for s in sorted(eligible):
            print(f"  {s}")
        print(f"\nTotal: {len(eligible)} ETFs would receive predictions.")
        return

    # --- Load bars ---
    china_df, cross_df, macro_kwargs = _load_bars_for_inference(
        repo, eligible, config.cross_market, config.global_risk,
    )

    # --- Build features & predict ---
    predictions = _build_features_and_predict(
        china_df, cross_df, macro_kwargs,
        config.cross_market, models, feature_cols,
    )

    if predictions.empty:
        logger.warning("No predictions generated. Exiting.")
        return

    # --- Determine asof_date and data freshness ---
    asof_date = pd.Timestamp(predictions["date"].iloc[0]).date()

    # Reshape for PredictionRepository
    snapshot_df = pd.DataFrame({
        "symbol": predictions["symbol"],
        "current_close": predictions["current_close"],
        "y_close_1d_pred": predictions["y_close_1d_pred"],
        "y_close_3d_pred": predictions["y_close_3d_pred"],
        "y_close_5d_pred": predictions["y_close_5d_pred"],
    })

    # --- Write to DB ---
    n_written = pred_repo.write_snapshot(
        predictions=snapshot_df,
        model_version=model_version,
        asof_date=asof_date,
        data_freshness_date=asof_date,
    )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Dashboard predictions written to DB")
    print(f"{'=' * 60}")
    print(f"  Model version : {model_version}")
    print(f"  As-of date    : {asof_date}")
    print(f"  Symbols       : {n_written}")
    print(f"  Features used : {len(feature_cols)}")
    print("\nSample predictions:")
    sample = predictions.head(5)
    for _, row in sample.iterrows():
        print(
            f"  {row['symbol']:12s}  close={row['current_close']:.3f}"
            f"  1d={row['y_close_1d_pred']:.3f}"
            f"  3d={row['y_close_3d_pred']:.3f}"
            f"  5d={row['y_close_5d_pred']:.3f}"
        )


if __name__ == "__main__":
    main()
