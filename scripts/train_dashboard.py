"""CLI entry point: python scripts/train_dashboard.py [--config path/to/config.toml]

Trains the dashboard price-prediction model (1d / 3d / 5d raw close forecasts).
One XGBoost regressor per horizon, using dedicated dashboard features built from
raw Close (per ETF_feature_engineering_dash.txt).

Targets: y_close_Hd = close_{t+H}  (actual future close price)

Requires: run scripts/run_pipeline.py and scripts/build_targets.py first.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, load_model_config  # noqa: E402
from src.data.cleaner import clean_china_etfs, clean_cross_market  # noqa: E402
from src.data.cross_market import align_cross_market_to_china  # noqa: E402
from src.data.loader import load_china_etfs, load_cross_market_etfs  # noqa: E402
from src.evaluation.metrics import compute_dashboard_metrics  # noqa: E402
from src.features.dashboard_features import build_dashboard_features  # noqa: E402
from src.features.screening import run_feature_selection  # noqa: E402
from src.models.splitter import chronological_split  # noqa: E402
from src.models.trainer import (  # noqa: E402
    safe_X,
    save_dashboard_bundle,
    train_dashboard_regressor,
)
from src.data.backend import DataBackend, StorageBackend  # noqa: E402
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_version_tag(config_path: Path) -> str:
    model_cfg = load_model_config(config_path)
    t = int(model_cfg.train_ratio * 100)
    v = int(model_cfg.val_ratio * 100)
    esp = model_cfg.early_stopping_patience
    return f"t{t}_v{v}_esp{esp}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dashboard price-prediction models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "china_open_universe_minimal.template.toml",
    )
    args = parser.parse_args()

    pipe_cfg = load_config(args.config)
    model_cfg = load_model_config(args.config)
    raw_dir = PROJECT_ROOT / pipe_cfg.raw_dir
    processed_dir = PROJECT_ROOT / Path(pipe_cfg.processed_dir)
    model_dir = PROJECT_ROOT / Path(model_cfg.model_dir)

    version_tag = _build_version_tag(args.config)
    output_dir = PROJECT_ROOT / Path(model_cfg.output_dir) / f"dashboard_{version_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Build dashboard-specific features from raw data
    # ---------------------------------------------------------------
    logger.info("=== Building dashboard features from raw data ===")

    # Use DataBackend when configured for DB; fall back to file loaders
    backend = None
    if pipe_cfg.database.backend != "file":
        backend = DataBackend(pipe_cfg, PROJECT_ROOT)

    use_db = backend is not None and backend.backend != StorageBackend.FILE

    if use_db:
        logger.info("Loading raw data from DB backend")
        china_raw = backend.load_china_etfs()
        cross_raw = backend.load_cross_market_etfs()
    else:
        logger.info("Loading raw data from local files")
        china_raw = load_china_etfs(raw_dir, pipe_cfg.universe_core, pipe_cfg.universe_optional)
        cross_raw = load_cross_market_etfs(raw_dir, pipe_cfg.cross_market)

    china_clean = clean_china_etfs(china_raw)
    cross_clean = clean_cross_market(cross_raw)

    china_dates = china_clean["date"].drop_duplicates().sort_values()

    macro_loader = backend.load_macro_series if use_db else None
    cross_aligned = align_cross_market_to_china(
        china_dates, cross_clean, pipe_cfg.cross_market,
        raw_dir=raw_dir if not use_db else None,
        macro_loader=macro_loader,
    )

    dash_features = build_dashboard_features(china_clean, cross_aligned)

    # Save for inspection
    dash_features.to_parquet(processed_dir / "dashboard_features.parquet", index=False)
    dash_features.to_csv(processed_dir / "dashboard_features.csv", index=False)
    logger.info("Saved dashboard features: %d rows, %d cols", len(dash_features), len(dash_features.columns))

    # ---------------------------------------------------------------
    # Load dashboard targets
    # ---------------------------------------------------------------
    dashboard_targets = pd.read_parquet(processed_dir / "dashboard_targets.parquet")
    logger.info("Loaded dashboard targets: %d rows", len(dashboard_targets))

    target_cols = [c for c in dashboard_targets.columns if c.startswith("y_close_")]
    logger.info("Target columns: %s", target_cols)

    # ---------------------------------------------------------------
    # Merge features with targets
    # ---------------------------------------------------------------
    merged = dash_features.merge(
        dashboard_targets[["date", "symbol"] + target_cols],
        on=["date", "symbol"],
        how="inner",
    )
    logger.info("Merged dataset: %d rows", len(merged))

    feature_cols = [c for c in dash_features.columns if c not in ("date", "symbol", "close")]
    logger.info("Feature columns (%d): %s", len(feature_cols), sorted(feature_cols))

    # ---------------------------------------------------------------
    # Two-layer feature selection (Pearson + XGBoost importance)
    # ---------------------------------------------------------------
    fs_cfg = pipe_cfg.feature_selection
    # Use shortest-horizon target as proxy for feature selection
    proxy_target_col = target_cols[0]  # e.g. y_close_1d
    logger.info("=== Running feature selection (proxy target: %s) ===", proxy_target_col)

    sel_df = merged[feature_cols].copy()
    sel_df["target"] = merged[proxy_target_col].values

    selected_features, fs_metadata = run_feature_selection(
        sel_df, feature_cols, target_col="target",
        correlation_threshold=fs_cfg.correlation_threshold,
        importance_top_k=fs_cfg.importance_top_k,
        protected_prefixes=list(fs_cfg.protected_prefixes),
        seed=pipe_cfg.seed,
    )
    logger.info(
        "Feature selection: %d → %d features. Pearson dropped: %s",
        len(feature_cols), len(selected_features),
        fs_metadata.get("pearson_dropped", []),
    )

    # Save importance ranking for notebook visualization
    importance_path = output_dir / "dashboard_feature_importance.json"
    with open(importance_path, "w") as f:
        json.dump(fs_metadata.get("importance_ranking", {}), f, indent=2)
    logger.info("Saved feature importance ranking: %s", importance_path)

    # Keep all features for now (user decides top_k from notebook)
    # Store the selected set for reference
    all_feature_cols = feature_cols
    feature_cols = sorted(selected_features)
    logger.info("Post-selection features (%d): %s", len(feature_cols), feature_cols)

    # ---------------------------------------------------------------
    # Chronological split
    # ---------------------------------------------------------------
    split = chronological_split(
        merged,
        train_ratio=model_cfg.train_ratio,
        val_ratio=model_cfg.val_ratio,
        test_ratio=model_cfg.test_ratio,
    )

    X_train = safe_X(split.train, feature_cols)
    X_val = safe_X(split.val, feature_cols)
    X_test = safe_X(split.test, feature_cols)

    y_train = {col: split.train[col].values for col in target_cols}
    y_val = {col: split.val[col].values for col in target_cols}
    y_test = {col: split.test[col].values for col in target_cols}

    # ---------------------------------------------------------------
    # Train on raw close prices
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("=== Training Dashboard Price Models (raw close targets) ===")
    logger.info("=" * 60)

    result = train_dashboard_regressor(
        X_train, y_train,
        X_val, y_val,
        feature_cols, model_cfg,
    )
    save_dashboard_bundle(result, model_dir)

    # ---------------------------------------------------------------
    # Evaluate on test set
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("=== Evaluating on Test Set ===")
    logger.info("=" * 60)

    eval_report = {
        "split": {
            "train_rows": len(split.train),
            "val_rows": len(split.val),
            "test_rows": len(split.test),
            "train_dates": [str(split.train_dates[0]), str(split.train_dates[-1])],
            "val_dates": [str(split.val_dates[0]), str(split.val_dates[-1])],
            "test_dates": [str(split.test_dates[0]), str(split.test_dates[-1])],
        },
        "horizons": {},
    }

    for col in target_cols:
        model = result.models[col]
        pred_close = model.predict(X_test)
        true_close = y_test[col]

        close_metrics = compute_dashboard_metrics(true_close, pred_close)
        logger.info("Dashboard %s TEST: %s", col, close_metrics)

        eval_report["horizons"][col] = close_metrics.to_dict()

    eval_report["feature_selection"] = {
        "pre_selection_count": len(all_feature_cols),
        "post_selection_count": len(feature_cols),
        "pearson_dropped": fs_metadata.get("pearson_dropped", []),
        "correlation_threshold": fs_cfg.correlation_threshold,
        "importance_top_k": fs_cfg.importance_top_k,
    }

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2, default=str)
    logger.info("Saved evaluation report: %s", report_path)

    # Save test predictions for plotting
    test_pred_df = split.test[["date", "symbol", "close"]].copy()
    for col in target_cols:
        test_pred_df[f"{col}_true"] = y_test[col]
        test_pred_df[f"{col}_pred"] = result.models[col].predict(X_test)
    test_pred_path = output_dir / "test_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False)
    logger.info("Saved test predictions: %s", test_pred_path)

    # Save eval histories
    for col, hist in result.eval_histories.items():
        hist_path = output_dir / f"{col}_eval_history.json"
        with open(hist_path, "w") as f:
            json.dump(hist, f, indent=2)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Dashboard training complete! (raw close-price targets)")
    print("=" * 60)
    for col in target_cols:
        m = eval_report["horizons"][col]
        print(
            f"\n  {col}:"
            f"\n    MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  MAPE={m['mape']:.2f}%  R2={m['r2']:.6f}"
        )
    print(f"\nModel artifacts: {model_dir}")
    print(f"Evaluation output: {output_dir}")


if __name__ == "__main__":
    main()
