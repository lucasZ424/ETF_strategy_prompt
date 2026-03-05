"""CLI entry point: python scripts/train_xgboost.py [--config path/to/config.toml]"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, load_model_config  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.evaluation.signal import generate_forecasts, optimize_threshold  # noqa: E402
from src.models.splitter import chronological_split  # noqa: E402
from src.models.trainer import safe_feature_array, save_model, train_xgboost  # noqa: E402


def _build_version_tag(model_cfg, n_features: int) -> str:
    """Build version tag from config and feature count.

    Example: xgboost_7ft_t70_v15_esp10
    """
    t = int(model_cfg.train_ratio * 100)
    v = int(model_cfg.val_ratio * 100)
    esp = model_cfg.early_stopping_patience
    return f"xgboost_{n_features}ft_t{t}_v{v}_esp{esp}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost for ETF return prediction"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "china_open_universe_minimal.template.toml",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Override path to features CSV (default: data/processed/features_v1.csv)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # 1. Load configs
    pipeline_cfg = load_config(args.config)
    model_cfg = load_model_config(args.config)

    # 2. Load feature data
    csv_path = args.input_csv or (
        PROJECT_ROOT / pipeline_cfg.processed_dir / "features_v1.csv"
    )
    logger.info("Loading features from %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["date"])
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # 3. Chronological split
    split = chronological_split(
        df,
        train_ratio=model_cfg.train_ratio,
        val_ratio=model_cfg.val_ratio,
        test_ratio=model_cfg.test_ratio,
    )

    # 4. Save target statistics from training set (for inverse-scale visualizations)
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_target = split.train["target"].values
    target_stats = {
        "mean": float(np.mean(train_target)),
        "std": float(np.std(train_target)),
        "min": float(np.min(train_target)),
        "max": float(np.max(train_target)),
    }
    stats_path = data_dir / "target_stats.json"
    with open(stats_path, "w") as f:
        json.dump(target_stats, f, indent=2)
    logger.info("Saved target stats: %s", stats_path)

    # 5. Train with Optuna
    logger.info(
        "=== Training XGBoost with Optuna (%d trials, patience=%d) ===",
        model_cfg.optuna_trials, model_cfg.early_stopping_patience,
    )
    result = train_xgboost(split.train, split.val, model_cfg)

    # 6. Build version tag and save model
    n_features = len(result.feature_names)
    version_tag = _build_version_tag(model_cfg, n_features)
    logger.info("Version tag: %s", version_tag)

    model_dir = PROJECT_ROOT / model_cfg.model_dir
    bundle_path = save_model(result, model_dir, name=version_tag)

    # 7. Evaluate on all splits
    logger.info("=== Evaluation ===")
    feature_cols = result.feature_names
    report = {}
    for name, subset in [("train", split.train), ("val", split.val), ("test", split.test)]:
        preds = result.model.predict(safe_feature_array(subset, feature_cols))
        metrics = compute_metrics(subset["target"].values, preds)
        logger.info("[%5s] %s", name, metrics)
        report[name] = metrics.to_dict()

    # 8. Threshold optimization on validation set
    logger.info("=== Signal Threshold Optimization ===")
    val_preds = result.model.predict(safe_feature_array(split.val, feature_cols))
    signal_cfg = optimize_threshold(
        split.val["target"].values,
        val_preds,
        thresholds=model_cfg.signal_threshold_search,
        criterion="directional_accuracy",
    )
    logger.info(
        "Signal config: long > %.5f, short < %.5f",
        signal_cfg.long_threshold,
        signal_cfg.short_threshold,
    )

    # 9. Generate signals on test set
    logger.info("=== Generating Test Set Signals ===")
    test_preds = result.model.predict(safe_feature_array(split.test, feature_cols))
    forecasts, summary_df = generate_forecasts(split.test, test_preds, signal_cfg)

    # 10. Save outputs under versioned directory
    output_dir = PROJECT_ROOT / model_cfg.output_dir / version_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "test_predictions.csv"
    summary_df.to_csv(predictions_path, index=False)
    logger.info("Saved test predictions: %s", predictions_path)

    signal_path = output_dir / "signal_config.json"
    with open(signal_path, "w") as f:
        json.dump(
            {
                "long_threshold": signal_cfg.long_threshold,
                "short_threshold": signal_cfg.short_threshold,
                "criterion": signal_cfg.criterion_name,
                "criterion_value": signal_cfg.criterion_value,
            },
            f,
            indent=2,
        )
    logger.info("Saved signal config: %s", signal_path)

    # Save eval history for loss curve plotting
    history_path = output_dir / "eval_history.json"
    with open(history_path, "w") as f:
        json.dump(result.eval_history, f)
    logger.info("Saved eval history: %s", history_path)

    report["best_params"] = result.best_params
    report["signal_config"] = {
        "long_threshold": signal_cfg.long_threshold,
        "short_threshold": signal_cfg.short_threshold,
    }
    report["version_tag"] = version_tag
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved evaluation report: %s", report_path)

    print("\nTraining complete.")
    print(f"  Version:     {version_tag}")
    print(f"  Model:       {bundle_path}")
    print(f"  Outputs:     {output_dir}")
    print(f"  Report:      {report_path}")
    print(f"\nRun: python scripts/plot_results.py --outputs-dir {output_dir} "
          f"--model-path {bundle_path}")


if __name__ == "__main__":
    main()
