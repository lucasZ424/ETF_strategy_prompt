"""CLI entry point: python scripts/train_xgboost.py [--config path/to/config.toml]

Trains all three XGBoost models from pre-built feature and target parquets:
  1. Alpha regressor   — predict 10d forward log return
  2. Gate classifier   — predict triple-barrier ternary label {-1, 0, +1}
  3. Regime classifier — predict regime state {0, 1, 2}

Requires: run scripts/run_pipeline.py and scripts/build_targets.py first.
"""

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
from src.evaluation.metrics import (  # noqa: E402
    compute_classification_metrics,
    compute_ranker_metrics,
    compute_regression_metrics,
)
from src.models.splitter import chronological_split  # noqa: E402
from src.models.trainer import (  # noqa: E402
    get_feature_cols,
    safe_X,
    save_model_bundle,
    train_alpha_regressor,
    train_gate_classifier,
    train_regime_classifier,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_version_tag(config_path: Path) -> str:
    """Build a short version tag from model config for output directory naming."""
    model_cfg = load_model_config(config_path)
    t = int(model_cfg.train_ratio * 100)
    v = int(model_cfg.val_ratio * 100)
    esp = model_cfg.early_stopping_patience
    return f"t{t}_v{v}_esp{esp}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost models for ETF strategy.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "china_open_universe_minimal.template.toml",
    )
    args = parser.parse_args()

    pipe_cfg = load_config(args.config)
    model_cfg = load_model_config(args.config)
    processed_dir = PROJECT_ROOT / Path(pipe_cfg.processed_dir)
    model_dir = PROJECT_ROOT / Path(model_cfg.model_dir)

    version_tag = _build_version_tag(args.config)
    output_dir = PROJECT_ROOT / Path(model_cfg.output_dir) / f"xgboost_{version_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Load feature datasets and targets
    # ---------------------------------------------------------------
    alpha_features = pd.read_parquet(processed_dir / "alpha_features.parquet")
    gate_features = pd.read_parquet(processed_dir / "gate_features.parquet")
    regime_features = pd.read_parquet(processed_dir / "regime_features.parquet")

    alpha_targets = pd.read_parquet(processed_dir / "alpha_targets.parquet")
    barrier_labels = pd.read_parquet(processed_dir / "barrier_labels.parquet")
    regime_labels = pd.read_parquet(processed_dir / "regime_labels.parquet")

    logger.info(
        "Loaded features: alpha=%d, gate=%d, regime=%d",
        len(alpha_features), len(gate_features), len(regime_features),
    )
    logger.info(
        "Loaded targets: alpha=%d, barrier=%d, regime=%d",
        len(alpha_targets), len(barrier_labels), len(regime_labels),
    )

    # ---------------------------------------------------------------
    # 1. Alpha regressor
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("=== Training Alpha Regressor ===")
    logger.info("=" * 60)

    # Merge features with target on [date, symbol]
    alpha_df = alpha_features.merge(
        alpha_targets[["date", "symbol", "y_alpha"]],
        on=["date", "symbol"],
        how="inner",
    )
    logger.info("Alpha merged dataset: %d rows", len(alpha_df))

    alpha_feature_cols = get_feature_cols(alpha_features)
    alpha_split = chronological_split(
        alpha_df,
        train_ratio=model_cfg.train_ratio,
        val_ratio=model_cfg.val_ratio,
        test_ratio=model_cfg.test_ratio,
    )

    X_alpha_train = safe_X(alpha_split.train, alpha_feature_cols)
    y_alpha_train = alpha_split.train["y_alpha"].values
    X_alpha_val = safe_X(alpha_split.val, alpha_feature_cols)
    y_alpha_val = alpha_split.val["y_alpha"].values
    X_alpha_test = safe_X(alpha_split.test, alpha_feature_cols)
    y_alpha_test = alpha_split.test["y_alpha"].values

    alpha_result = train_alpha_regressor(
        X_alpha_train, y_alpha_train,
        X_alpha_val, y_alpha_val,
        alpha_feature_cols, model_cfg,
    )
    save_model_bundle(alpha_result, "alpha_xgboost", model_dir)

    # Evaluate alpha on test set
    alpha_test_preds = alpha_result.model.predict(X_alpha_test)
    alpha_reg_metrics = compute_regression_metrics(y_alpha_test, alpha_test_preds)
    alpha_rank_metrics = compute_ranker_metrics(
        alpha_split.test, alpha_test_preds, y_col="y_alpha",
    )
    logger.info("Alpha TEST regression: %s", alpha_reg_metrics)
    logger.info("Alpha TEST ranking:    %s", alpha_rank_metrics)

    # ---------------------------------------------------------------
    # 2. Gate classifier
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("=== Training Gate Classifier ===")
    logger.info("=" * 60)

    gate_df = gate_features.merge(
        barrier_labels[["date", "symbol", "barrier_label"]],
        on=["date", "symbol"],
        how="inner",
    )
    logger.info("Gate merged dataset: %d rows", len(gate_df))

    gate_feature_cols = get_feature_cols(gate_features)
    gate_split = chronological_split(
        gate_df,
        train_ratio=model_cfg.train_ratio,
        val_ratio=model_cfg.val_ratio,
        test_ratio=model_cfg.test_ratio,
    )

    X_gate_train = safe_X(gate_split.train, gate_feature_cols)
    y_gate_train = gate_split.train["barrier_label"].values
    X_gate_val = safe_X(gate_split.val, gate_feature_cols)
    y_gate_val = gate_split.val["barrier_label"].values
    X_gate_test = safe_X(gate_split.test, gate_feature_cols)
    y_gate_test = gate_split.test["barrier_label"].values

    gate_result = train_gate_classifier(
        X_gate_train, y_gate_train,
        X_gate_val, y_gate_val,
        gate_feature_cols, model_cfg,
    )
    save_model_bundle(gate_result, "gate_xgboost", model_dir)

    # Evaluate gate on test set — remap predictions back to {-1, 0, +1}
    gate_test_preds_mapped = gate_result.model.predict(X_gate_test)
    gate_test_preds = gate_test_preds_mapped.astype(int) - 1  # {0,1,2} → {-1,0,+1}
    gate_cls_metrics = compute_classification_metrics(
        y_gate_test, gate_test_preds,
        class_labels=["-1 (avoid)", "0 (flat)", "+1 (long)"],
    )
    logger.info("Gate TEST classification: %s", gate_cls_metrics)
    logger.info("Gate confusion matrix:\n%s", np.array(gate_cls_metrics.confusion))

    # ---------------------------------------------------------------
    # 3. Regime classifier
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("=== Training Regime Classifier ===")
    logger.info("=" * 60)

    # Regime features are date-level, merge with regime labels on date
    regime_df = regime_features.merge(
        regime_labels[["date", "regime_label"]],
        on="date",
        how="inner",
    )
    logger.info("Regime merged dataset: %d rows", len(regime_df))

    regime_feature_cols = [c for c in regime_features.columns if c != "date"]
    regime_split = chronological_split(
        regime_df,
        train_ratio=model_cfg.train_ratio,
        val_ratio=model_cfg.val_ratio,
        test_ratio=model_cfg.test_ratio,
    )

    X_regime_train = safe_X(regime_split.train, regime_feature_cols)
    y_regime_train = regime_split.train["regime_label"].values
    X_regime_val = safe_X(regime_split.val, regime_feature_cols)
    y_regime_val = regime_split.val["regime_label"].values
    X_regime_test = safe_X(regime_split.test, regime_feature_cols)
    y_regime_test = regime_split.test["regime_label"].values

    regime_result = train_regime_classifier(
        X_regime_train, y_regime_train,
        X_regime_val, y_regime_val,
        regime_feature_cols, model_cfg,
    )
    save_model_bundle(regime_result, "regime_xgboost", model_dir)

    # Evaluate regime on test set
    regime_test_preds = regime_result.model.predict(X_regime_test)
    regime_cls_metrics = compute_classification_metrics(
        y_regime_test, regime_test_preds,
        class_labels=["0 (defensive)", "1 (balanced)", "2 (aggressive)"],
    )
    logger.info("Regime TEST classification: %s", regime_cls_metrics)
    logger.info("Regime confusion matrix:\n%s", np.array(regime_cls_metrics.confusion))

    # ---------------------------------------------------------------
    # Save evaluation report
    # ---------------------------------------------------------------
    eval_report = {
        "alpha": {
            "regression": alpha_reg_metrics.to_dict(),
            "ranking": alpha_rank_metrics.to_dict(),
            "split": {
                "train_rows": len(alpha_split.train),
                "val_rows": len(alpha_split.val),
                "test_rows": len(alpha_split.test),
                "train_dates": [str(d) for d in [alpha_split.train_dates[0], alpha_split.train_dates[-1]]],
                "val_dates": [str(d) for d in [alpha_split.val_dates[0], alpha_split.val_dates[-1]]],
                "test_dates": [str(d) for d in [alpha_split.test_dates[0], alpha_split.test_dates[-1]]],
            },
        },
        "gate": {
            "classification": gate_cls_metrics.to_dict(),
            "split": {
                "train_rows": len(gate_split.train),
                "val_rows": len(gate_split.val),
                "test_rows": len(gate_split.test),
            },
        },
        "regime": {
            "classification": regime_cls_metrics.to_dict(),
            "split": {
                "train_rows": len(regime_split.train),
                "val_rows": len(regime_split.val),
                "test_rows": len(regime_split.test),
            },
        },
    }

    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2, default=str)
    logger.info("Saved evaluation report: %s", report_path)

    # Save eval histories
    for name, result in [("alpha", alpha_result), ("gate", gate_result), ("regime", regime_result)]:
        hist_path = output_dir / f"{name}_eval_history.json"
        with open(hist_path, "w") as f:
            json.dump(result.eval_history, f, indent=2)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nAlpha regressor (test):  {alpha_reg_metrics}")
    print(f"Alpha ranking (test):   {alpha_rank_metrics}")
    print(f"\nGate classifier (test): {gate_cls_metrics}")
    print(f"\nRegime classifier (test): {regime_cls_metrics}")
    print(f"\nModel artifacts: {model_dir}")
    print(f"Evaluation output: {output_dir}")


if __name__ == "__main__":
    main()
