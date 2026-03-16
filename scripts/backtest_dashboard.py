"""Backtest dashboard raw close-price models on unseen (or user-specified) ETFs.

Usage:
    python scripts/backtest_dashboard.py [--config path/to/config.toml]
                                         [--symbols 512880.SS 159919.SZ]

Loads trained dashboard models (1d/3d/5d), builds features & targets for
the specified ETFs from data/unseen_etfs/, and evaluates prediction quality.

Outputs per-ETF and aggregate metrics to outputs/dashboard_backtest/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, load_model_config  # noqa: E402
from src.data.cleaner import clean_china_etfs, clean_cross_market  # noqa: E402
from src.data.cross_market import align_cross_market_to_china  # noqa: E402
from src.data.loader import (  # noqa: E402
    load_cross_market_etfs,
    load_unseen_etfs,
)
from src.evaluation.metrics import compute_dashboard_metrics  # noqa: E402
from src.features.dashboard_features import build_dashboard_features  # noqa: E402
from src.models.trainer import load_model, load_feature_manifest, safe_X  # noqa: E402
from src.targets.dashboard_target import build_dashboard_targets  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest dashboard models on unseen ETFs.",
    )
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "china_open_universe_minimal.template.toml",
    )
    parser.add_argument(
        "--symbols", nargs="*", default=None,
        help="ETF symbols to test (default: all in data/unseen_etfs/)",
    )
    args = parser.parse_args()

    pipe_cfg = load_config(args.config)
    model_cfg = load_model_config(args.config)
    raw_dir = PROJECT_ROOT / pipe_cfg.raw_dir
    model_dir = PROJECT_ROOT / Path(model_cfg.model_dir)
    output_dir = PROJECT_ROOT / Path(model_cfg.output_dir) / "dashboard_backtest"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Load unseen ETF data
    # ---------------------------------------------------------------
    symbols = args.symbols or pipe_cfg.unseen_etfs or None
    logger.info("=== Loading unseen ETFs: %s ===", symbols)
    unseen_raw = load_unseen_etfs(raw_dir, symbols)
    unseen_clean = clean_china_etfs(unseen_raw)

    # ---------------------------------------------------------------
    # 2. Load cross-market data (shared with training)
    # ---------------------------------------------------------------
    logger.info("=== Loading cross-market data ===")
    cross_raw = load_cross_market_etfs(raw_dir, pipe_cfg.cross_market)
    cross_clean = clean_cross_market(cross_raw)

    unseen_dates = unseen_clean["date"].drop_duplicates().sort_values()
    cross_aligned = align_cross_market_to_china(
        unseen_dates, cross_clean, pipe_cfg.cross_market, raw_dir=raw_dir,
    )

    # ---------------------------------------------------------------
    # 3. Build dashboard features for unseen ETFs
    # ---------------------------------------------------------------
    logger.info("=== Building dashboard features for unseen ETFs ===")
    unseen_features = build_dashboard_features(unseen_clean, cross_aligned)
    logger.info("Unseen features: %d rows, %d cols", len(unseen_features), len(unseen_features.columns))

    # ---------------------------------------------------------------
    # 4. Build dashboard targets for unseen ETFs
    # ---------------------------------------------------------------
    logger.info("=== Building dashboard targets for unseen ETFs ===")
    # build_dashboard_targets expects [date, symbol, close]
    unseen_backbone = unseen_clean[["date", "symbol", "close"]].copy()
    unseen_targets = build_dashboard_targets(
        unseen_backbone, horizons=pipe_cfg.dashboard_target.horizons,
    )
    target_cols = [c for c in unseen_targets.columns if c.startswith("y_close_")]
    logger.info("Target columns: %s", target_cols)

    # ---------------------------------------------------------------
    # 5. Merge features with targets
    # ---------------------------------------------------------------
    merged = unseen_features.merge(
        unseen_targets[["date", "symbol"] + target_cols],
        on=["date", "symbol"], how="inner",
    )
    logger.info("Merged unseen dataset: %d rows", len(merged))

    if len(merged) == 0:
        logger.error("No rows after merge. Check date alignment between features and targets.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # 6. Load trained models and feature manifest
    # ---------------------------------------------------------------
    logger.info("=== Loading trained dashboard models ===")
    feature_manifest = load_feature_manifest(model_dir, "dashboard")

    models = {}
    for tcol in target_cols:
        model_name = f"dashboard_{tcol}"
        models[tcol] = load_model(model_dir, model_name)
        logger.info("Loaded model: %s", model_name)

    # Align features to manifest (handle missing features gracefully)
    available_features = [c for c in merged.columns if c not in ("date", "symbol", "close") and c not in target_cols]
    missing_features = [f for f in feature_manifest if f not in available_features]
    extra_features = [f for f in available_features if f not in feature_manifest]

    if missing_features:
        logger.warning("Missing features (will be zero-filled): %s", missing_features)
        for mf in missing_features:
            merged[mf] = 0.0
    if extra_features:
        logger.info("Extra features (not in manifest, ignored): %s", extra_features)

    X_unseen = safe_X(merged, feature_manifest)

    # ---------------------------------------------------------------
    # 7. Predict and evaluate
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("=== Evaluating Dashboard Models on Unseen ETFs ===")
    logger.info("=" * 60)

    eval_report = {
        "unseen_symbols": sorted(merged["symbol"].unique().tolist()),
        "n_rows": len(merged),
        "date_range": [str(merged["date"].min()), str(merged["date"].max())],
        "aggregate": {},
        "per_etf": {},
    }

    pred_df = merged[["date", "symbol", "close"]].copy()

    for tcol in target_cols:
        model = models[tcol]
        pred_close = model.predict(X_unseen)
        true_close = merged[tcol].values

        # Aggregate metrics
        agg_metrics = compute_dashboard_metrics(true_close, pred_close)
        eval_report["aggregate"][tcol] = agg_metrics.to_dict()
        logger.info("AGGREGATE %s: %s", tcol, agg_metrics)

        pred_df[f"{tcol}_true"] = true_close
        pred_df[f"{tcol}_pred"] = pred_close

        # Per-ETF metrics
        eval_report["per_etf"][tcol] = {}
        for sym in sorted(merged["symbol"].unique()):
            mask = merged["symbol"] == sym
            if mask.sum() == 0:
                continue
            sym_metrics = compute_dashboard_metrics(
                true_close[mask.values], pred_close[mask.values],
            )
            eval_report["per_etf"][tcol][sym] = sym_metrics.to_dict()
            logger.info("  %s %s: %s", tcol, sym, sym_metrics)

    # ---------------------------------------------------------------
    # 8. Save outputs
    # ---------------------------------------------------------------
    report_path = output_dir / "backtest_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2, default=str)
    logger.info("Saved evaluation report: %s", report_path)

    pred_path = output_dir / "backtest_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info("Saved predictions: %s", pred_path)

    # ---------------------------------------------------------------
    # 9. Prediction vs Actual plots (saved as PNG)
    # ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        unseen_symbols = sorted(merged["symbol"].unique())
        for tcol in target_cols:
            for sym in unseen_symbols:
                sym_df = pred_df[pred_df["symbol"] == sym].sort_values("date")
                if len(sym_df) == 0:
                    continue

                fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

                # Top: actual vs predicted
                ax1 = axes[0]
                ax1.plot(sym_df["date"], sym_df[f"{tcol}_true"], label="Actual", linewidth=1)
                ax1.plot(sym_df["date"], sym_df[f"{tcol}_pred"], label="Predicted", linewidth=1, alpha=0.8)
                ax1.set_ylabel("Close Price")
                ax1.set_title(f"{sym} — {tcol} Prediction vs Actual")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Bottom: prediction error
                ax2 = axes[1]
                error = sym_df[f"{tcol}_pred"].values - sym_df[f"{tcol}_true"].values
                ax2.bar(sym_df["date"], error, color="steelblue", alpha=0.6, width=2)
                ax2.axhline(y=0, color="black", linewidth=0.5)
                ax2.set_ylabel("Error (Pred - Actual)")
                ax2.set_xlabel("Date")
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plot_path = output_dir / f"{sym}_{tcol}_backtest.png"
                fig.savefig(plot_path, dpi=120)
                plt.close(fig)
                logger.info("Saved plot: %s", plot_path)

    except ImportError:
        logger.warning("matplotlib not available, skipping plots.")

    # ---------------------------------------------------------------
    # 10. Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Dashboard Backtest Complete (Unseen ETFs)")
    print("=" * 60)
    print(f"ETFs tested: {eval_report['unseen_symbols']}")
    print(f"Date range: {eval_report['date_range'][0]} to {eval_report['date_range'][1]}")
    print(f"Total rows: {eval_report['n_rows']}")

    for tcol in target_cols:
        m = eval_report["aggregate"][tcol]
        print(
            f"\n  {tcol} (aggregate):"
            f"\n    MSE={m['mse']:.4f}  MAE={m['mae']:.4f}  "
            f"MAPE={m['mape']:.2f}%  R2={m['r2']:.6f}"
        )
        for sym in eval_report["per_etf"][tcol]:
            sm = eval_report["per_etf"][tcol][sym]
            print(
                f"    {sym}: MSE={sm['mse']:.4f}  MAE={sm['mae']:.4f}  "
                f"MAPE={sm['mape']:.2f}%  R2={sm['r2']:.6f}"
            )

    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
