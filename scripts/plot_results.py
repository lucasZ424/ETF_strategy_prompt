"""Generate diagnostic plots for dashboard price-prediction models.

Usage: python scripts/plot_results.py [--outputs-dir outputs/dashboard_<tag>]

Plots saved to outputs/dashboard_<tag>/plots/:
  1. loss_curves.png          — Train vs Val RMSE per boosting round, one subplot per horizon
  2. true_vs_pred_close.png   — True vs predicted close price on test set, one subplot per horizon
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def plot_loss_curves(output_dir: Path, plots_dir: Path) -> None:
    """Train vs Val RMSE per boosting round — one subplot per horizon."""
    # Discover eval history files
    hist_files = sorted(output_dir.glob("y_close_*_eval_history.json"))
    if not hist_files:
        print("  No eval_history files found, skipping loss curves.")
        return

    n = len(hist_files)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, hf in zip(axes, hist_files):
        horizon_name = hf.stem.replace("_eval_history", "")
        with open(hf) as f:
            hist = json.load(f)

        train_rmse = hist["train_rmse"]
        val_rmse = hist["val_rmse"]
        epochs = range(1, len(train_rmse) + 1)

        # Convert RMSE to MSE for display (RMSE^2 = MSE)
        train_mse = [v ** 2 for v in train_rmse]
        val_mse = [v ** 2 for v in val_rmse]

        best_epoch = int(np.argmin(val_mse)) + 1
        best_val = val_mse[best_epoch - 1]

        ax.plot(epochs, train_mse, label="Train MSE", alpha=0.8, linewidth=1)
        ax.plot(epochs, val_mse, label="Val MSE", alpha=0.8, linewidth=1)
        ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.6,
                   label=f"Best epoch={best_epoch}")
        ax.scatter([best_epoch], [best_val], color="red", zorder=5, s=40)

        ax.set_xlabel("Boosting Round")
        ax.set_ylabel("MSE")
        ax.set_title(f"{horizon_name}\n(best MSE={best_val:.6f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Loss Curves (MSE)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved loss_curves.png ({n} horizons)")


def plot_true_vs_pred_close(
    output_dir: Path,
    plots_dir: Path,
) -> None:
    """True vs predicted close price scatter — one subplot per horizon.

    Reads test-set predictions saved during training. If no prediction CSV
    exists, we re-derive from evaluation_report + raw data.
    """
    # Load evaluation report for metrics annotation
    report_path = output_dir / "evaluation_report.json"
    with open(report_path) as f:
        report = json.load(f)

    horizons = sorted(report["horizons"].keys())
    n = len(horizons)

    # Try to load saved test predictions; if not available, load from processed data
    pred_path = output_dir / "test_predictions.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
    else:
        pred_df = _build_test_predictions(output_dir, report)

    if pred_df is None:
        print("  Cannot build true vs pred plot — no prediction data available.")
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    axes = axes[0]

    for ax, horizon in zip(axes, horizons):
        true_col = f"{horizon}_true"
        pred_col = f"{horizon}_pred"

        if true_col not in pred_df.columns or pred_col not in pred_df.columns:
            ax.set_title(f"{horizon}\n(no data)")
            continue

        y_true = pred_df[true_col].values
        y_pred = pred_df[pred_col].values

        # Scatter
        ax.scatter(y_true, y_pred, alpha=0.15, s=8, edgecolors="none", c="steelblue")

        # y=x line
        all_vals = np.concatenate([y_true, y_pred])
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        margin = (vmax - vmin) * 0.05
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                "r--", alpha=0.6, linewidth=1, label="y=x")

        # Metrics annotation — flat structure (no nested "close_price" key)
        m = report["horizons"][horizon]
        text = (
            f"R\u00b2 = {m['r2']:.6f}\n"
            f"RMSE = {m['rmse']:.4f}\n"
            f"MAE = {m['mae']:.4f}\n"
            f"MAPE = {m['mape']:.2f}%"
        )
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

        ax.set_xlabel("True Close")
        ax.set_ylabel("Predicted Close")
        ax.set_title(f"{horizon}")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Test Set: True vs Predicted Close Price", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "true_vs_pred_close.png", dpi=150)
    plt.close(fig)
    print(f"  Saved true_vs_pred_close.png ({n} horizons)")


def _build_test_predictions(output_dir: Path, report: dict) -> pd.DataFrame | None:
    """Re-generate test predictions by running models on test data.

    Falls back to loading processed data + model bundles.
    """
    import joblib

    from src.config import load_config, load_model_config
    from src.data.cleaner import clean_china_etfs, clean_cross_market
    from src.data.cross_market import align_cross_market_to_china
    from src.data.loader import load_china_etfs, load_cross_market_etfs
    from src.features.dashboard_features import build_dashboard_features
    from src.models.splitter import chronological_split
    from src.models.trainer import safe_X

    # Find config
    config_path = PROJECT_ROOT / "configs" / "china_open_universe_minimal.template.toml"
    if not config_path.exists():
        return None

    pipe_cfg = load_config(config_path)
    model_cfg = load_model_config(config_path)
    raw_dir = PROJECT_ROOT / pipe_cfg.raw_dir
    processed_dir = PROJECT_ROOT / Path(pipe_cfg.processed_dir)
    model_dir = PROJECT_ROOT / Path(model_cfg.model_dir)

    # Load model bundle
    bundle_path = model_dir / "dashboard_bundle.joblib"
    if not bundle_path.exists():
        return None

    bundle = joblib.load(bundle_path)
    feature_cols = bundle["feature_cols"]
    models = bundle["models"]

    # Build features
    china_raw = load_china_etfs(raw_dir, pipe_cfg.universe_core, pipe_cfg.universe_optional)
    cross_raw = load_cross_market_etfs(raw_dir, pipe_cfg.cross_market)
    china_clean = clean_china_etfs(china_raw)
    cross_clean = clean_cross_market(cross_raw)
    china_dates = china_clean["date"].drop_duplicates().sort_values()
    cross_aligned = align_cross_market_to_china(
        china_dates, cross_clean, pipe_cfg.cross_market, raw_dir=raw_dir,
    )
    dash_features = build_dashboard_features(china_clean, cross_aligned)

    # Load targets and merge
    dashboard_targets = pd.read_parquet(processed_dir / "dashboard_targets.parquet")
    target_cols = [c for c in dashboard_targets.columns if c.startswith("y_close_")]
    merge_cols = ["date", "symbol"] + target_cols
    merged = dash_features.merge(dashboard_targets[merge_cols], on=["date", "symbol"], how="inner")

    # Split to get test set
    split = chronological_split(
        merged,
        train_ratio=model_cfg.train_ratio,
        val_ratio=model_cfg.val_ratio,
        test_ratio=model_cfg.test_ratio,
    )

    X_test = safe_X(split.test, feature_cols)
    result = split.test[["date", "symbol", "close"]].copy()

    for col in target_cols:
        model = models[col]
        result[f"{col}_true"] = split.test[col].values
        result[f"{col}_pred"] = model.predict(X_test)

    # Save for future runs
    result.to_csv(output_dir / "test_predictions.csv", index=False)
    print("  Generated and saved test_predictions.csv")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dashboard diagnostic plots")
    parser.add_argument("--outputs-dir", type=Path, default=None,
                        help="Versioned output dir (e.g. outputs/dashboard_t70_v15_esp50)")
    args = parser.parse_args()

    # Auto-detect latest dashboard output
    if args.outputs_dir is None:
        outputs_root = PROJECT_ROOT / "outputs"
        version_dirs = sorted(
            [d for d in outputs_root.iterdir()
             if d.is_dir() and d.name.startswith("dashboard_")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not version_dirs:
            print("No dashboard output directories found under outputs/. "
                  "Run train_dashboard.py first.")
            sys.exit(1)
        args.outputs_dir = version_dirs[0]

    plots_dir = args.outputs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating dashboard plots from {args.outputs_dir.name}...")

    plot_loss_curves(args.outputs_dir, plots_dir)
    plot_true_vs_pred_close(args.outputs_dir, plots_dir)

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
