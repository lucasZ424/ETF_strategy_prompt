"""Generate diagnostic plots after training.

Usage: python scripts/plot_results.py --outputs-dir outputs/<version_tag> --model-path models/<version_tag>.joblib

Plots saved to outputs/<version_tag>/plots/:
  1. loss_curve.png          — Train vs Val RMSE per boosting round (early stop marked)
  2. true_vs_pred_test.png   — Scatter: true vs predicted log returns on test set
  3. residual_dist.png       — Residual histogram + Q-Q plot
  4. feature_importance.png  — All feature importances (gain)
  5. pred_timeseries.png     — True vs predicted over time (per-ETF facets)
  6. signal_distribution.png — Distribution of predictions colored by signal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def plot_loss_curve(eval_history: dict, plots_dir: Path) -> None:
    """Train vs Val RMSE per boosting round."""
    train_rmse = eval_history["train_rmse"]
    val_rmse = eval_history["val_rmse"]
    epochs = range(1, len(train_rmse) + 1)

    best_epoch = int(np.argmin(val_rmse)) + 1
    best_val = val_rmse[best_epoch - 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_rmse, label="Train RMSE", alpha=0.8)
    ax.plot(epochs, val_rmse, label="Val RMSE", alpha=0.8)
    ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.6,
               label=f"Best epoch={best_epoch}")
    ax.scatter([best_epoch], [best_val], color="red", zorder=5, s=60)
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("RMSE")
    ax.set_title("Training Loss Curve (Early Stopping)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved loss_curve.png (best epoch={best_epoch}, val RMSE={best_val:.6f})")


def plot_true_vs_pred(test_df: pd.DataFrame, report: dict, plots_dir: Path) -> None:
    """Scatter plot of true vs predicted with metrics annotation."""
    y_true = test_df["target"].values
    y_pred = test_df["predicted_return"].values

    metrics = report.get("test", {})

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.25, s=10, edgecolors="none")

    lim_min = min(y_true.min(), y_pred.min()) * 1.1
    lim_max = max(y_true.max(), y_pred.max()) * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", alpha=0.6, label="y=x")

    text = (
        f"R\u00b2 = {metrics.get('r2', 0):.4f}\n"
        f"RMSE = {metrics.get('rmse', 0):.6f}\n"
        f"MAE = {metrics.get('mae', 0):.6f}\n"
        f"DirAcc = {metrics.get('directional_accuracy', 0):.4f}\n"
        f"N = {metrics.get('n_samples', 0)}"
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("True Log Return")
    ax.set_ylabel("Predicted Log Return")
    ax.set_title("Test Set: True vs Predicted (Log Return)")
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "true_vs_pred_test.png", dpi=150)
    plt.close(fig)
    print("  Saved true_vs_pred_test.png")


def plot_residuals(test_df: pd.DataFrame, plots_dir: Path) -> None:
    """Residual histogram and Q-Q plot."""
    residuals = test_df["target"].values - test_df["predicted_return"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(residuals, bins=80, density=True, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Residual (True - Predicted)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.axvline(0, color="red", linestyle="--", alpha=0.6)
    mu, sigma = np.mean(residuals), np.std(residuals)
    ax.text(0.95, 0.95, f"\u03bc={mu:.6f}\n\u03c3={sigma:.6f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            horizontalalignment="right", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sp_stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot (Normal)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "residual_dist.png", dpi=150)
    plt.close(fig)
    print("  Saved residual_dist.png")


def plot_feature_importance(bundle: dict, plots_dir: Path) -> None:
    """All feature importances by gain (adapts to actual feature count)."""
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    n_features = len(feature_names)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig_height = max(4, n_features * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    names = [feature_names[i] for i in indices]
    values = importances[indices]
    ax.barh(range(len(names)), values[::-1], align="center")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Importance (Gain)")
    ax.set_title(f"Feature Importances ({n_features} features)")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved feature_importance.png ({n_features} features)")


def plot_pred_timeseries(test_df: pd.DataFrame, plots_dir: Path) -> None:
    """True vs predicted log return over time, faceted by ETF symbol."""
    symbols = sorted(test_df["symbol"].unique())
    n = len(symbols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, sym in zip(axes, symbols):
        sub = test_df[test_df["symbol"] == sym].sort_values("date")
        ax.plot(sub["date"], sub["target"], alpha=0.7, linewidth=0.8, label="True")
        ax.plot(sub["date"], sub["predicted_return"], alpha=0.7, linewidth=0.8, label="Predicted")
        ax.set_ylabel("Log Return")
        ax.set_title(sym, fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Test Set: True vs Predicted Log Return by ETF", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(plots_dir / "pred_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved pred_timeseries.png")


def plot_signal_distribution(test_df: pd.DataFrame, plots_dir: Path) -> None:
    """Distribution of predicted returns colored by signal."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"long": "green", "short": "red", "flat": "gray"}
    for sig, color in colors.items():
        sub = test_df[test_df["signal"] == sig]
        if len(sub) == 0:
            continue
        ax.hist(sub["predicted_return"], bins=60, alpha=0.6, color=color,
                label=f"{sig.upper()} (n={len(sub)})", edgecolor="black", linewidth=0.2)

    ax.set_xlabel("Predicted Log Return")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Distribution by Signal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "signal_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved signal_distribution.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diagnostic plots")
    parser.add_argument("--outputs-dir", type=Path, default=None,
                        help="Versioned output dir (e.g. outputs/xgboost_7ft_t70_v15_esp10)")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Path to .joblib model bundle")
    args = parser.parse_args()

    # Auto-detect latest versioned output if not specified
    if args.outputs_dir is None or args.model_path is None:
        outputs_root = PROJECT_ROOT / "outputs"
        version_dirs = sorted(
            [d for d in outputs_root.iterdir()
             if d.is_dir() and d.name.startswith("xgboost_")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not version_dirs:
            print("No versioned output directories found under outputs/. "
                  "Run train_xgboost.py first.")
            sys.exit(1)
        latest = version_dirs[0]
        if args.outputs_dir is None:
            args.outputs_dir = latest
        if args.model_path is None:
            args.model_path = PROJECT_ROOT / "models" / f"{latest.name}.joblib"

    plots_dir = args.outputs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading artifacts from {args.outputs_dir.name}...")
    bundle = joblib.load(args.model_path)

    test_df = pd.read_csv(args.outputs_dir / "test_predictions.csv", parse_dates=["date"])
    with open(args.outputs_dir / "evaluation_report.json") as f:
        report = json.load(f)
    with open(args.outputs_dir / "eval_history.json") as f:
        eval_history = json.load(f)

    print(f"Generating plots to {plots_dir}/")

    plot_loss_curve(eval_history, plots_dir)
    plot_true_vs_pred(test_df, report, plots_dir)
    plot_residuals(test_df, plots_dir)
    plot_feature_importance(bundle, plots_dir)
    plot_pred_timeseries(test_df, plots_dir)
    plot_signal_distribution(test_df, plots_dir)

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
